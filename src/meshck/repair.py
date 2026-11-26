"""Automatic repair of triangle meshes.

``repair`` runs a fixed sequence of steps, each only if the analysis finds the
problem it addresses:

1. Remove collapsed triangles (repeated vertex) and duplicate triangles.
2. Flip misoriented triangles.
3. Split non-manifold edges so every body becomes its own shell.
4. Rebuild shells that are still open with pymeshfix, one shell at a time;
   leftovers that enclose no volume are removed.
5. Merge overlapping shells, and shells touching over an area, into one
   solid with a boolean union.
6. Optionally fill cavities: remove every closed shell enclosed by another.

Separate shells are kept, and so are cavities unless filling is requested.
Self-intersecting shells are not repaired.
"""

from dataclasses import dataclass
from functools import reduce

import numpy as np
import pymeshfix
import rtree
import trimesh
from manifold3d import Manifold, Mesh64
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshck.analysis import FLAT_VOLUME_RTOL, Analysis, adjacency, analyze, edge_ids, inside

# Surface samples per changed original triangle for the deviation estimate.
_SAMPLES_PER_TRIANGLE = 4

# At most this many changed triangles are sampled for the deviation estimate,
# and points are measured in chunks of _DEVIATION_CHUNK (bounds memory use).
_MAX_DEVIATION_TRIANGLES = 20_000
_DEVIATION_CHUNK = 20_000

# Triangles with the nearest bounding boxes checked per deviation sample.
_DISTANCE_CANDIDATES = 16

# Deviation samples tested at once for lying inside the repaired solid.
_INSIDE_BATCH = 64

# An open shell thinner than this fraction of its size counts as flat.
_FLAT_RTOL = 1e-6

# Vertices pymeshfix moved by less than this fraction of the part size are
# snapped back to their original position.
_SNAP_RTOL = 1e-6


@dataclass
class RepairResult:
    """Result of ``repair``.

    Attributes:
        vertices: ``(n, 3)`` float array of the repaired mesh.
        faces: ``(m, 3)`` int array of the repaired mesh.
        analysis: Analysis of the repaired mesh.
        steps: Human-readable description of each step that changed the mesh.
        deviation: Estimated largest distance in mm from the original surface
            to the repaired one. Surface added by the repair (filled holes)
            and surface now inside the solid (merged overlaps) are not
            counted; removed shells are.
    """

    vertices: np.ndarray
    faces: np.ndarray
    analysis: Analysis
    steps: list[str]
    deviation: float


def repair(vertices: np.ndarray, faces: np.ndarray, analysis: Analysis | None = None,
           fill_cavities: bool = False) -> RepairResult:
    """Repair a mesh so that it is closed, oriented and free of overlaps.

    Args:
        vertices: ``(n, 3)`` float array of vertex positions in mm, with
            duplicate vertices already merged.
        faces: ``(m, 3)`` int array of vertex indices per triangle.
        analysis: ``analyze(vertices, faces)``, if already computed.
        fill_cavities: Also remove all shells enclosed by another shell, so
            that only the outer surfaces remain.

    Returns:
        The repaired mesh and a log of what was done. If nothing needed
        repair, the mesh is returned unchanged and ``steps`` is empty.
    """
    v0 = np.asarray(vertices, dtype=np.float64)
    f0 = np.asarray(faces, dtype=np.int64)
    v, f, steps = v0, f0, []
    result = analysis or analyze(v0, f0)

    pipeline = [_clean, _orient, _split_nonmanifold, _rebuild_shells, _union_overlapping, _orient]
    if fill_cavities:
        pipeline.append(_fill_cavities)
    for step in pipeline:
        changed = step(v, f, result)
        if changed is not None:
            v, f, message = changed
            steps.append(message)
            result = analyze(v, f)

    if steps:
        v, f = _compact(v, f)
        result = analyze(v, f)
    deviation = _deviation(v0, f0, v, f) if steps else 0.0
    return RepairResult(vertices=v, faces=f, analysis=result, steps=steps, deviation=deviation)


# Each step returns None if it has nothing to do, or the changed mesh and a
# message describing the change.
Step = tuple[np.ndarray, np.ndarray, str] | None


def _clean(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Drop collapsed triangles and duplicates.

    Two copies of a triangle with opposite winding cancel each other (they
    are the shared wall of two touching bodies); copies with the same winding
    are reduced to one.
    """
    collapsed = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    kept = faces[~collapsed]

    # Rotate each triangle so its smallest index comes first; the order of
    # the other two then encodes the winding.
    first = np.argmin(kept, axis=1)
    rot = np.stack([kept[np.arange(len(kept)), (first + k) % 3] for k in range(3)], axis=1)
    positive = rot[:, 1] < rot[:, 2]
    _, group, counts = np.unique(np.sort(kept, axis=1), axis=0, return_inverse=True, return_counts=True)
    group = group.ravel()

    keep = np.ones(len(kept), dtype=bool)
    for g in np.flatnonzero(counts > 1):
        members = np.flatnonzero(group == g)
        pos, neg = members[positive[members]], members[~positive[members]]
        keep[members] = False
        if len(pos) != len(neg):
            keep[(pos if len(pos) > len(neg) else neg)[0]] = True

    parts = []
    if collapsed.any():
        parts.append(f"{int(collapsed.sum())} collapsed")
    if not keep.all():
        parts.append(f"{int((~keep).sum())} duplicate")
    if not parts:
        return None
    return vertices, kept[keep], f"removed {' and '.join(parts)} triangles"


def _orient(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Flip all triangles the analysis marks as misoriented."""
    n = int(result.misoriented.sum())
    if n == 0:
        return None
    faces = faces.copy()
    faces[result.misoriented] = faces[result.misoriented][:, ::-1]
    return vertices, faces, f"flipped {n} misoriented triangles"


def _split_nonmanifold(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Give each manifold patch its own vertices.

    Patches are triangles connected through edges used exactly twice. At an
    edge shared by more than two triangles the patches separate, so two
    bodies touching along an edge become two closed shells.
    """
    if result.nonmanifold_edges == 0:
        return None
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edge_face = np.repeat(np.arange(len(faces)), 3)
    edge_id, counts = edge_ids(directed, len(vertices))
    order = np.argsort(edge_id, kind="stable")
    two = order[counts[edge_id[order]] == 2].reshape(-1, 2)
    _, patch = connected_components(
        adjacency(edge_face[two[:, 0]], edge_face[two[:, 1]], len(faces)), directed=False)

    key = patch[:, None].astype(np.int64) * len(vertices) + faces
    unique, inverse = np.unique(key.ravel(), return_inverse=True)
    return (vertices[unique % len(vertices)], inverse.reshape(-1, 3),
            f"split {result.nonmanifold_edges} non-manifold edges")


def _rebuild_shells(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Run pymeshfix on every open shell and remove flat closed shells.

    If pymeshfix returns nothing or a flat leftover, the shell is removed if
    it is flat itself (stray triangles, flat sheets) and otherwise kept
    unchanged. Closed shells that are flat enclose no volume and are removed
    too. Nothing is removed if that would leave no triangles at all.
    """
    flat_closed = [s for s in result.shells if s.flat]
    broken = [s for s in result.shells if not s.closed]
    if not broken and not flat_closed:
        return None

    keep = np.ones(len(faces), dtype=bool)
    parts_v, parts_f, offset = [vertices], [], len(vertices)
    rebuilt, removed = 0, list(flat_closed)
    removed_tris = sum(len(s.faces) for s in flat_closed)
    for shell in broken:
        used, local = np.unique(faces[shell.faces], return_inverse=True)
        local = local.reshape(-1, 3)
        fix = pymeshfix.MeshFix(vertices[used], local)
        fix.repair(joincomp=False, remove_smallest_components=False)
        # pymeshfix returns nothing when it gives up; then only a flat shell
        # is known to be junk. A result without any solid shows the shell
        # had no thickness (a fin or sheet).
        no_solid = len(fix.faces) > 0 and not any(
            s.closed and not s.flat for s in analyze(fix.points, fix.faces).shells)
        if len(fix.faces) == 0 or no_solid:
            if no_solid or _is_stray(vertices[used], local):
                removed.append(shell)
                removed_tris += len(shell.faces)
            continue
        keep[shell.faces] = False
        parts_v.append(_snap(np.asarray(fix.points, dtype=np.float64), vertices[used]))
        parts_f.append(np.asarray(fix.faces, dtype=np.int64) + offset)
        offset += len(fix.points)
        rebuilt += 1

    if removed and removed_tris == keep.sum() and not rebuilt:
        removed, removed_tris = [], 0
    for shell in removed:
        keep[shell.faces] = False

    messages = []
    if rebuilt:
        messages.append(f"rebuilt {rebuilt} open shell(s) with pymeshfix")
    if removed:
        messages.append(f"removed {len(removed)} loose surface piece(s) enclosing no volume ({removed_tris} triangles)")
    if not messages:
        return None
    return np.vstack(parts_v), np.vstack([faces[keep], *parts_f]), "; ".join(messages)


def _snap(points: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Move points back onto original vertices they differ from only by rounding.

    pymeshfix shifts vertices it keeps by up to about 1e-7 of the part size;
    snapping restores their exact positions so unchanged triangles stay
    unchanged.
    """
    tol = _SNAP_RTOL * max(float(np.ptp(original, axis=0).max()), 1.0)
    dist, nearest = cKDTree(original).query(points, distance_upper_bound=tol)
    near = np.isfinite(dist)
    points = points.copy()
    points[near] = original[nearest[near]]
    return points


def _is_stray(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Return whether a shell is open and flat, so it cannot become a solid.

    Such leftovers are single triangles and flat sheets: all their corners
    lie in one plane.
    """
    if analyze(vertices, faces).boundary_edges == 0:
        return False
    return _is_flat(np.asarray(vertices, dtype=np.float64)[np.unique(faces)])


def _is_flat(points: np.ndarray) -> bool:
    """Return whether all points lie in one plane, up to ``_FLAT_RTOL`` of their extent."""
    centered = points - points.mean(axis=0)
    thinnest = np.linalg.svd(centered, full_matrices=False)[2][-1]
    thickness = float(np.ptp(centered @ thinnest))
    return thickness <= _FLAT_RTOL * max(float(np.ptp(points, axis=0).max()), 1.0)


def _union_overlapping(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Replace each group of overlapping or touching shells by their boolean union."""
    if not result.overlapping and not result.touching:
        return None

    pairs = np.array(result.overlapping + result.touching)
    _, group = connected_components(adjacency(pairs[:, 0], pairs[:, 1], len(result.shells)),
                                    directed=False)
    involved = np.unique(pairs)

    keep = np.ones(len(faces), dtype=bool)
    parts_v, parts_f, offset = [vertices], [], len(vertices)
    merged_shells, merged_groups = 0, 0
    for g in np.unique(group[involved]):
        members = [i for i in involved if group[i] == g]
        solids = [_solid(vertices, faces, result.shells[i].faces) for i in members]
        if any(s is None for s in solids):
            continue
        # The union can contain zero-volume slivers where solids touched.
        pieces = [p for p in reduce(lambda a, b: a + b, solids).decompose() if not _encloses_nothing(p)]
        union = Manifold.compose(pieces).to_mesh64()
        for i in members:
            keep[result.shells[i].faces] = False
        parts_v.append(np.asarray(union.vert_properties[:, :3], dtype=np.float64))
        parts_f.append(np.asarray(union.tri_verts, dtype=np.int64) + offset)
        offset += len(union.vert_properties)
        merged_shells += len(members)
        merged_groups += 1

    if not merged_groups:
        return None
    kind = ("overlapping" if not result.touching else
            "touching" if not result.overlapping else "overlapping or touching")
    return (np.vstack(parts_v), np.vstack([faces[keep], *parts_f]),
            f"merged {merged_shells} {kind} shells into {merged_groups}")


def _encloses_nothing(solid: Manifold) -> bool:
    """Return whether a solid has (almost) no volume for its size."""
    lo, hi = np.array(solid.bounding_box()).reshape(2, 3)
    return abs(solid.volume()) <= FLAT_VOLUME_RTOL * max(float((hi - lo).max()), 1.0) ** 3


def _fill_cavities(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Remove every closed shell enclosed by another: cavities and anything inside them."""
    inner = [s for s in result.shells if s.closed and s.depth > 0]
    if not inner:
        return None
    keep = np.ones(len(faces), dtype=bool)
    for shell in inner:
        keep[shell.faces] = False
    return vertices, faces[keep], f"filled cavities: removed {len(inner)} inner shell(s)"


def _solid(vertices: np.ndarray, faces: np.ndarray, shell_faces: np.ndarray) -> Manifold | None:
    """Return a shell as an outward-oriented solid, or None if not manifold."""
    used, local = np.unique(faces[shell_faces], return_inverse=True)
    tris = local.reshape(-1, 3)
    solid = Manifold(Mesh64(vert_properties=vertices[used], tri_verts=tris.astype(np.uint64)))
    if solid.status().name != "NoError":
        return None
    if solid.volume() <= 0:
        solid = Manifold(Mesh64(vert_properties=vertices[used],
                                tri_verts=tris[:, ::-1].astype(np.uint64)))
    return solid


def _compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no triangle refers to."""
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def _deviation(v0: np.ndarray, f0: np.ndarray, v1: np.ndarray, f1: np.ndarray) -> float:
    """Estimate how far the original surface moved during repair.

    Original triangles that reappear unchanged (same corner positions, in
    any order or winding) have distance zero. The others are sampled and
    measured against the repaired surface; samples that end up inside the
    repaired solid belong to removed inner walls and are ignored. If very
    many triangles changed, a random subset of them is sampled.
    """
    if len(f1) == 0:
        return float("inf")
    changed = np.flatnonzero(~np.isin(_triangle_keys(v0, f0), _triangle_keys(v1, f1)))
    if len(changed) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    if len(changed) > _MAX_DEVIATION_TRIANGLES:
        changed = rng.choice(changed, _MAX_DEVIATION_TRIANGLES, replace=False)

    tri = v0[f0[changed]]
    weights = rng.dirichlet((1, 1, 1), size=(len(tri), _SAMPLES_PER_TRIANGLE))
    points = np.vstack([np.einsum("tsk,tkd->tsd", weights, tri).reshape(-1, 3), tri.reshape(-1, 3)])
    dist = _surface_distance(v1, f1, points)
    far = np.flatnonzero(dist > 1e-6)
    if len(far) == 0:
        return 0.0
    if not trimesh.Trimesh(v1, f1, process=False).is_watertight:
        return float(dist[far].max())
    # The answer is the farthest sample outside the solid: test the farthest
    # first and stop at the first one outside.
    far = far[np.argsort(-dist[far])]
    for start in range(0, len(far), _INSIDE_BATCH):
        batch = far[start:start + _INSIDE_BATCH]
        outside = ~inside(v1, f1, points[batch])
        if outside.any():
            return float(dist[batch[np.argmax(outside)]])
    return 0.0


def _surface_distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return the distance from each point to the nearest of a few candidate triangles.

    The candidates are the triangles with the nearest bounding boxes, so
    memory stays proportional to the number of points. The result is exact
    unless the nearest triangle is not among them, in which case it is
    slightly too large.
    """
    tri = vertices[faces]
    props = rtree.index.Property()
    props.dimension = 3
    tree = rtree.index.Index((np.arange(len(tri), dtype=np.int64), tri.min(axis=1), tri.max(axis=1)),
                             properties=props)
    # Degenerate triangles give NaN; the nearest vertex bounds every point.
    dist, _ = cKDTree(vertices).query(points)
    for start in range(0, len(points), _DEVIATION_CHUNK):
        chunk = points[start:start + _DEVIATION_CHUNK]
        ids, counts = tree.nearest_v(chunk, chunk, num_results=_DISTANCE_CANDIDATES, strict=True)
        owner = np.repeat(np.arange(len(chunk)), counts.astype(np.int64))
        with np.errstate(divide="ignore", invalid="ignore"):
            closest = trimesh.triangles.closest_point(tri[ids.astype(np.int64)], chunk[owner])
        candidate = np.linalg.norm(closest - chunk[owner], axis=1)
        np.fmin.at(dist, start + owner, candidate)
    return dist


def _triangle_keys(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one comparable key per triangle: its sorted corner positions."""
    corners = np.round(vertices[faces], 9) + 0.0  # + 0.0 turns -0.0 into 0.0
    order = np.lexsort(corners.transpose(2, 0, 1)[::-1], axis=-1)
    corners = np.ascontiguousarray(np.take_along_axis(corners, order[..., None], axis=1))
    return corners.reshape(len(faces), 9).view(np.dtype((np.void, 72))).ravel()

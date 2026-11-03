"""Automatic repair of triangle meshes.

``repair`` runs a fixed sequence of steps, each only if the analysis finds the
problem it addresses:

1. Remove collapsed triangles (repeated vertex) and duplicate triangles.
2. Flip misoriented triangles.
3. Split non-manifold edges so every body becomes its own shell.
4. Rebuild shells that are still open or degenerate with pymeshfix, one
   shell at a time; leftovers that enclose no volume are removed.
5. Merge overlapping shells into one solid with a boolean union.

Separate shells and cavities are kept as they are. Self-intersecting shells
are not repaired.
"""

from dataclasses import dataclass
from functools import reduce

import numpy as np
import pymeshfix
import trimesh
from manifold3d import Manifold, Mesh64
from scipy.sparse.csgraph import connected_components

from meshck.analysis import Analysis, adjacency, analyze, edge_ids

# Surface samples per changed original triangle for the deviation estimate.
_SAMPLES_PER_TRIANGLE = 4


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


def repair(vertices: np.ndarray, faces: np.ndarray, analysis: Analysis | None = None) -> RepairResult:
    """Repair a mesh so that it is closed, oriented and free of overlaps.

    Args:
        vertices: ``(n, 3)`` float array of vertex positions in mm, with
            duplicate vertices already merged.
        faces: ``(m, 3)`` int array of vertex indices per triangle.
        analysis: ``analyze(vertices, faces)``, if already computed.

    Returns:
        The repaired mesh and a log of what was done. If nothing needed
        repair, the mesh is returned unchanged and ``steps`` is empty.
    """
    v0 = np.asarray(vertices, dtype=np.float64)
    f0 = np.asarray(faces, dtype=np.int64)
    v, f, steps = v0, f0, []
    result = analysis or analyze(v0, f0)

    for step in (_clean, _orient, _split_nonmanifold, _rebuild_shells, _union_overlapping, _orient):
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
    """Run pymeshfix on every shell that is open or has degenerate triangles."""
    broken = [s for s in result.shells if not s.closed or result.degenerate[s.faces].any()]
    if not broken:
        return None

    keep = np.ones(len(faces), dtype=bool)
    parts_v, parts_f, offset = [vertices], [], len(vertices)
    rebuilt, removed, removed_tris = 0, 0, 0
    for shell in broken:
        keep[shell.faces] = False
        used, local = np.unique(faces[shell.faces], return_inverse=True)
        fix = pymeshfix.MeshFix(vertices[used], local.reshape(-1, 3))
        fix.repair(joincomp=False, remove_smallest_components=False)
        if len(fix.faces) == 0 or _is_stray(fix.points, fix.faces):
            removed += 1
            removed_tris += len(shell.faces)
            continue
        parts_v.append(np.asarray(fix.points, dtype=np.float64))
        parts_f.append(np.asarray(fix.faces, dtype=np.int64) + offset)
        offset += len(fix.points)
        rebuilt += 1

    messages = []
    if rebuilt:
        messages.append(f"rebuilt {rebuilt} open or degenerate shell(s) with pymeshfix")
    if removed:
        messages.append(f"removed {removed} shell(s) enclosing no volume ({removed_tris} triangles)")
    return np.vstack(parts_v), np.vstack([faces[keep], *parts_f]), "; ".join(messages)


def _is_stray(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Return whether a shell is still open and encloses no volume.

    Such leftovers (single triangles, flat sheets) cannot become solids.
    """
    if analyze(vertices, faces).boundary_edges == 0:
        return False
    tri = np.asarray(vertices, dtype=np.float64)[faces]
    volume = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
    return volume <= 1e-9 * max(float(np.ptp(tri.reshape(-1, 3), axis=0).max()), 1.0) ** 3


def _union_overlapping(vertices: np.ndarray, faces: np.ndarray, result: Analysis) -> Step:
    """Replace each group of overlapping shells by their boolean union."""
    if not result.overlapping:
        return None

    pairs = np.array(result.overlapping)
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
        union = reduce(lambda a, b: a + b, solids).to_mesh64()
        for i in members:
            keep[result.shells[i].faces] = False
        parts_v.append(np.asarray(union.vert_properties[:, :3], dtype=np.float64))
        parts_f.append(np.asarray(union.tri_verts, dtype=np.int64) + offset)
        offset += len(union.vert_properties)
        merged_shells += len(members)
        merged_groups += 1

    if not merged_groups:
        return None
    return (np.vstack(parts_v), np.vstack([faces[keep], *parts_f]),
            f"merged {merged_shells} overlapping shells into {merged_groups}")


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
    repaired solid belong to removed inner walls and are ignored.
    """
    if len(f1) == 0:
        return float("inf")
    changed = ~np.isin(_triangle_keys(v0, f0), _triangle_keys(v1, f1))
    if not changed.any():
        return 0.0

    tri = v0[f0[changed]]
    weights = np.random.default_rng(0).dirichlet((1, 1, 1), size=(len(tri), _SAMPLES_PER_TRIANGLE))
    points = np.vstack([np.einsum("tsk,tkd->tsd", weights, tri).reshape(-1, 3), tri.reshape(-1, 3)])
    repaired = trimesh.Trimesh(v1, f1, process=False)
    _, dist, _ = trimesh.proximity.closest_point(repaired, points)

    far = dist > 1e-6
    if far.any() and repaired.is_watertight:
        far[far] = ~repaired.contains(points[far])
    return float(dist[far].max()) if far.any() else 0.0


def _triangle_keys(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one comparable key per triangle: its sorted corner positions."""
    corners = np.round(vertices[faces], 9) + 0.0  # + 0.0 turns -0.0 into 0.0
    order = np.lexsort(corners.transpose(2, 0, 1)[::-1], axis=-1)
    corners = np.ascontiguousarray(np.take_along_axis(corners, order[..., None], axis=1))
    return corners.reshape(len(faces), 9).view(np.dtype((np.void, 72))).ravel()

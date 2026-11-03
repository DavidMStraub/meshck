"""Topological and geometric analysis of a triangle mesh.

Everything here only measures the mesh; ``meshck.check`` turns the
measurements into findings.

Terms:
    boundary edge: An edge used by exactly one triangle (the mesh is open).
    hole: A closed loop of boundary edges.
    non-manifold edge: An edge used by more than two triangles.
    shell: A set of triangles connected through shared edges.
    misoriented triangle: A triangle whose normal points the wrong way
        relative to its neighbours or, for the whole shell, relative to the
        solid it encloses. A shell nested at odd depth that mostly points
        inwards is a cavity; one that mostly points outwards is a solid
        inside a solid and counts as overlapping its enclosing shell.
    self-intersecting triangle: A triangle crossing another triangle of the
        same closed shell.
"""

from dataclasses import dataclass, field

import numpy as np
import rtree
from manifold3d import Manifold, Mesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Relative tolerance for volumes from boolean intersections.
_VOLUME_RTOL = 1e-4

# Triangle pairs tested for crossings at once (bounds memory use).
_PAIR_CHUNK = 500_000

# Crossings shallower than this (mm) are rounding noise, not defects.
_CROSSING_DEPTH = 1e-3


@dataclass
class Shell:
    """Measurements of one shell.

    Attributes:
        faces: Indices of the shell's triangles in the mesh.
        closed: True if the shell has no boundary or non-manifold edges.
        volume: Enclosed volume in mm³ once misoriented triangles are fixed;
            negative for cavities. Only meaningful if ``closed``.
        depth: Number of other shells enclosing this one.
        parent: Index of the innermost shell enclosing this one, or None.
    """

    faces: np.ndarray
    closed: bool
    volume: float = 0.0
    depth: int = 0
    parent: int | None = None


@dataclass
class Analysis:
    """Result of ``analyze``.

    Attributes:
        triangles: Number of triangles.
        edges: Number of distinct edges.
        boundary_edges: Edges used by exactly one triangle.
        holes: Loops formed by the boundary edges.
        nonmanifold_edges: Edges used by more than two triangles.
        degenerate: Boolean mask over all triangles with (near) zero area.
        misoriented: Boolean mask over all triangles.
        self_intersecting: Boolean mask over all triangles crossing another
            triangle of the same closed shell.
        shells: One entry per shell.
        overlapping: Index pairs into ``shells`` of closed shells that
            partially intersect each other, or of a solid and the solid
            enclosing it.
        size: Axis-aligned extent in mm along x, y and z.
        volume: Total enclosed volume in mm³ (cavities subtracted), summed
            over closed shells.
    """

    triangles: int
    edges: int
    boundary_edges: int
    holes: int
    nonmanifold_edges: int
    degenerate: np.ndarray
    misoriented: np.ndarray
    self_intersecting: np.ndarray
    shells: list[Shell]
    overlapping: list[tuple[int, int]] = field(default_factory=list)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume: float = 0.0


def analyze(vertices: np.ndarray, faces: np.ndarray) -> Analysis:
    """Measure topology, orientation and shell structure of a mesh.

    Args:
        vertices: ``(n, 3)`` float array of vertex positions in mm, with
            duplicate vertices already merged.
        faces: ``(m, 3)`` int array of vertex indices per triangle.

    Returns:
        The measurements.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_faces = len(faces)

    # Triangles with repeated vertex indices have no proper edges; they are
    # only counted as degenerate and left out of the edge topology.
    collapsed = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    scale = float(np.ptp(vertices, axis=0).max()) if len(vertices) else 0.0
    degenerate = collapsed | (area <= 1e-12 * max(scale, 1.0) ** 2)

    valid = np.flatnonzero(~collapsed)
    directed = faces[valid][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edge_face = np.repeat(valid, 3)
    edge_id, counts = edge_ids(directed, len(vertices))
    per_edge = counts[edge_id]

    boundary_mask = per_edge == 1
    holes = _count_loops(directed[boundary_mask])

    # Pairs of triangles across each edge used exactly twice. "same" means
    # both traverse the edge in the same direction, i.e. inconsistent winding.
    order = np.argsort(edge_id, kind="stable")
    two = order[per_edge[order] == 2].reshape(-1, 2)
    fa, fb = edge_face[two[:, 0]], edge_face[two[:, 1]]
    same = np.all(directed[two[:, 0]] == directed[two[:, 1]], axis=1)

    # Shells: triangles connected through any shared edge (including
    # non-manifold ones, chained along each edge's triangle list).
    many = order[per_edge[order] > 2]
    many_edges = edge_id[many]
    chain = many_edges[1:] == many_edges[:-1]
    sa = np.concatenate([fa, edge_face[many[:-1][chain]]])
    sb = np.concatenate([fb, edge_face[many[1:][chain]]])
    _, shell_of = connected_components(adjacency(sa, sb, n_faces), directed=False)

    # Orientation classes: a signed graph over (face, +) and (face, -) nodes.
    # Consistent neighbours share their sign, inconsistent ones swap it; the
    # component of (face, +) then names the face's orientation class.
    cons, flip = ~same, same
    src = np.concatenate([fa[cons], fa[cons] + n_faces, fa[flip], fa[flip] + n_faces])
    dst = np.concatenate([fb[cons], fb[cons] + n_faces, fb[flip] + n_faces, fb[flip]])
    _, signed_comp = connected_components(adjacency(src, dst, 2 * n_faces), directed=False)
    face_class = signed_comp[:n_faces]

    # Patches: triangles connected through edges used exactly twice. A patch
    # has one orientation of its own; a shell joined only through non-manifold
    # edges (bodies touching along an edge) consists of several patches.
    _, patch_of = connected_components(adjacency(fa, fb, n_faces), directed=False)

    # Closed means every edge is used exactly twice, within the shell or patch.
    open_faces = np.zeros(n_faces, dtype=bool)
    open_faces[edge_face[per_edge != 2]] = True
    _, pe_id, pe_counts = np.unique(patch_of[edge_face].astype(np.int64) * len(counts) + edge_id,
                                    return_inverse=True, return_counts=True)
    open_in_patch = np.zeros(n_faces, dtype=bool)
    open_in_patch[edge_face[pe_counts[pe_id.ravel()] != 2]] = True

    # Each patch's first face defines its reference orientation; faces of the
    # other class get flipped. vol_ref is the volume after that flip.
    contrib = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    flipped = np.zeros(n_faces, dtype=bool)
    patches = _groups(patch_of, valid)
    vol_ref = np.zeros(len(patches))
    for p, idx in enumerate(patches):
        in_ref = face_class[idx] == face_class[idx[0]]
        flipped[idx[~in_ref]] = True
        vol_ref[p] = contrib[idx][in_ref].sum() - contrib[idx][~in_ref].sum()

    patch_index = {int(patch_of[idx[0]]): p for p, idx in enumerate(patches)}
    shells: list[Shell] = []
    shell_index = {}
    for idx in _groups(shell_of, valid):
        closed = not open_faces[idx].any()
        # A closed shell is a single patch.
        volume = float(vol_ref[patch_index[int(patch_of[idx[0]])]]) if closed else 0.0
        shell_index[int(shell_of[idx[0]])] = len(shells)
        shells.append(Shell(faces=idx, closed=closed, volume=volume))

    oriented = faces.copy()
    oriented[flipped] = oriented[flipped][:, ::-1]
    overlapping = _nest(vertices, oriented, shells)

    # Decide per patch which orientation class is correct. Closed patches at
    # even depth must enclose a positive volume. At odd depth the majority of
    # triangles decides: pointing inwards makes a cavity, pointing outwards
    # a solid inside a solid. Open patches follow the majority.
    misoriented = np.zeros(n_faces, dtype=bool)
    for p, idx in enumerate(patches):
        s = shell_index[int(shell_of[idx[0]])]
        shell = shells[s]
        majority_is_ref = (~flipped[idx]).sum() >= flipped[idx].sum()
        if open_in_patch[idx].any():
            ref_correct = majority_is_ref
        elif shell.depth % 2 == 0:
            ref_correct = vol_ref[p] > 0
        else:
            solid = (vol_ref[p] > 0) == majority_is_ref
            ref_correct = (vol_ref[p] > 0) == solid
            if solid and shell.parent is not None:
                overlapping.append((min(s, shell.parent), max(s, shell.parent)))
        misoriented[idx] = flipped[idx] if ref_correct else ~flipped[idx]
        if shell.closed and not ref_correct:
            shell.volume = -shell.volume

    closed_volume = sum(s.volume for s in shells if s.closed)
    x, y, z = (float(v) for v in np.ptp(vertices[faces.ravel()], axis=0)) if n_faces else (0.0, 0.0, 0.0)
    return Analysis(
        triangles=n_faces,
        edges=len(counts),
        boundary_edges=int(boundary_mask.sum()),
        holes=holes,
        nonmanifold_edges=int((counts > 2).sum()),
        degenerate=degenerate,
        misoriented=misoriented,
        self_intersecting=_self_intersecting(vertices, faces, shells),
        shells=shells,
        overlapping=overlapping,
        size=(x, y, z),
        volume=float(closed_volume),
    )


def edge_ids(directed: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Identify undirected edges.

    Args:
        directed: ``(k, 2)`` int array of directed edges.
        n_vertices: Number of vertices (upper bound of the indices).

    Returns:
        For each directed edge the id of its undirected edge, and the number
        of directed edges per id.
    """
    lo, hi = np.minimum(directed[:, 0], directed[:, 1]), np.maximum(directed[:, 0], directed[:, 1])
    _, edge_id, counts = np.unique(lo * np.int64(n_vertices) + hi,
                                   return_inverse=True, return_counts=True)
    return edge_id.ravel(), counts


def adjacency(src: np.ndarray, dst: np.ndarray, n: int) -> coo_matrix:
    """Return an ``n × n`` sparse adjacency matrix with the given edges."""
    return coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))


def _groups(labels: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    """Split ``idx`` into arrays of equal ``labels[idx]``, each sorted."""
    if len(idx) == 0:
        return []
    order = idx[np.argsort(labels[idx], kind="stable")]
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return np.split(order, cuts)


def _count_loops(edges: np.ndarray) -> int:
    """Count connected loops formed by a set of (boundary) edges."""
    if len(edges) == 0:
        return 0
    nodes, inv = np.unique(edges, return_inverse=True)
    inv = inv.reshape(-1, 2)
    n, _ = connected_components(adjacency(inv[:, 0], inv[:, 1], len(nodes)), directed=False)
    return int(n)


def _nest(vertices: np.ndarray, faces: np.ndarray, shells: list[Shell]) -> list[tuple[int, int]]:
    """Set ``depth`` and ``parent`` of closed shells and return partially overlapping pairs.

    ``faces`` must be consistently oriented within each shell.
    """
    closed = [i for i, s in enumerate(shells) if s.closed]
    if len(closed) < 2:
        return []

    solids: dict[int, Manifold] = {}
    lo, hi, vol = {}, {}, {}
    for i in closed:
        f = faces[shells[i].faces]
        if shells[i].volume < 0:
            f = f[:, ::-1]
        used, local = np.unique(f, return_inverse=True)
        mesh = Mesh(vert_properties=vertices[used].astype(np.float32),
                    tri_verts=local.reshape(-1, 3).astype(np.uint32))
        solid = Manifold(mesh)
        if solid.status().name == "NoError":
            solids[i] = solid
        lo[i], hi[i] = vertices[used].min(axis=0), vertices[used].max(axis=0)
        vol[i] = abs(shells[i].volume)

    overlapping = []
    enclosing: dict[int, list[int]] = {}
    for a_pos, a in enumerate(closed):
        for b in closed[a_pos + 1:]:
            if np.any(hi[a] < lo[b]) or np.any(hi[b] < lo[a]):
                continue
            small, big = (a, b) if vol[a] <= vol[b] else (b, a)
            if a in solids and b in solids:
                common = (solids[a] ^ solids[b]).volume()
                if common <= _VOLUME_RTOL * vol[small]:
                    continue
                if common < (1 - _VOLUME_RTOL) * vol[small]:
                    overlapping.append((a, b))
                    continue
            if _inside(vertices, faces[shells[big].faces], _probe(vertices, faces[shells[small].faces])):
                shells[small].depth += 1
                enclosing.setdefault(small, []).append(big)
    for i, outer in enclosing.items():
        shells[i].parent = max(outer, key=lambda j: shells[j].depth)
    return overlapping


def _self_intersecting(vertices: np.ndarray, faces: np.ndarray, shells: list[Shell]) -> np.ndarray:
    """Return a mask of triangles that cross another triangle of their closed shell.

    Two triangles cross if an edge of one passes through the interior of the
    other and each reaches more than ``_CROSSING_DEPTH`` through the plane of
    the other. Touching at shared vertices or edges and coplanar overlaps do
    not count.
    """
    mask = np.zeros(len(faces), dtype=bool)
    closed = [s.faces for s in shells if s.closed]
    if not closed:
        return mask
    cand = np.concatenate(closed)
    shell_of = np.repeat(np.arange(len(closed)), [len(c) for c in closed])
    tri = vertices[faces[cand]]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    tri = tri - (lo.min(axis=0) + hi.max(axis=0)) / 2
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])
    extent = (hi - lo).max(axis=1)
    # Plane distances are scaled by |normal|; this is the depth threshold.
    depth_tol = np.maximum(_CROSSING_DEPTH * np.linalg.norm(normal, axis=1), 1e-10 * extent**3)

    props = rtree.index.Property()
    props.dimension = 3
    tree = rtree.index.Index((np.arange(len(cand), dtype=np.int64), lo, hi), properties=props)
    ids, counts = tree.intersection_v(lo, hi)
    a, b = np.repeat(np.arange(len(cand)), counts.astype(np.int64)), ids.astype(np.int64)
    keep = (a < b) & (shell_of[a] == shell_of[b])
    a, b = a[keep], b[keep]
    # Neighbours across an edge can only overlap coplanar.
    shared = (faces[cand[a]][:, :, None] == faces[cand[b]][:, None, :]).sum(axis=(1, 2))
    a, b = a[shared < 2], b[shared < 2]

    for start in range(0, len(a), _PAIR_CHUNK):
        pa, pb = a[start:start + _PAIR_CHUNK], b[start:start + _PAIR_CHUNK]
        tol = 1e-10 * np.maximum(extent[pa], extent[pb]) ** 3
        # Crossing triangles straddle each other's plane; this cheap test
        # rejects almost all pairs, including coplanar fans and shallow folds.
        s = (_straddles(tri[pa], normal[pb], offset[pb], depth_tol[pb])
             & _straddles(tri[pb], normal[pa], offset[pa], depth_tol[pa]))
        pa, pb, tol = pa[s], pb[s], tol[s]
        origin = tri[pa, :1]
        t1, t2 = tri[pa] - origin, tri[pb] - origin
        hit = _edges_cross(t1, t2, tol) | _edges_cross(t2, t1, tol)
        mask[cand[pa[hit]]] = True
        mask[cand[pb[hit]]] = True
    return mask


def _straddles(t: np.ndarray, normal: np.ndarray, offset: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Return per pair whether triangle ``t`` has corners strictly on both sides of a plane.

    The plane is the set of points ``x`` with ``normal · x = offset``.
    """
    side = np.einsum("ikj,ij->ik", t, normal) - offset[:, None]
    return (side > tol[:, None]).any(axis=1) & (side < -tol[:, None]).any(axis=1)


def _edges_cross(t1: np.ndarray, t2: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Return per pair whether an edge of ``t1`` strictly crosses triangle ``t2``.

    Args:
        t1: ``(k, 3, 3)`` corners of the first triangle of each pair.
        t2: ``(k, 3, 3)`` corners of the second triangle of each pair.
        tol: ``(k,)`` magnitude below which an orientation counts as zero.
    """
    a, b, c = t2[:, 0], t2[:, 1], t2[:, 2]
    side = [_orient(a, b, c, t1[:, k]) for k in range(3)]
    result = np.zeros(len(t1), dtype=bool)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p, q = t1[:, i], t1[:, j]
        opposite = ((side[i] > tol) & (side[j] < -tol)) | ((side[i] < -tol) & (side[j] > tol))
        o = [_orient(p, q, u, v) for u, v in ((a, b), (b, c), (c, a))]
        inside = ((o[0] > tol) & (o[1] > tol) & (o[2] > tol)) | ((o[0] < -tol) & (o[1] < -tol) & (o[2] < -tol))
        result |= opposite & inside
    return result


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Return six times the signed volume of each tetrahedron ``abcd``."""
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))


def _probe(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return a point on the surface: the centroid of the largest triangle."""
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return tri[np.argmax(area)].mean(axis=0)


def _inside(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> bool:
    """Return whether ``point`` lies inside the closed surface ``faces``.

    Uses the winding number (sum of solid angles, Van Oosterom–Strackee),
    which is ±1 inside and 0 outside regardless of orientation sign.
    """
    a, b, c = (vertices[faces[:, k]] - point for k in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
           + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb)
    winding = np.arctan2(num, den).sum() / (2 * np.pi)
    return abs(winding) > 0.5

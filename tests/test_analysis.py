import numpy as np
import pytest
import trimesh

from meshck.analysis import analyze


def box(size=10.0, center=(0, 0, 0)):
    return trimesh.creation.box(extents=(size, size, size)).apply_translation(center)


def combine(*meshes):
    """Concatenate meshes and merge coincident vertices."""
    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    return mesh.vertices, mesh.faces


def inverted(mesh):
    mesh = mesh.copy()
    mesh.faces = mesh.faces[:, ::-1]
    return mesh


def test_clean_cube():
    result = analyze(*combine(box()))
    assert result.triangles == 12
    assert result.boundary_edges == result.holes == result.nonmanifold_edges == 0
    assert not result.degenerate.any()
    assert not result.misoriented.any()
    assert len(result.shells) == 1 and result.shells[0].closed
    assert result.volume == pytest.approx(1000)
    assert result.size == pytest.approx((10, 10, 10))


def test_open_cube():
    v, f = combine(box())
    result = analyze(v, f[2:])  # drop one side (two triangles)
    assert result.boundary_edges == 4
    assert result.holes == 1
    assert not result.shells[0].closed


def test_two_holes():
    v, f = combine(box())
    normals = trimesh.Trimesh(v, f).face_normals
    keep = ~np.isclose(np.abs(normals[:, 2]), 1)  # drop top and bottom
    result = analyze(v, f[keep])
    assert result.holes == 2
    assert result.boundary_edges == 8


def test_single_flipped_triangle():
    v, f = combine(box())
    f = f.copy()
    f[0] = f[0][::-1]
    result = analyze(v, f)
    assert result.misoriented.sum() == 1
    assert np.flatnonzero(result.misoriented).tolist() == [0]


def test_fully_inverted_cube():
    result = analyze(*combine(inverted(box())))
    assert result.misoriented.sum() == 12
    assert result.volume == pytest.approx(1000)


def test_separate_shells():
    result = analyze(*combine(box(), box(center=(20, 0, 0))))
    assert len(result.shells) == 2
    assert [s.depth for s in result.shells] == [0, 0]
    assert result.overlapping == []
    assert result.volume == pytest.approx(2000)


def test_overlapping_shells():
    result = analyze(*combine(box(), box(center=(5, 0, 0))))
    assert result.overlapping == [(0, 1)]
    assert not result.misoriented.any()


def test_cavity():
    result = analyze(*combine(box(), inverted(box(2))))
    depths = sorted(s.depth for s in result.shells)
    assert depths == [0, 1]
    assert not result.misoriented.any()
    assert result.overlapping == []
    assert result.volume == pytest.approx(1000 - 8)


def test_solid_inside_solid_overlaps():
    result = analyze(*combine(box(), box(2)))
    assert not result.misoriented.any()
    assert result.overlapping == [(0, 1)]


def test_cavity_with_few_flipped_triangles():
    v, f = combine(box(), inverted(box(2)))
    f = f.copy()
    f[12:14] = f[12:14, ::-1]
    result = analyze(v, f)
    assert result.misoriented.sum() == 2
    assert result.overlapping == []


def test_self_intersecting_shell():
    v, f = combine(box())
    v = v.copy()
    v[np.argmax(v.sum(axis=1))] = [-8, -8, -8]  # push a corner through the opposite side
    result = analyze(v, f)
    assert result.self_intersecting.any()
    assert result.boundary_edges == 0


def test_crossings_below_one_micrometre_are_ignored():
    v, f = combine(box(size=1e-3))
    v = v.copy()
    v[np.argmax(v.sum(axis=1))] = [-8e-4, -8e-4, -8e-4]
    assert not analyze(v, f).self_intersecting.any()


def test_no_self_intersections_in_valid_meshes():
    for mesh in (combine(box(), inverted(box(2))), combine(box(), box(center=(5, 0, 0))),
                 (trimesh.creation.icosphere(4).vertices, trimesh.creation.icosphere(4).faces)):
        assert not analyze(*mesh).self_intersecting.any()


def test_empty_mesh():
    result = analyze(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert result.triangles == 0


def test_part_inside_cavity():
    """A ball in a cage: solid, cavity, and a part inside the cavity."""
    result = analyze(*combine(box(), inverted(box(6)), box(2)))
    assert sorted(s.depth for s in result.shells) == [0, 1, 2]
    assert not result.misoriented.any()


def test_nonmanifold_edge():
    result = analyze(*combine(box(), box(center=(10, 10, 0))))  # touch along one edge
    assert result.nonmanifold_edges == 1
    assert len(result.shells) == 1
    assert not result.misoriented.any()


def test_collapsed_triangle_is_degenerate():
    v, f = combine(box())
    f = np.vstack([f, [f[0][0], f[0][0], f[0][1]]])
    result = analyze(v, f)
    assert result.degenerate.sum() == 1
    assert result.boundary_edges == 0

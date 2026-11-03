import numpy as np
import pytest
import trimesh

from meshck.analysis import analyze
from meshck.cli import EXIT_FAIL, EXIT_PASS, main
from meshck.repair import repair

from test_analysis import box, combine, inverted


def assert_clean(result):
    after = analyze(result.vertices, result.faces)
    assert after.boundary_edges == 0
    assert after.nonmanifold_edges == 0
    assert not after.misoriented.any()
    assert not after.degenerate.any()
    assert after.overlapping == []
    return after


def test_clean_mesh_is_untouched():
    v, f = combine(box())
    result = repair(v, f)
    assert result.steps == []
    assert result.deviation == 0.0
    assert len(result.faces) == 12


def test_flipped_triangles():
    v, f = combine(box())
    f = f.copy()
    f[:3] = f[:3, ::-1]
    result = repair(v, f)
    assert result.steps == ["flipped 3 misoriented triangles"]
    assert_clean(result)
    assert result.deviation == pytest.approx(0, abs=1e-9)


def test_inverted_part():
    result = repair(*combine(inverted(box())))
    after = assert_clean(result)
    assert after.volume == pytest.approx(1000)


def test_hole_is_closed():
    v, f = combine(box())
    result = repair(v, f[2:])
    after = assert_clean(result)
    assert after.volume == pytest.approx(1000)
    assert any("pymeshfix" in s for s in result.steps)


def test_collapsed_and_duplicate_triangles():
    v, f = combine(box())
    f = np.vstack([f, f[0], [f[1][0], f[1][0], f[1][1]]])
    result = repair(v, f)
    assert result.steps == ["removed 1 collapsed and 1 duplicate triangles"]
    assert len(assert_clean(result).shells) == 1


def test_shared_wall_cancels():
    """Two boxes sharing a face with identical triangulation become one."""
    a = box()
    # Mirror a at x = 5: the touching faces get identical triangles with
    # opposite winding once b's faces are reversed back to outward.
    b = trimesh.Trimesh(a.vertices * [-1, 1, 1] + [10, 0, 0], a.faces[:, ::-1])
    result = repair(*combine(a, b))
    assert result.steps == ["removed 4 duplicate triangles"]
    after = assert_clean(result)
    assert len(after.shells) == 1
    assert after.volume == pytest.approx(2000)
    assert result.deviation == pytest.approx(0, abs=1e-9)


def test_touching_faces_different_triangulation():
    a, b = box(), box(center=(10, 0, 0))
    after = assert_clean(repair(*combine(a, b)))
    assert after.volume == pytest.approx(2000)


def test_deviation_ignores_filled_hole_but_counts_removed_shell():
    v, f = combine(box())
    assert repair(v, f[2:]).deviation == pytest.approx(0, abs=1e-9)
    v = np.vstack([v, [[20, 0, 0], [21, 0, 0], [20, 1, 0]]])
    f = np.vstack([f, [len(v) - 3, len(v) - 2, len(v) - 1]])
    assert repair(v, f).deviation > 10


def test_touching_edge_is_split():
    result = repair(*combine(box(), box(center=(10, 10, 0))))
    assert result.steps[0] == "split 1 non-manifold edges"
    after = assert_clean(result)
    assert len(after.shells) == 2
    assert after.volume == pytest.approx(2000)


def test_overlapping_shells_merged():
    result = repair(*combine(box(), box(center=(5, 0, 0))))
    assert result.steps == ["merged 2 overlapping shells into 1"]
    after = assert_clean(result)
    assert len(after.shells) == 1
    assert after.volume == pytest.approx(1500)
    assert result.deviation == pytest.approx(0, abs=1e-6)


def test_cavity_and_separate_parts_kept():
    v, f = combine(box(), inverted(box(2)), box(center=(20, 0, 0)))
    result = repair(v, f)
    assert result.steps == []


def test_stray_triangle_removed():
    v, f = combine(box())
    v = np.vstack([v, [[20, 0, 0], [21, 0, 0], [20, 1, 0]]])
    f = np.vstack([f, [len(v) - 3, len(v) - 2, len(v) - 1]])
    result = repair(v, f)
    after = assert_clean(result)
    assert len(after.shells) == 1


def test_cli_repair(tmp_path, capsys):
    path = tmp_path / "open.stl"
    cube = trimesh.creation.box(extents=(10, 10, 10))
    trimesh.Trimesh(cube.vertices, cube.faces[2:]).export(path)
    assert main([str(path)]) == EXIT_FAIL
    assert main(["--repair", str(path)]) == EXIT_PASS
    out = capsys.readouterr().out
    assert "open_repaired.stl: PASS" in out
    assert "before:" in out and "repair:" in out
    assert main([str(tmp_path / "open_repaired.stl")]) == EXIT_PASS


def test_cli_repair_nothing_to_do(cube_stl, tmp_path):
    assert main(["--repair", str(cube_stl)]) == EXIT_PASS
    assert not (tmp_path / "cube_repaired.stl").exists()


def test_cli_output_option(tmp_path):
    path = tmp_path / "flipped.stl"
    trimesh.Trimesh(*combine(inverted(box()))).export(path)
    out = tmp_path / "fixed.stl"
    assert main(["--repair", "-o", str(out), str(path)]) == EXIT_PASS
    assert out.exists()


def test_cli_refuses_to_overwrite_input(tmp_path):
    path = tmp_path / "flipped.stl"
    trimesh.Trimesh(*combine(inverted(box()))).export(path)
    assert main(["--repair", "-o", str(path), str(path)]) == 2


def test_cli_output_requires_repair(cube_stl):
    with pytest.raises(SystemExit):
        main(["-o", "x.stl", str(cube_stl)])

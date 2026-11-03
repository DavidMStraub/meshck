import json

import pytest
import trimesh

from meshck import __version__
from meshck.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_requires_file():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_text_output(cube_stl, capsys):
    assert main([str(cube_stl)]) == EXIT_PASS
    assert "cube.stl: PASS" in capsys.readouterr().out


def test_strict_fails_on_warning(tmp_path):
    path = tmp_path / "two_parts.stl"
    a = trimesh.creation.box(extents=(10, 10, 10))
    b = a.copy().apply_translation((20, 0, 0))
    trimesh.util.concatenate([a, b]).export(path)
    assert main([str(path)]) == EXIT_PASS
    assert main(["--strict", str(path)]) == EXIT_FAIL


def test_errors_fail(tmp_path, capsys):
    path = tmp_path / "open.stl"
    cube = trimesh.creation.box(extents=(10, 10, 10))
    trimesh.Trimesh(cube.vertices, cube.faces[2:]).export(path)
    assert main([str(path)]) == EXIT_FAIL
    assert "mesh is open" in capsys.readouterr().out


def test_json_output(cube_stl, capsys):
    main(["--json", str(cube_stl)])
    (result,) = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["findings"] == []
    assert result["metrics"]["triangles"] == 12
    assert result["metrics"]["volume_mm3"] == pytest.approx(1000)


def test_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.stl")]) == EXIT_ERROR
    assert "no such file" in capsys.readouterr().err


def test_unreadable_file(tmp_path):
    path = tmp_path / "junk.stl"
    path.write_bytes(b"not an stl")
    assert main([str(path)]) == EXIT_ERROR

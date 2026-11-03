import pytest
import trimesh


@pytest.fixture
def cube_stl(tmp_path):
    path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(path)
    return path

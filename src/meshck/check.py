"""Run all checks for one mesh file."""

from pathlib import Path

import numpy as np
import trimesh

from meshck.analysis import Analysis, analyze
from meshck.report import Metrics, Report, Severity


class LoadError(Exception):
    """Raised when a file cannot be read as a triangle mesh."""


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Read an STL file (binary or ASCII) into a mesh.

    Args:
        path: The STL file.

    Returns:
        The mesh, with duplicate vertices merged.

    Raises:
        LoadError: If the file is missing, not a readable STL, or empty.
    """
    if not path.is_file():
        raise LoadError(f"{path}: no such file")
    try:
        mesh = trimesh.load_mesh(path, file_type="stl")
    except Exception as exc:
        raise LoadError(f"{path}: cannot read STL ({exc})") from exc
    if len(mesh.faces) == 0:
        raise LoadError(f"{path}: contains no triangles")
    return mesh


def check_file(path: Path) -> Report:
    """Check one STL file.

    Args:
        path: The STL file.

    Returns:
        A report with metrics and all findings.

    Raises:
        LoadError: If the file cannot be read.
    """
    mesh = load_mesh(path)
    return check_mesh(mesh.vertices, mesh.faces, path)


def check_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path) -> Report:
    """Check a mesh given as arrays.

    Args:
        vertices: ``(n, 3)`` float array of vertex positions in mm.
        faces: ``(m, 3)`` int array of vertex indices per triangle.
        path: File name to show in the report.

    Returns:
        A report with metrics and all findings.
    """
    return check_analysis(analyze(vertices, faces), path)


def check_analysis(result: Analysis, path: Path) -> Report:
    """Turn an existing analysis into a report.

    Args:
        result: Analysis of the mesh.
        path: File name to show in the report.

    Returns:
        A report with metrics and all findings.
    """
    report = Report(path=path, metrics=metrics(result))
    check_topology(result, report)
    check_shells(result, report)
    return report


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a mesh as binary STL."""
    trimesh.Trimesh(vertices, faces, process=False).export(path, file_type="stl")


def metrics(result: Analysis) -> Metrics:
    """Return the measured values reported for every file."""
    return {
        "triangles": result.triangles,
        "shells": len(result.shells),
        "boundary_edges": result.boundary_edges,
        "holes": result.holes,
        "misoriented": int(result.misoriented.sum()),
        "nonmanifold_edges": result.nonmanifold_edges,
        "degenerate": int(result.degenerate.sum()),
        "size_mm": [round(v, 3) for v in result.size],
        "volume_mm3": round(result.volume, 3),
    }


def check_topology(result: Analysis, report: Report) -> None:
    """Add findings for rules every printable mesh must meet."""
    if result.boundary_edges:
        report.add("closed", Severity.ERROR,
                   f"mesh is open: {result.holes} hole(s), {result.boundary_edges} boundary edges")
    if result.nonmanifold_edges:
        report.add("manifold", Severity.ERROR,
                   f"{result.nonmanifold_edges} edges are shared by more than two triangles")
    misoriented = int(result.misoriented.sum())
    if misoriented:
        report.add("oriented", Severity.ERROR, f"{misoriented} misoriented triangles")
    degenerate = int(result.degenerate.sum())
    if degenerate:
        report.add("degenerate", Severity.WARN, f"{degenerate} triangles have zero area")


def check_shells(result: Analysis, report: Report) -> None:
    """Add findings for meshes made of several shells."""
    if result.overlapping:
        report.add("shells.overlapping", Severity.ERROR,
                   f"{len(result.overlapping)} pair(s) of shells intersect each other")
    parts = sum(1 for s in result.shells if s.depth % 2 == 0)
    if parts > 1:
        report.add("shells.separate", Severity.WARN,
                   f"{parts} separate parts; fine for parts meant to move, otherwise join them")

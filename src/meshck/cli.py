"""Command line interface: ``meshck [options] FILE...``."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from meshck import __version__
from meshck.analysis import analyze
from meshck.check import LoadError, check_analysis, check_file, load_mesh, write_stl
from meshck.repair import repair
from meshck.report import RepairInfo, Report

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``meshck`` command."""
    parser = argparse.ArgumentParser(
        prog="meshck",
        description="Check (and optionally repair) STL files for additive manufacturing.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE")
    parser.add_argument("-r", "--repair", action="store_true",
                        help="repair and write a fixed copy (default: FILE_repaired.stl)")
    parser.add_argument("-o", "--output", type=Path, metavar="OUT",
                        help="where to write the repaired copy (only with --repair and one FILE)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``meshck`` command.

    Args:
        argv: Command line arguments without the program name; defaults to
            ``sys.argv[1:]``.

    Returns:
        ``EXIT_PASS`` if all files (or their repaired copies) pass,
        ``EXIT_FAIL`` if any still has errors, ``EXIT_ERROR`` if a file
        cannot be read or written.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output and not args.repair:
        parser.error("--output requires --repair")
    if args.output and len(args.files) > 1:
        parser.error("--output works with a single FILE only")

    reports, exit_code = [], EXIT_PASS
    for path in args.files:
        try:
            report = repair_file(path, args.output) if args.repair else check_file(path)
        except (LoadError, OSError) as exc:
            print(f"meshck: {exc}", file=sys.stderr)
            exit_code = EXIT_ERROR
            continue
        reports.append(report)
        if not report.passed(args.strict) and exit_code == EXIT_PASS:
            exit_code = EXIT_FAIL

    if args.json:
        print(json.dumps([r.to_dict(args.strict) for r in reports], indent=2))
    else:
        for report in reports:
            print(report.to_text(args.strict))
    return exit_code


def repair_file(path: Path, output: Path | None = None) -> Report:
    """Repair one STL file and check the result.

    The repaired copy is written only if repair changed something; the input
    file is never overwritten.

    Args:
        path: The STL file.
        output: Where to write the repaired copy; defaults to
            ``<stem>_repaired.stl`` next to ``path``.

    Returns:
        The report of the repaired copy, or of ``path`` itself if nothing
        needed repair.

    Raises:
        LoadError: If the file cannot be read.
        OSError: If the output would overwrite the input or cannot be written.
    """
    mesh = load_mesh(path)
    analysis = analyze(mesh.vertices, mesh.faces)
    before = check_analysis(analysis, path)
    result = repair(mesh.vertices, mesh.faces, analysis)
    if not result.steps:
        return before

    output = output or path.with_name(f"{path.stem}_repaired.stl")
    if output.resolve() == path.resolve():
        raise OSError(f"{path}: refusing to overwrite the input file")
    write_stl(output, result.vertices, result.faces)
    report = check_analysis(result.analysis, output)
    report.repair = RepairInfo(source=path, before=before.metrics,
                               steps=result.steps, deviation=result.deviation)
    return report

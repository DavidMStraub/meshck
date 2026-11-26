# meshck

Check and repair triangle meshes (STL) before additive manufacturing.

meshck finds holes, flipped triangles and overlapping bodies in STL files
and repairs them. Like `fsck` for file systems, it checks by default and
repairs on request. It runs on Linux, macOS and Windows.

Status: early development.

## Installation

```sh
pip install git+https://github.com/DavidMStraub/meshck
```

or, in a uv project:

```sh
uv add git+https://github.com/DavidMStraub/meshck
```

Requires Python 3.11 or newer.

## Usage

Check one or more files:

```sh
meshck part.stl
```

```
part.stl: FAIL
  triangles=327580  shells=1  boundary_edges=54  holes=1  misoriented=4900  ...
  error  closed: mesh is open: 1 hole(s), 54 boundary edges
  error  oriented: 4900 misoriented triangles
```

Repair and write a fixed copy (`part_repaired.stl`, or the path given with
`-o`). The input file is never overwritten:

```sh
meshck --repair part.stl
```

```
part.stl -> part_repaired.stl: PASS
  before: triangles=327580  shells=1  boundary_edges=54  holes=1  misoriented=4900  ...
  after:  triangles=327674  shells=1  boundary_edges=0  holes=0  misoriented=0  ...
  repair: flipped 4900 misoriented triangles
  repair: rebuilt 1 open shell(s) with pymeshfix
  deviation_mm=0.0000
```

Options:

| Option | Effect |
|---|---|
| `-r`, `--repair` | repair and write a fixed copy |
| `-o OUT`, `--output OUT` | where to write the repaired copy (one input file only) |
| `--fill-cavities` | when repairing, also fill cavities (see below) |
| `--strict` | treat warnings as errors |
| `--json` | print results as JSON |

Exit codes: `0` all files pass, `1` errors remain, `2` a file could not be
read or written.

## What is checked

| Check | Severity |
|---|---|
| Empty mesh (no triangles) | error |
| Open mesh (holes, boundary edges) | error |
| Edges shared by more than two triangles | error |
| Misoriented triangles (including inverted parts and cavities) | error |
| Triangles crossing other triangles of the same shell | error |
| Overlapping shells, including a solid inside a solid | error |
| Shells touching over an area (they fuse when printed) | warning |
| Closed shells of zero thickness (they do not print) | warning |
| Degenerate (zero-area) triangles | warning |
| Several separate parts | warning |

Separate parts are only a warning because moving parts printed in one piece
are separate by design. Size and volume are reported without a limit.

A closed shell inside another one is a cavity if it points inwards and a
solid inside a solid if it points outwards. Crossings less than 1 µm deep are
ignored as rounding noise. Shells touching only along an edge or at a point
do not count as touching.

## What repair does

Each step runs only if the check finds the problem it addresses:

1. Remove collapsed and duplicate triangles.
2. Flip misoriented triangles.
3. Split edges shared by more than two triangles, so touching bodies become
   separate closed shells.
4. Rebuild shells that are still open with
   [pymeshfix](https://github.com/pyvista/pymeshfix), one shell at a time,
   and remove leftovers of zero thickness: stray triangles, flat sheets and
   fins. If pymeshfix cannot close a shell that has thickness, the shell is
   kept as it is.
5. Merge overlapping shells, and shells touching over an area, with a boolean
   union ([manifold3d](https://github.com/elalish/manifold)).
6. With `--fill-cavities` only: remove every shell enclosed by another one,
   so that cavities become solid.

`deviation_mm` is the largest distance the original surface moved. Filled
holes, filled cavities and inner walls removed by a merge are not counted,
deleted shells are.

Separate parts are kept, and so are cavities unless `--fill-cavities` is
given. meshck does not rebuild the whole outer surface and does not repair
self-intersecting shells. If the steps above are not enough, the mesh has to
be fixed by hand.

## Python API

```python
from pathlib import Path

from meshck.check import check_mesh, load_mesh
from meshck.repair import repair

mesh = load_mesh(Path("part.stl"))
result = repair(mesh.vertices, mesh.faces)
print(result.steps, result.deviation)
print(check_mesh(result.vertices, result.faces, Path("part.stl")).to_text())
```

## Scope

meshck checks whether a mesh is a valid solid. It does not check design
rules such as wall thickness, build volume or clearances.

## License

MIT, see [LICENSE](LICENSE).

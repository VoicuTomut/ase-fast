# fmt: off

"""Reference implementation of reader and writer for standard XYZ files.

See https://en.wikipedia.org/wiki/XYZ_file_format

Note that the .xyz files are handled by the extxyz module by default.
"""
import numpy as np

from ase.atoms import Atoms
from ase.io.utils import validate_comment_line

# ---------------------------------------------------------------------------
# Optional Rust fast path for simple XYZ hot loops (Phase 10).
# ---------------------------------------------------------------------------
try:
    from ase._io_rs import (  # type: ignore[import]
        parse_xyz_block_rs as _parse_xyz_block_rs,
        format_xyz_block_rs as _format_xyz_block_rs,
    )
    _HAVE_RUST_IO = True
except ImportError:
    _HAVE_RUST_IO = False


def read_xyz(fileobj, index):
    # This function reads first all atoms and then yields based on the index.
    # Perfomance could be improved, but this serves as a simple reference.
    # It'd require more code to estimate the total number of images
    # without reading through the whole file (note: the number of atoms
    # can differ for every image).
    lines = fileobj.readlines()
    images = []
    while len(lines) > 0:
        natoms = int(lines.pop(0))
        lines.pop(0)  # Comment line; ignored
        if _HAVE_RUST_IO:
            block = lines[:natoms]
            del lines[:natoms]
            syms, pos = _parse_xyz_block_rs(block)
            images.append(Atoms(symbols=syms, positions=pos))
        else:
            symbols = []
            positions = []
            for _ in range(natoms):
                line = lines.pop(0)
                symbol, x, y, z = line.split()[:4]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
            images.append(Atoms(symbols=symbols, positions=positions))
    yield from images[index]


def write_xyz(fileobj, images, comment='', fmt='%22.15f'):
    comment = validate_comment_line(comment)

    for atoms in images:
        natoms = len(atoms)
        fileobj.write('%d\n%s\n' % (natoms, comment))
        if _HAVE_RUST_IO and fmt == '%22.15f':
            fileobj.write(_format_xyz_block_rs(
                list(atoms.symbols),
                np.ascontiguousarray(atoms.positions, dtype=np.float64),
            ))
        else:
            for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
                fileobj.write('%-2s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z))


# Compatibility with older releases
simple_read_xyz = read_xyz
simple_write_xyz = write_xyz

"""
ase.ai — natural language → ASE Atoms objects.

Requires the ``anthropic`` package::

    pip install anthropic

Quick start::

    from ase.ai import build
    atoms = build("FCC copper, 3x3x3 supercell")

Or with more control::

    from ase.ai import AtomicStructureBuilder
    builder = AtomicStructureBuilder(model="claude-opus-4-6")
    atoms = builder.build("Cu(111) slab, 4 layers, 12 Angstrom vacuum")
"""

from ase.ai.builder import AtomicStructureBuilder, build

__all__ = ["AtomicStructureBuilder", "build"]

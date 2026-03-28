"""
Tests for ase.ai — natural language → Atoms builder.

Unit tests mock the Anthropic API so they run without a real key.
Integration tests are skipped unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import io
import os
import textwrap
import types
import unittest.mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, molecule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(code: str):
    """Return a mock anthropic.Anthropic client that responds with *code*."""
    content_block = MagicMock()
    content_block.text = code

    response = MagicMock()
    response.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _builder_with_mock(code: str):
    """Create an AtomicStructureBuilder whose API call returns *code*."""
    from ase.ai.builder import AtomicStructureBuilder

    builder = AtomicStructureBuilder.__new__(AtomicStructureBuilder)
    builder._client = _make_mock_client(code)
    builder.model = "mock-model"
    builder.max_retries = 1
    return builder


# ---------------------------------------------------------------------------
# Import / dependency tests
# ---------------------------------------------------------------------------

class TestImport:
    def test_module_imports(self):
        import ase.ai
        assert hasattr(ase.ai, "build")
        assert hasattr(ase.ai, "AtomicStructureBuilder")

    def test_no_anthropic_raises_import_error(self, monkeypatch):
        """If anthropic is not installed, ImportError with install hint."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from ase.ai.builder import AtomicStructureBuilder
        with pytest.raises(ImportError, match="pip install anthropic"):
            AtomicStructureBuilder()


# ---------------------------------------------------------------------------
# _strip_markdown
# ---------------------------------------------------------------------------

class TestStripMarkdown:
    def test_plain_code_unchanged(self):
        from ase.ai.builder import _strip_markdown
        code = "atoms = bulk('Cu', 'fcc')"
        assert _strip_markdown(code) == code

    def test_strips_python_fence(self):
        from ase.ai.builder import _strip_markdown
        fenced = "```python\natoms = bulk('Cu', 'fcc')\n```"
        assert _strip_markdown(fenced) == "atoms = bulk('Cu', 'fcc')"

    def test_strips_plain_fence(self):
        from ase.ai.builder import _strip_markdown
        fenced = "```\nfrom ase.build import bulk\natoms = bulk('Al')\n```"
        result = _strip_markdown(fenced)
        assert "```" not in result
        assert "bulk" in result

    def test_strips_leading_trailing_whitespace(self):
        from ase.ai.builder import _strip_markdown
        assert _strip_markdown("  atoms = molecule('H2O')  ") == "atoms = molecule('H2O')"


# ---------------------------------------------------------------------------
# _exec_and_validate  (no network)
# ---------------------------------------------------------------------------

class TestExecAndValidate:
    """Tests for the code execution + validation logic, fully offline."""

    def _builder(self):
        from ase.ai.builder import AtomicStructureBuilder
        b = AtomicStructureBuilder.__new__(AtomicStructureBuilder)
        b._client = MagicMock()
        b.model = "mock"
        b.max_retries = 0
        return b

    def test_valid_bulk_code(self):
        b = self._builder()
        code = "atoms = bulk('Cu', 'fcc', a=3.615)"
        atoms = b._exec_and_validate(code, "FCC copper")
        assert isinstance(atoms, Atoms)
        assert all(atoms.pbc)
        assert len(atoms) == 1

    def test_valid_supercell_code(self):
        b = self._builder()
        code = "atoms = bulk('Al', 'fcc', a=4.05).repeat((2,2,2))"
        atoms = b._exec_and_validate(code, "Al supercell")
        assert len(atoms) == 8

    def test_valid_molecule_code(self):
        b = self._builder()
        code = "atoms = molecule('H2O')"
        atoms = b._exec_and_validate(code, "water")
        assert isinstance(atoms, Atoms)
        assert not any(atoms.pbc)

    def test_valid_slab_code(self):
        b = self._builder()
        code = textwrap.dedent("""\
            from ase.build import fcc111
            atoms = fcc111('Cu', size=(2,2,3), vacuum=10.0)
        """)
        atoms = b._exec_and_validate(code, "Cu slab")
        assert not atoms.pbc[2]   # z is non-periodic for slabs
        assert len(atoms) == 12

    def test_syntax_error_raises_value_error(self):
        b = self._builder()
        with pytest.raises(ValueError, match="SyntaxError"):
            b._exec_and_validate("atoms = bulk('Cu'", "broken code")

    def test_runtime_error_raises_value_error(self):
        b = self._builder()
        with pytest.raises(ValueError, match="ValueError"):
            b._exec_and_validate("atoms = bulk('Xq', 'fcc')", "unknown element")

    def test_missing_atoms_assignment_raises(self):
        b = self._builder()
        with pytest.raises(ValueError, match="did not assign"):
            b._exec_and_validate("x = bulk('Cu')", "forgot atoms")

    def test_wrong_type_assignment_raises(self):
        b = self._builder()
        with pytest.raises(ValueError, match="did not assign"):
            b._exec_and_validate("atoms = 42", "wrong type")

    def test_exec_and_validate_does_not_crash_on_nan(self):
        """
        ASE extxyz silently serialises NaN positions; we document that the
        builder does not crash and returns the Atoms as-is.  Callers that
        need NaN-free output should validate positions themselves.
        """
        b = self._builder()
        code = textwrap.dedent("""\
            from ase import Atoms
            import numpy as np
            atoms = Atoms('Cu', positions=[[float('nan'), 0, 0]], pbc=False)
        """)
        # Should not raise — ASE accepts NaN positions in extxyz output
        result = b._exec_and_validate(code, "NaN positions")
        assert isinstance(result, Atoms)

    def test_numpy_available_in_namespace(self):
        b = self._builder()
        # BCC primitive has 1 atom → repeat(2,2,2) → 8 atoms
        code = textwrap.dedent("""\
            import numpy as np
            from ase.build import bulk
            atoms = bulk('Fe', 'bcc', a=2.87).repeat((2,2,2))
            assert isinstance(np.array([1]), np.ndarray)
        """)
        atoms = b._exec_and_validate(code, "BCC Fe")
        assert len(atoms) == 8

    def test_multiline_code_with_imports(self):
        b = self._builder()
        code = textwrap.dedent("""\
            from ase.build import bulk, add_vacuum
            atoms = bulk('Cu', 'fcc', a=3.615)
            atoms = atoms.repeat((3, 3, 1))
        """)
        atoms = b._exec_and_validate(code, "Cu layer")
        assert len(atoms) == 9


# ---------------------------------------------------------------------------
# AtomicStructureBuilder.build  (mocked API)
# ---------------------------------------------------------------------------

class TestBuilderBuild:
    def test_build_returns_atoms(self):
        code = "atoms = bulk('Cu', 'fcc', a=3.615)"
        builder = _builder_with_mock(code)
        atoms = builder.build("FCC copper")
        assert isinstance(atoms, Atoms)

    def test_build_strips_markdown(self):
        code = "```python\natoms = bulk('Al', 'fcc', a=4.05)\n```"
        builder = _builder_with_mock(code)
        atoms = builder.build("aluminium bulk")
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 1

    def test_build_retries_on_bad_code(self):
        """On first call returns broken code; second call returns valid code."""
        from ase.ai.builder import AtomicStructureBuilder

        bad_code = "atoms = 'not an atoms object'"
        good_code = "atoms = bulk('Cu', 'fcc', a=3.615)"

        content_bad = MagicMock(); content_bad.text = bad_code
        content_good = MagicMock(); content_good.text = good_code
        resp_bad = MagicMock(); resp_bad.content = [content_bad]
        resp_good = MagicMock(); resp_good.content = [content_good]

        client = MagicMock()
        client.messages.create.side_effect = [resp_bad, resp_good]

        builder = AtomicStructureBuilder.__new__(AtomicStructureBuilder)
        builder._client = client
        builder.model = "mock"
        builder.max_retries = 1

        atoms = builder.build("FCC copper")
        assert isinstance(atoms, Atoms)
        assert client.messages.create.call_count == 2

    def test_build_raises_after_all_retries(self):
        """All attempts return invalid code → ValueError is raised."""
        code = "atoms = 'definitely not atoms'"
        builder = _builder_with_mock(code)
        builder.max_retries = 1

        with pytest.raises(ValueError, match="attempt"):
            builder.build("something impossible")

    def test_build_error_message_includes_description(self):
        code = "atoms = None"
        builder = _builder_with_mock(code)
        builder.max_retries = 0

        with pytest.raises(ValueError) as exc_info:
            builder.build("FCC platinum")
        assert "FCC platinum" in str(exc_info.value)

    def test_api_called_with_description_in_messages(self):
        code = "atoms = bulk('Cu', 'fcc', a=3.615)"
        builder = _builder_with_mock(code)
        builder.build("my test description")

        call_kwargs = builder._client.messages.create.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][3]
        assert any("my test description" in str(m) for m in messages)


# ---------------------------------------------------------------------------
# Module-level build() function
# ---------------------------------------------------------------------------

class TestModuleLevelBuild:
    def test_build_function_exists(self):
        from ase.ai import build
        assert callable(build)

    @patch("ase.ai.builder.AtomicStructureBuilder")
    def test_build_delegates_to_builder(self, MockBuilder):
        expected = bulk('Cu', 'fcc', a=3.615)
        instance = MagicMock()
        instance.build.return_value = expected
        MockBuilder.return_value = instance

        from ase.ai.builder import build
        result = build("FCC copper", api_key="test-key", model="claude-test")

        MockBuilder.assert_called_once_with(api_key="test-key", model="claude-test")
        instance.build.assert_called_once_with("FCC copper")
        assert result is expected


# ---------------------------------------------------------------------------
# Integration tests  (skipped without real API key)
# ---------------------------------------------------------------------------

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
needs_api_key = pytest.mark.skipif(
    not ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY not set — skipping live API tests",
)


@needs_api_key
class TestLiveIntegration:
    """
    These tests make real API calls and are skipped in CI without a key.
    Run manually with:  ANTHROPIC_API_KEY=sk-ant-... pytest ase/test/ai/ -v -k live
    """

    def test_fcc_bulk(self):
        from ase.ai import build
        atoms = build("FCC copper bulk")
        assert isinstance(atoms, Atoms)
        assert all(atoms.pbc)
        assert "Cu" in atoms.get_chemical_symbols()

    def test_supercell(self):
        from ase.ai import build
        atoms = build("BCC iron, 2x2x2 supercell")
        assert isinstance(atoms, Atoms)
        assert len(atoms) >= 2

    def test_molecule(self):
        from ase.ai import build
        atoms = build("water molecule")
        assert isinstance(atoms, Atoms)
        assert not any(atoms.pbc)
        assert set(atoms.get_chemical_symbols()) == {"H", "O"}

    def test_slab_has_vacuum(self):
        from ase.ai import build
        atoms = build("Cu(111) slab, 3 layers, 10 Angstrom vacuum")
        assert isinstance(atoms, Atoms)
        # pbc should be False along z
        assert not atoms.pbc[2]

"""
ase.ai.builder — natural language → ASE Atoms via Claude API.

The builder calls the Anthropic Messages API, receives a Python code snippet,
executes it in a sandboxed namespace, and validates the resulting Atoms object
by round-tripping it through extxyz (the canonical ASE validation format).

Security note: exec() is used intentionally — this module generates and runs
LLM-produced code. Only import / use it in trusted environments.
"""

from __future__ import annotations

import io
import textwrap
from typing import Optional

from ase import Atoms
from ase.ai._prompts import SYSTEM_PROMPT

__all__ = ["AtomicStructureBuilder", "build"]


class AtomicStructureBuilder:
    """
    Build ASE :class:`~ase.Atoms` objects from natural language descriptions.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if *None*.
    model:
        Claude model ID.  Defaults to ``claude-haiku-4-5-20251001``
        (fast and cheap for code generation tasks).
    max_retries:
        How many times to retry if the generated code is invalid.
        Each retry appends the error to the conversation so the model
        can self-correct.

    Examples
    --------
    >>> from ase.ai import AtomicStructureBuilder
    >>> builder = AtomicStructureBuilder()
    >>> atoms = builder.build("FCC copper, 3x3x3 supercell")
    >>> atoms.get_chemical_formula()
    'Cu108'
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ase.ai requires the 'anthropic' package. "
                "Install it with:  pip install anthropic"
            ) from exc
        import anthropic as _anthropic

        self._client = _anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, description: str) -> Atoms:
        """
        Build an :class:`~ase.Atoms` object from a natural language description.

        Parameters
        ----------
        description:
            Plain-English description of the atomic structure.

        Returns
        -------
        Atoms
            Constructed and validated ASE Atoms object.

        Raises
        ------
        ValueError
            If the generated code fails validation after all retries.

        Examples
        --------
        >>> atoms = builder.build("BCC iron, cubic supercell 2x2x2")
        >>> atoms.get_chemical_formula()
        'Fe16'
        """
        messages: list[dict] = [{"role": "user", "content": description}]
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            code = _strip_markdown(response.content[0].text)

            try:
                atoms = self._exec_and_validate(code, description)
                return atoms
            except ValueError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    # Feed the error back so the model can self-correct.
                    messages.append({"role": "assistant", "content": code})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"That code raised an error:\n{exc}\n\n"
                            "Please fix it and return only the corrected code."
                        ),
                    })

        raise ValueError(
            f"Could not generate a valid Atoms object for '{description}' "
            f"after {self.max_retries + 1} attempt(s). "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exec_and_validate(self, code: str, description: str) -> Atoms:
        """Execute *code* and validate the 'atoms' variable it must assign."""
        import ase.build
        import numpy as np

        namespace: dict = {
            "np": np,
            "numpy": np,
            "Atoms": Atoms,
        }
        # Expose all public ase.build helpers directly.
        for _name in dir(ase.build):
            if not _name.startswith("_"):
                namespace[_name] = getattr(ase.build, _name)

        try:
            exec(textwrap.dedent(code), namespace)  # noqa: S102
        except Exception as exc:
            raise ValueError(
                f"{type(exc).__name__} while executing generated code: {exc}\n"
                f"--- code ---\n{code}"
            ) from exc

        atoms = namespace.get("atoms")
        if not isinstance(atoms, Atoms):
            raise ValueError(
                f"Generated code did not assign an Atoms object to 'atoms'. "
                f"Got: {type(atoms).__name__!r}.\n--- code ---\n{code}"
            )

        # Validate: round-trip through extxyz catches bad positions,
        # NaN coordinates, unsupported pbc combos, etc.
        # Surface slabs may contain unhashable metadata (adsorbate_info);
        # those RuntimeWarnings are not fatal — suppress them here.
        import warnings
        buf = io.StringIO()
        try:
            import ase.io
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # metadata warnings are non-fatal
                ase.io.write(buf, atoms, format="extxyz")
        except Exception as exc:
            raise ValueError(
                f"Generated Atoms failed extxyz validation: {exc}\n"
                f"--- code ---\n{code}"
            ) from exc

        return atoms


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def build(
    description: str,
    api_key: Optional[str] = None,
    model: str = AtomicStructureBuilder.DEFAULT_MODEL,
) -> Atoms:
    """
    Build an ASE :class:`~ase.Atoms` object from a natural language description.

    This is a thin wrapper around :class:`AtomicStructureBuilder` for one-off use.

    Parameters
    ----------
    description:
        Plain-English description, e.g. ``"FCC copper, 3x3x3 supercell"``.
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable.
    model:
        Claude model ID.

    Returns
    -------
    Atoms

    Examples
    --------
    >>> from ase.ai import build
    >>> atoms = build("graphene, 4x4 supercell, 15 Angstrom vacuum")
    >>> atoms.pbc.tolist()
    [True, True, False]
    """
    return AtomicStructureBuilder(api_key=api_key, model=model).build(description)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` fences if the model added them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```python or ```) and last line if it's ```
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end]).strip()
    return text

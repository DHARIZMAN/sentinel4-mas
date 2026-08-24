"""Sandboxed evidence-file reader.

Agents frequently want to read an artefact the operator mentioned. Allowing an
LLM-supplied path to reach ``open()`` unfiltered is a path-traversal hazard, so
every read is confined to the repository's ``evidence/`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: Root of the evidence sandbox, overridable for tests via ``MAS_EVIDENCE_DIR``.
EVIDENCE_ROOT = Path(os.getenv("MAS_EVIDENCE_DIR", "evidence")).resolve()

#: Hard ceiling on how much text a single read may return to an agent.
MAX_CHARS = 8000


def read_evidence_file(filename: str, max_chars: int = MAX_CHARS) -> dict[str, Any]:
    """Read a text artefact from the sandboxed evidence directory.

    Args:
        filename: Name of the file, relative to the evidence root. Traversal
            segments are rejected rather than resolved.
        max_chars: Maximum characters to return; content beyond this is
            truncated and flagged.

    Returns:
        A dictionary with keys:
            ``filename`` (str): the requested name.
            ``exists`` (bool): whether the file was found.
            ``content`` (str): the file text, possibly truncated.
            ``truncated`` (bool): whether truncation occurred.
            ``bytes`` (int): size on disk, ``0`` when missing.

    Raises:
        PermissionError: If the resolved path escapes the evidence sandbox.
    """
    candidate = (EVIDENCE_ROOT / filename).resolve()

    # [HUMAN-REVIEW] The AI's version checked for the literal string ".." in the
    # filename. That is defeated by symlinks and by absolute paths, so we replaced
    # it with a resolved-path containment check, which is the only form of this
    # guard that actually holds.
    if EVIDENCE_ROOT not in candidate.parents and candidate != EVIDENCE_ROOT:
        raise PermissionError(
            f"Refused: '{filename}' resolves outside the evidence sandbox ({EVIDENCE_ROOT})."
        )

    if not candidate.is_file():
        return {
            "filename": filename,
            "exists": False,
            "content": "",
            "truncated": False,
            "bytes": 0,
        }

    raw = candidate.read_text(encoding="utf-8", errors="replace")
    truncated = len(raw) > max_chars
    return {
        "filename": filename,
        "exists": True,
        "content": raw[:max_chars],
        "truncated": truncated,
        "bytes": candidate.stat().st_size,
    }

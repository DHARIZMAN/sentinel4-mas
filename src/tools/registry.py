"""Tool registry and invocation guard for the SENTINEL-4 MAS.

Agents do not call Python functions directly. They emit a tool *name*, and this
registry resolves it. That indirection is deliberate: it is the point at which a
**hallucinated tool** — a plausible-sounding name the model invented, such as
``run_nmap_scan`` — is caught and converted into a structured warning instead of
an ``AttributeError`` that would kill the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


class ToolExecutionError(RuntimeError):
    """A registered tool was found but raised while executing."""


class HallucinatedToolError(KeyError):
    """An agent requested a tool name that does not exist in the registry."""


@dataclass(frozen=True)
class ToolSpec:
    """Metadata describing one callable tool.

    Attributes:
        name: The identifier agents use to request the tool.
        description: One-line summary injected into agent system prompts.
        func: The Python callable implementing the tool.
        owner: Which agent role is expected to invoke it.
    """

    name: str
    description: str
    func: Callable[..., dict[str, Any]]
    owner: str


class ToolRegistry:
    """Name-to-callable resolver with hallucination detection and timing.

    Attributes:
        invocation_log: Chronological record of every tool call attempted,
            including refused (hallucinated) ones.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._tools: dict[str, ToolSpec] = {}
        self.invocation_log: list[dict[str, Any]] = []

    def register(self, spec: ToolSpec) -> None:
        """Add a tool to the registry.

        Args:
            spec: The tool specification to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool registration: {spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        """List every registered tool name.

        Returns:
            Sorted list of tool identifiers.
        """
        return sorted(self._tools)

    def catalogue_for(self, owner: str) -> str:
        """Render the tool catalogue for one agent role as prompt text.

        Args:
            owner: The agent role whose tools should be listed.

        Returns:
            A newline-separated ``- name: description`` block, or a placeholder
            when the role owns no tools.
        """
        lines = [
            f"- {s.name}: {s.description}"
            for s in self._tools.values()
            if s.owner in (owner, "shared")
        ]
        return "\n".join(lines) if lines else "- (no tools assigned to this role)"

    def invoke(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a registered tool by name.

        Args:
            name: The tool identifier requested by an agent.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            The tool's structured result dictionary.

        Raises:
            HallucinatedToolError: The name is not registered.
            ToolExecutionError: The tool was found but raised while running.
        """
        started = time.monotonic()
        spec = self._tools.get(name)
        if spec is None:
            self.invocation_log.append(
                {"tool": name, "status": "hallucinated", "duration_ms": 0.0}
            )
            raise HallucinatedToolError(
                f"Agent requested unknown tool '{name}'. Registered tools: {self.names()}"
            )
        try:
            result = spec.func(**kwargs)
        except TypeError as exc:
            # [HUMAN-REVIEW] TypeError is separated from the generic case on
            # purpose. An LLM that invents *arguments* for a real tool produces a
            # TypeError, and that is a prompt-design problem we want reported
            # distinctly from a genuine bug inside the tool body.
            self.invocation_log.append({"tool": name, "status": "bad_arguments", "duration_ms": 0.0})
            raise ToolExecutionError(f"Tool '{name}' called with invalid arguments: {exc}") from exc
        except Exception as exc:
            self.invocation_log.append({"tool": name, "status": "error", "duration_ms": 0.0})
            raise ToolExecutionError(f"Tool '{name}' failed: {type(exc).__name__}: {exc}") from exc

        duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        self.invocation_log.append({"tool": name, "status": "ok", "duration_ms": duration_ms})
        return result


def build_default_registry() -> ToolRegistry:
    """Construct the registry with all five production tools installed.

    Returns:
        A :class:`ToolRegistry` populated with the audio, video, intelligence,
        indicator-parsing and evidence-reading tools.
    """
    from src.tools.audio_forensics import scan_audio_artifacts
    from src.tools.evidence_reader import read_evidence_file
    from src.tools.indicator_parser import parse_indicators
    from src.tools.threat_intel import query_threat_intel
    from src.tools.video_forensics import check_frame_consistency

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="scan_audio_artifacts",
        description="Run spectral/prosodic forensics on an audio artefact and return synthesis indicators.",
        func=scan_audio_artifacts,
        owner="audio_analyst",
    ))
    registry.register(ToolSpec(
        name="check_frame_consistency",
        description="Inspect a video artefact for temporal, lighting and boundary manipulation artefacts.",
        func=check_frame_consistency,
        owner="video_detector",
    ))
    registry.register(ToolSpec(
        name="query_threat_intel",
        description="Look up an indicator (IP, domain, hash, CVE, campaign) in the threat-intelligence corpus.",
        func=query_threat_intel,
        owner="cyber_coordinator",
    ))
    registry.register(ToolSpec(
        name="parse_indicators",
        description="Extract IPv4 addresses, domains, hashes and CVE identifiers from free text.",
        func=parse_indicators,
        owner="shared",
    ))
    registry.register(ToolSpec(
        name="read_evidence_file",
        description="Read a text artefact from the sandboxed evidence directory.",
        func=read_evidence_file,
        owner="shared",
    ))
    return registry

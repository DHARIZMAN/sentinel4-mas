"""Shared-state ("blackboard") definition for the SENTINEL-4 MAS.

Every node in the LangGraph workflow reads from and writes to a single
:class:`MissionState` dictionary. Choosing an explicit shared state — rather than
passing messages agent-to-agent — is what lets a late agent (the Strategic
Predictor) consult the findings of every earlier agent, and what lets the
fallback handler assemble a partial answer from whatever happens to be on the
blackboard when a failure occurs.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal, TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two blackboard sections instead of replacing one.

    LangGraph applies a *reducer* when two branches write the same state key in
    the same super-step. The Audio and Video analysts run concurrently and both
    write to ``agent_outputs``.

    Args:
        left: The existing value on the blackboard.
        right: The value produced by a node in this super-step.

    Returns:
        A new dictionary containing both sets of keys, with ``right`` winning
        on collision.
    """
    # [HUMAN-REVIEW] The AI's first draft used the default "last write wins"
    # behaviour here, which silently deleted the Audio analyst's report whenever
    # the Video analyst finished in the same super-step. We replaced it with an
    # explicit merge reducer; this is the single most important correctness fix
    # in the whole state layer.
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def append_lists(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer that concatenates log-style state channels.

    Args:
        left: Entries already on the blackboard.
        right: Entries produced by the current node.

    Returns:
        A new list holding ``left`` followed by ``right``.
    """
    return list(left or []) + list(right or [])


ThreatBand = Literal["MINIMAL", "ELEVATED", "SEVERE", "CRITICAL"]


class MissionState(TypedDict, total=False):
    """The blackboard passed between every node of the countermeasure graph.

    Attributes:
        run_id: Unique identifier for this mission run.
        started_at: ``time.monotonic()`` reading taken at ingestion, used for
            all elapsed-time and fallback-latency measurements.
        raw_input: The operator's original multi-vector incident brief.
        evidence_refs: Filenames or URIs of artefacts referenced by the brief.
        route_decision: The router's structured verdict (mode, agents, rationale).
        activated_agents: Names of specialists the router dispatched to.
        agent_outputs: Blackboard section holding each specialist's JSON report,
            keyed by agent name.
        threat_score: Fused 0-100 confidence-weighted severity score.
        threat_band: Categorical banding derived from ``threat_score``.
        escalation_path: Which branch the threat gate selected.
        evaluation: Output of the self-evaluation module.
        refinement_count: How many times the workflow has looped for refinement;
            bounded by ``max_refinement_loops`` to prevent infinite cycling.
        warnings: Non-fatal advisories raised by any node.
        errors: Structured error records; presence of entries with
            ``fatal=True`` diverts the graph to the fallback handler.
        fallback_triggered: Whether the degraded-output path was taken.
        final_report: The mission product handed back to the operator.
        trace: Ordered execution log used for the live demonstration.
    """

    run_id: str
    started_at: float
    raw_input: str
    evidence_refs: list[str]

    route_decision: dict[str, Any]
    activated_agents: Annotated[list[str], append_lists]

    agent_outputs: Annotated[dict[str, Any], merge_dicts]

    threat_score: float
    threat_band: ThreatBand
    escalation_path: str

    evaluation: dict[str, Any]
    refinement_count: int

    warnings: Annotated[list[dict[str, Any]], append_lists]
    errors: Annotated[list[dict[str, Any]], append_lists]
    fallback_triggered: bool

    final_report: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], append_lists]


def new_mission_state(raw_input: str, evidence_refs: list[str] | None = None) -> MissionState:
    """Create a fresh blackboard for one incident.

    Args:
        raw_input: The operator's incident brief, free text.
        evidence_refs: Optional list of artefact identifiers accompanying it.

    Returns:
        A :class:`MissionState` with every channel initialised, so downstream
        nodes never have to guard against missing keys.
    """
    return MissionState(
        run_id=f"SENTINEL-{uuid.uuid4().hex[:8].upper()}",
        started_at=time.monotonic(),
        raw_input=raw_input,
        evidence_refs=list(evidence_refs or []),
        route_decision={},
        activated_agents=[],
        agent_outputs={},
        threat_score=0.0,
        threat_band="MINIMAL",
        escalation_path="undetermined",
        evaluation={},
        refinement_count=0,
        warnings=[],
        errors=[],
        fallback_triggered=False,
        final_report={},
        trace=[],
    )


def elapsed_ms(state: MissionState) -> float:
    """Return milliseconds elapsed since the mission started.

    Args:
        state: The current blackboard.

    Returns:
        Elapsed wall-clock time in milliseconds, or ``0.0`` if the state has no
        start marker (which happens only in isolated unit tests).
    """
    start = state.get("started_at")
    return 0.0 if start is None else (time.monotonic() - start) * 1000.0


def trace_event(node: str, detail: str, **extra: Any) -> dict[str, Any]:
    """Build one structured trace record.

    Args:
        node: Name of the graph node emitting the event.
        detail: Human-readable description shown in the live demo.
        **extra: Any additional structured fields to attach.

    Returns:
        A dictionary suitable for appending to ``state["trace"]``.
    """
    return {"node": node, "detail": detail, "ts": time.strftime("%H:%M:%S"), **extra}


def band_for_score(score: float) -> ThreatBand:
    """Convert a numeric threat score into its categorical band.

    Args:
        score: Fused threat score in the range 0-100.

    Returns:
        The matching :data:`ThreatBand` label.
    """
    if score >= 80.0:
        return "CRITICAL"
    if score >= 60.0:
        return "SEVERE"
    if score >= 35.0:
        return "ELEVATED"
    return "MINIMAL"

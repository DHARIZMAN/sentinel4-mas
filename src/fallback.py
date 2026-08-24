"""Risk warning and fallback strategy (风险提示和退让策略).

This module is the answer to the brief's risk-mitigation clause: when an agent
loops, hallucinates a tool, breaches its timeout or the endpoint disappears, the
system must not crash — it must emit a partial answer built strictly from
whatever was already verified.

Two properties make that guarantee real:

* **No LLM calls.** The fallback path is pure Python over the blackboard, so it
  cannot itself fail for the reason that triggered it. This is why it completes
  in milliseconds, comfortably inside the brief's 5-second ceiling.
* **No invention.** Every line of a fallback product is copied from a report
  that actually completed. Where nothing completed, the product says so.
"""

from __future__ import annotations

from typing import Any

from src.state import MissionState, elapsed_ms, trace_event

#: Maximum time the fallback path is permitted to take (assessment: < 5 s).
FALLBACK_BUDGET_SECONDS = 5.0

#: Operator-facing guidance keyed by failure class.
RISK_GUIDANCE: dict[str, str] = {
    "LLMUnavailableError": (
        "Inference endpoint unreachable. Treat the findings below as UNCONFIRMED "
        "and re-run once the endpoint is restored."
    ),
    "LLMParseError": (
        "A specialist returned unparseable output. Its findings are excluded; "
        "coverage of that modality is unknown, not clear."
    ),
    "LLMContractError": (
        "A specialist violated its output contract. Its partial findings are "
        "excluded to avoid propagating malformed fields."
    ),
    "HallucinatedToolError": (
        "An agent requested a non-existent tool. Evidence quality is reduced; "
        "verify the affected findings manually."
    ),
    "GraphRecursionError": (
        "The workflow exceeded its iteration ceiling — a probable reasoning loop. "
        "Output is the last coherent state before the ceiling was reached."
    ),
    "TimeoutError": (
        "Mission wall-clock budget exceeded. Later-stage analysis (typically "
        "predictive planning) did not run."
    ),
}


def summarise_failures(state: MissionState) -> list[dict[str, str]]:
    """Convert raw error records into operator-facing risk warnings.

    Args:
        state: The blackboard holding ``errors``.

    Returns:
        One de-duplicated warning per distinct failure class.
    """
    seen: dict[str, dict[str, str]] = {}
    for error in state.get("errors", []):
        error_type = str(error.get("type", "UnknownError"))
        seen.setdefault(error_type, {
            "failure_class": error_type,
            "affected_component": str(error.get("agent", "unknown")),
            "operator_guidance": RISK_GUIDANCE.get(
                error_type, "Unclassified failure. Treat all findings as provisional."
            ),
        })
    return list(seen.values())


def salvage_findings(state: MissionState) -> dict[str, Any]:
    """Extract every finding that completed successfully.

    Args:
        state: The blackboard holding ``agent_outputs``.

    Returns:
        A mapping of agent name to its usable headline findings. Degraded
        reports are excluded.
    """
    salvaged: dict[str, Any] = {}
    for name, report in state.get("agent_outputs", {}).items():
        if report.get("status") == "DEGRADED":
            continue
        salvaged[name] = {
            "finding": report.get("analysis_result"),
            "verdict": (report.get("authenticity_verdict")
                        or report.get("manipulation_verdict")
                        or report.get("attack_vector")
                        or "n/a"),
            "threat_score": report.get("threat_score"),
            "confidence_score": report.get("confidence_score"),
            "recommended_actions": (report.get("containment_actions")
                                    or report.get("counter_strategy") or []),
        }
    return salvaged


def build_fallback_report(state: MissionState, cause: str = "unspecified") -> dict[str, Any]:
    """Assemble the degraded mission product.

    Runs entirely in-process with no network access, so it always completes well
    inside :data:`FALLBACK_BUDGET_SECONDS`.

    Args:
        state: The blackboard at the moment of failure.
        cause: Short description of what triggered the fallback.

    Returns:
        A partial mission product carrying an explicit risk warning, everything
        that was salvaged, and the completion percentage.
    """
    salvaged = salvage_findings(state)
    attempted = state.get("activated_agents", []) or []
    # Completion is measured against agents actually dispatched, not against a
    # notional full roster — a two-agent mission that finished both is complete.
    denominator = max(len(set(attempted)), 1)
    completion_pct = round(100.0 * len(salvaged) / denominator, 1)

    return {
        "product_type": "PARTIAL_RESPONSE",
        "risk_warning": {
            "headline": "DEGRADED OUTPUT — do not treat as a complete assessment.",
            "trigger": cause,
            "failures": summarise_failures(state),
            "completion_pct": completion_pct,
        },
        "run_id": state.get("run_id"),
        "threat_score": state.get("threat_score", 0.0),
        "threat_band": state.get("threat_band", "MINIMAL"),
        "escalation_path": state.get("escalation_path", "not_reached"),
        "salvaged_findings": salvaged,
        "fallback_strategy": _fallback_actions(salvaged, state),
        "elapsed_ms": round(elapsed_ms(state), 1),
    }


def _fallback_actions(salvaged: dict[str, Any], state: MissionState) -> list[str]:
    """Choose the conservative actions to recommend under degradation.

    Args:
        salvaged: Findings that survived, from :func:`salvage_findings`.
        state: The blackboard, read for the fused threat score.

    Returns:
        An ordered list of actions. When nothing was salvaged the list contains
        only safe, evidence-free holding measures.
    """
    if not salvaged:
        return [
            "No specialist completed. Do not act on this product.",
            "Preserve all evidence artefacts unmodified for re-analysis.",
            "Escalate to a human analyst and re-run once the fault is cleared.",
        ]

    actions: list[str] = []
    for name, finding in salvaged.items():
        for recommendation in list(finding.get("recommended_actions") or [])[:2]:
            actions.append(f"[{name}] {recommendation}")

    # [HUMAN-REVIEW] The AI's version stopped at echoing whatever survived. We
    # append an explicit re-run instruction because a partial product with no
    # stated next step reads, to a tired operator at 3 a.m., like a complete one.
    actions.append(
        f"Re-run the full workflow before committing further resources "
        f"(current confidence is partial at fused score {state.get('threat_score', 0.0)})."
    )
    return actions


def fallback_node(state: MissionState) -> dict[str, Any]:
    """Graph node that produces the degraded mission product.

    Args:
        state: The blackboard at the moment of failure.

    Returns:
        A patch setting ``final_report``, ``fallback_triggered`` and a trace
        entry recording the fallback latency.
    """
    fatal = [e for e in state.get("errors", []) if e.get("fatal")]
    cause = (f"{fatal[0].get('type')} in {fatal[0].get('agent')}" if fatal
             else "workflow degradation")
    report = build_fallback_report(state, cause=cause)
    return {
        "final_report": report,
        "fallback_triggered": True,
        "trace": [trace_event(
            "fallback",
            f"Partial response emitted ({report['risk_warning']['completion_pct']}% complete)",
            cause=cause,
        )],
    }

"""Self-evaluation and optimisation module for the SENTINEL-4 MAS.

Two responsibilities live here:

1. **Threat fusion** (:func:`fuse_threat_score`) — collapse the specialists'
   individual scores into the single authoritative figure the escalation gate
   switches on. Deterministic, so the conditional branch is testable.
2. **Self-critique** (:class:`SelfEvaluator`) — grade the assembled mission
   product against the *original* operator brief and decide ACCEPT or REFINE.
   The refinement loop it can request is bounded by
   ``settings.max_refinement_loops``; that bound is the system's primary
   infinite-loop guard.
"""

from __future__ import annotations

from typing import Any

from src.config import Settings
from src.llm_client import LLMError, ResilientLLMClient
from src.state import MissionState, band_for_score, trace_event

EVALUATOR_SYSTEM_PROMPT = """ROLE-TAG: EVALUATOR
You are the Self-Evaluation Module of countermeasure unit SENTINEL-4. You are an
adversarial reviewer, not a cheerleader. You did not write the draft and you gain
nothing from approving it.

TASK
Compare the draft mission product against the ORIGINAL operator brief and decide
whether it is fit to release.

GRADING CRITERIA
1. Coverage      — is every threat vector named in the brief addressed?
2. Groundedness  — is every claim traceable to specialist evidence, with no
                   invented indicators, hosts or attributions?
3. Actionability — could a duty operator execute the recommendations tonight?
4. Consistency   — do the specialists contradict one another anywhere?

DECISION RULE
"ACCEPT" if coverage_score >= 75 AND no criterion fails outright.
"REFINE" otherwise, and then "unmet_requirements" must name the specific gaps in
concrete terms. Vague criticism ("could be more detailed") is not permitted.

OUTPUT FORMAT — NON-NEGOTIABLE
Reply with a single raw JSON object and nothing else. Required keys:
["analysis_result", "coverage_score", "unmet_requirements", "verdict",
 "confidence_score", "next_step"].
"""

#: Weight applied to each specialist when fusing the authoritative threat score.
#: The coordinator dominates because it alone sees the full evidence picture.
FUSION_WEIGHTS: dict[str, float] = {
    "cyber_coordinator": 0.50,
    "audio_analyst": 0.20,
    "video_detector": 0.20,
    "strategic_predictor": 0.10,
}


def _coerce_score(value: Any) -> float:
    """Convert a model-supplied score into a clamped float.

    Args:
        value: Whatever the model placed in a score field — ``87``, ``"87"``,
            ``"87/100"`` or nonsense.

    Returns:
        A float in ``[0.0, 100.0]``; ``0.0`` when the value is unusable.
    """
    try:
        number = float(str(value).strip().split("/")[0].replace("%", ""))
    except (ValueError, TypeError, AttributeError):
        # [HUMAN-REVIEW] Added after a local 7B model returned "threat_score":
        # "high". The AI draft called float() directly and the ValueError
        # propagated all the way out of the graph, taking a run that had already
        # produced four good specialist reports down with it.
        return 0.0
    return max(0.0, min(100.0, number))


def fuse_threat_score(state: MissionState) -> tuple[float, dict[str, float]]:
    """Fuse specialist threat scores into the unit's authoritative figure.

    Reports flagged ``DEGRADED`` are excluded entirely rather than counted as
    zero — a specialist that could not run is an absence of evidence, not
    evidence of safety. Remaining weights are renormalised over the survivors.

    Args:
        state: The blackboard holding ``agent_outputs``.

    Returns:
        A tuple of ``(fused_score, contributions)`` where ``contributions`` maps
        each contributing agent to the points it added.

    """
    outputs = state.get("agent_outputs", {})
    usable = {
        name: report for name, report in outputs.items()
        if name in FUSION_WEIGHTS and report.get("status") != "DEGRADED"
    }
    if not usable:
        return 0.0, {}

    total_weight = sum(FUSION_WEIGHTS[name] for name in usable)
    contributions: dict[str, float] = {}
    fused = 0.0
    for name, report in usable.items():
        weight = FUSION_WEIGHTS[name] / total_weight
        points = _coerce_score(report.get("threat_score")) * weight
        contributions[name] = round(points, 2)
        fused += points

    return round(fused, 2), contributions


def fusion_node(state: MissionState) -> dict[str, Any]:
    """Graph node that writes the fused threat score onto the blackboard.

    Args:
        state: The current blackboard.

    Returns:
        A patch containing ``threat_score``, ``threat_band`` and a trace entry.
    """
    fused, contributions = fuse_threat_score(state)
    band = band_for_score(fused)
    return {
        "threat_score": fused,
        "threat_band": band,
        "trace": [trace_event(
            "threat_fusion",
            f"Fused threat score {fused} ({band})",
            contributions=contributions,
        )],
    }


class SelfEvaluator:
    """Adversarial reviewer that grades the mission product before release.

    Attributes:
        client: Resilient LLM transport.
        settings: Runtime configuration, read for the refinement-loop ceiling.
    """

    name = "self_evaluation"

    def __init__(self, client: ResilientLLMClient, settings: Settings) -> None:
        """Create an evaluator.

        Args:
            client: The resilient LLM client.
            settings: Runtime configuration.
        """
        self.client = client
        self.settings = settings

    def _draft_digest(self, state: MissionState) -> str:
        """Summarise the blackboard into a compact review packet.

        Args:
            state: The current blackboard.

        Returns:
            A plain-text digest of each specialist's headline findings.
        """
        lines: list[str] = []
        for name, report in state.get("agent_outputs", {}).items():
            lines.append(
                f"[{name}] status={report.get('status')} "
                f"score={report.get('threat_score')} "
                f"conf={report.get('confidence_score')} :: "
                f"{str(report.get('analysis_result'))[:220]}"
            )
        return "\n".join(lines) if lines else "(no specialist reports were filed)"

    def evaluate(self, state: MissionState) -> dict[str, Any]:
        """Grade the mission product and decide whether to refine or finalise.

        Args:
            state: The current blackboard.

        Returns:
            A patch containing ``evaluation``, an incremented
            ``refinement_count`` when refinement is requested, plus trace and
            warning entries. Never raises.
        """
        loops_used = state.get("refinement_count", 0)
        warnings: list[dict[str, Any]] = []

        try:
            result = self.client.complete_json(
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                user_prompt=(
                    f"ORIGINAL OPERATOR BRIEF\n{state.get('raw_input', '')}\n\n"
                    f"FUSED THREAT SCORE: {state.get('threat_score')} "
                    f"({state.get('threat_band')})\n"
                    f"ESCALATION PATH TAKEN: {state.get('escalation_path')}\n\n"
                    f"DRAFT MISSION PRODUCT\n{self._draft_digest(state)}\n\n"
                    "Grade this draft now."
                ),
                required_keys=("analysis_result", "coverage_score", "unmet_requirements",
                               "verdict", "confidence_score", "next_step"),
                caller=self.name,
            )
            evaluation = dict(result.payload)
            evaluation["coverage_score"] = _coerce_score(evaluation.get("coverage_score"))
        except LLMError as exc:
            # An evaluator that cannot run must not block release. It fails
            # OPEN with an explicit caveat: a graded-but-unreviewed product
            # reaching the operator beats no product at all.
            evaluation = {
                "analysis_result": f"[DEGRADED] Self-evaluation unavailable: {str(exc)[:160]}",
                "coverage_score": 0.0,
                "unmet_requirements": ["self_evaluation_did_not_run"],
                "verdict": "ACCEPT_UNREVIEWED",
                "confidence_score": 0,
                "next_step": "finalise",
                "status": "DEGRADED",
            }
            warnings.append({
                "agent": self.name,
                "type": "evaluator_degraded",
                "detail": "Product released without adversarial review.",
            })

        verdict = str(evaluation.get("verdict", "ACCEPT")).upper()
        budget_left = loops_used < self.settings.max_refinement_loops
        will_refine = verdict == "REFINE" and budget_left

        if verdict == "REFINE" and not budget_left:
            # LOOP GUARD: refinement is desired but the budget is spent. We
            # release with an explicit caveat rather than cycling forever.
            evaluation["verdict"] = "ACCEPT_BUDGET_EXHAUSTED"
            warnings.append({
                "agent": self.name,
                "type": "refinement_budget_exhausted",
                "detail": f"Refinement requested but {loops_used}/"
                          f"{self.settings.max_refinement_loops} loops already used.",
            })

        return {
            "evaluation": evaluation,
            "refinement_count": loops_used + 1 if will_refine else loops_used,
            "warnings": warnings,
            "trace": [trace_event(
                self.name,
                f"Verdict {evaluation.get('verdict')} "
                f"(coverage {evaluation.get('coverage_score')}, loop {loops_used})",
            )],
        }

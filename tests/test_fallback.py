"""Tests for the risk-warning / fallback subsystem and its latency budget."""

from __future__ import annotations

import time

import pytest

from src.config import load_settings
from src.fallback import FALLBACK_BUDGET_SECONDS, build_fallback_report, summarise_failures
from src.graph import SentinelWorkflow
from src.llm_client import ResilientLLMClient
from src.state import MissionState, new_mission_state

BRIEF = ("Synthetic voicemail impersonating the CFO, manipulated CCTV footage, "
         "outbound beaconing to 185.220.101.44, zero-day suspected with lateral movement.")
ARTEFACTS = ["cfo.wav", "cctv.mp4"]


def _client_with(channel: str, caller: str) -> tuple[object, ResilientLLMClient]:
    """Build settings plus a client with one injection channel armed.

    Args:
        channel: Attribute name of the injection channel.
        caller: Agent name to fail.

    Returns:
        A ``(settings, client)`` tuple ready to pass to the workflow.
    """
    settings = load_settings()
    client = ResilientLLMClient(settings)
    getattr(client, channel).add(caller)
    return settings, client


def test_endpoint_outage_produces_partial_response_not_a_crash():
    """An outage at the fusion agent degrades the product instead of raising."""
    settings, client = _client_with("inject_unavailable", "cyber_coordinator")
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)

    report = state["final_report"]
    assert state["fallback_triggered"] is True
    assert report["product_type"] == "PARTIAL_RESPONSE"
    assert report["risk_warning"]["failures"][0]["failure_class"] == "LLMUnavailableError"
    # The partial response must still carry the work that did complete.
    assert set(report["salvaged_findings"]) == {"audio_analyst", "video_detector"}


def test_fallback_report_is_built_well_inside_the_five_second_budget():
    """The brief requires a fallback output in under five seconds."""
    state: MissionState = new_mission_state(BRIEF, ARTEFACTS)
    state["agent_outputs"] = {
        "audio_analyst": {"status": "OK", "analysis_result": "synthetic speech",
                          "threat_score": 88, "confidence_score": 86,
                          "authenticity_verdict": "SYNTHETIC"},
        "cyber_coordinator": {"status": "DEGRADED"},
    }
    state["activated_agents"] = ["audio_analyst", "cyber_coordinator"]
    state["errors"] = [{"agent": "cyber_coordinator", "type": "LLMUnavailableError",
                        "fatal": True}]

    started = time.monotonic()
    report = build_fallback_report(state, cause="unit test")
    elapsed = time.monotonic() - started

    assert elapsed < FALLBACK_BUDGET_SECONDS
    assert elapsed < 0.1, "fallback path must be pure in-process work, no network"
    assert report["risk_warning"]["completion_pct"] == 50.0


def test_end_to_end_fallback_latency_stays_within_budget():
    """Measured from mission start, a failed run still answers inside the budget."""
    settings, client = _client_with("inject_unavailable", "cyber_coordinator")
    started = time.monotonic()
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    assert state["fallback_triggered"] is True
    assert elapsed < FALLBACK_BUDGET_SECONDS


def test_malformed_json_degrades_one_agent_but_the_mission_completes():
    """A parse failure is contained to the agent that caused it."""
    settings, client = _client_with("inject_malformed", "audio_analyst")
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)

    assert state["agent_outputs"]["audio_analyst"]["status"] == "DEGRADED"
    assert state["agent_outputs"]["audio_analyst"]["failure_reason"] == "LLMParseError"
    assert state["final_report"]["product_type"] == "FULL_ASSESSMENT"


def test_contract_breach_is_caught_as_a_distinct_failure_class():
    """Valid JSON missing a promised key degrades that agent specifically."""
    settings, client = _client_with("inject_contract_breach", "video_detector")
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)

    assert state["agent_outputs"]["video_detector"]["failure_reason"] == "LLMContractError"


def test_infinite_loop_is_stopped_by_the_recursion_ceiling():
    """A graph that never converges is halted and salvaged, not left spinning."""

    class LoopingWorkflow(SentinelWorkflow):
        """Workflow whose evaluation gate never accepts."""

        def _evaluation_gate(self, state: MissionState) -> str:
            """Always request another refinement pass.

            Args:
                state: The current blackboard (unused).

            Returns:
                Always ``"threat_fusion"``.
            """
            return "threat_fusion"

    started = time.monotonic()
    state = LoopingWorkflow().run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    assert state["fallback_triggered"] is True
    assert any(e["type"] == "GraphRecursionError" for e in state["errors"])
    # Streaming preserves in-flight work, so the partial product is not empty.
    assert state["final_report"]["salvaged_findings"]
    assert elapsed < FALLBACK_BUDGET_SECONDS


def test_refinement_loop_respects_its_budget():
    """The self-evaluation loop cannot exceed the configured ceiling."""
    settings = load_settings()
    state = SentinelWorkflow(settings=settings).run(BRIEF, ARTEFACTS)
    assert state["refinement_count"] <= settings.max_refinement_loops


def test_fallback_with_nothing_salvageable_gives_safe_holding_advice():
    """When no specialist completed, the product refuses to recommend action."""
    state = new_mission_state(BRIEF)
    state["errors"] = [{"agent": "router", "type": "LLMUnavailableError", "fatal": True}]
    report = build_fallback_report(state, cause="total outage")

    assert report["salvaged_findings"] == {}
    assert "Do not act on this product." in report["fallback_strategy"][0]


def test_failure_summaries_are_deduplicated_and_carry_guidance():
    """Repeated failures of one class collapse into a single operator warning."""
    state = new_mission_state(BRIEF)
    state["errors"] = [
        {"agent": "audio_analyst", "type": "LLMParseError"},
        {"agent": "video_detector", "type": "LLMParseError"},
    ]
    summaries = summarise_failures(state)
    assert len(summaries) == 1
    assert summaries[0]["operator_guidance"]


@pytest.mark.parametrize("scenario_path, expected_branch", [
    ("scenarios/scenario_multi_vector.json", "strategic_predictor"),
    ("scenarios/scenario_audio_only.json", "strategic_predictor"),
    ("scenarios/scenario_low_threat.json", "standard_defense"),
])
def test_shipped_scenarios_take_their_documented_branch(scenario_path, expected_branch):
    """Each demo scenario behaves as its own `expected_path` field claims."""
    from main import load_scenario

    brief, evidence_refs, _ = load_scenario(scenario_path)
    state = SentinelWorkflow().run(brief, evidence_refs)
    assert state["escalation_path"] == expected_branch
    assert state["final_report"]["product_type"] == "FULL_ASSESSMENT"

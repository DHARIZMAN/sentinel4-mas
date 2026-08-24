"""Tests for the dynamic router, its static fallback, and the conditional gates."""

from __future__ import annotations

import pytest

from src.config import ModelRegistry, ModelRoutingError, load_settings
from src.evaluation import fuse_threat_score
from src.graph import SentinelWorkflow
from src.llm_client import ResilientLLMClient
from src.router import MissionRouter
from src.state import new_mission_state
from src.tools.registry import build_default_registry


@pytest.fixture()
def router():
    """Provide a router wired to the offline mock engine.

    Returns:
        A :class:`~src.router.MissionRouter`.
    """
    settings = load_settings()
    return MissionRouter(ResilientLLMClient(settings), build_default_registry())


def test_dynamic_routing_selects_both_media_specialists(router):
    """A brief naming voice and footage activates both media specialists."""
    state = new_mission_state("A cloned voice call and manipulated CCTV footage.")
    decision = router.route(state)["route_decision"]
    assert decision["mode"] == "DYNAMIC_SEMANTIC"
    assert {"audio_analyst", "video_detector", "cyber_coordinator"} == set(
        decision["selected_agents"]
    )


def test_dynamic_routing_leaves_irrelevant_specialists_idle(router):
    """A network-only brief does not activate media specialists."""
    decision = router.route(new_mission_state(
        "Unusual outbound traffic from a database host. No media of any kind."
    ))["route_decision"]
    assert "audio_analyst" not in decision["selected_agents"]
    assert "video_detector" not in decision["selected_agents"]


def test_router_degrades_to_static_when_endpoint_fails():
    """When semantic routing fails, keyword routing takes over and says so."""
    settings = load_settings()
    client = ResilientLLMClient(settings)
    client.inject_unavailable.add("router")
    degraded_router = MissionRouter(client, build_default_registry())

    patch = degraded_router.route(new_mission_state("Suspicious voicemail and CCTV clip."))
    decision = patch["route_decision"]

    assert decision["mode"] == "STATIC_FALLBACK"
    assert set(decision["selected_agents"]) == {
        "audio_analyst", "video_detector", "cyber_coordinator"
    }
    assert any(w["type"] == "router_degraded_to_static" for w in patch["warnings"])


def test_router_discards_hallucinated_agent_names(router):
    """An invented specialist name is filtered out of the dispatch list."""
    cleaned = router.sanitise(["audio_analyst", "malware_reverse_engineer", "ghost_agent"])
    assert cleaned == ["audio_analyst", "cyber_coordinator"]


def test_coordinator_is_always_dispatched(router):
    """The fusion point is unconditional, even when the LLM omits it."""
    assert "cyber_coordinator" in router.sanitise([])
    assert "cyber_coordinator" in router.static_route("nothing relevant here")["selected_agents"]


def test_embedding_model_in_chat_slot_is_refused():
    """The misrouting guard the brief warns about is enforced at construction."""
    with pytest.raises(ModelRoutingError):
        ModelRegistry(chat_model="text-embedding-nomic-embed-text-v1.5",
                      embed_model="text-embedding-nomic-embed-text-v1.5")


def test_resilience_budgets_are_clamped_to_brief_limits():
    """Timeout can never exceed 30 s, nor retries 3, whatever the environment says."""
    settings = load_settings()
    assert settings.request_timeout <= 30.0
    assert settings.max_retries <= 3


@pytest.mark.parametrize(
    "brief, expected_branch",
    [
        ("Zero-day exploitation, ransomware staging, exfiltration to 185.220.101.44, "
         "deepfake voicemail and lateral movement across the SCADA segment.",
         "strategic_predictor"),
        ("An internal email used an older logo. It passes DMARC, has no links, no "
         "attachments and no requests for credentials. Nothing anomalous in the logs. "
         "Advise on proportionate handling of this routine report.",
         "standard_defense"),
    ],
)
def test_escalation_gate_selects_the_correct_branch(brief, expected_branch):
    """CONDITIONAL PATH 3: the threat score decides which strategy agent runs."""
    state = SentinelWorkflow().run(brief)
    assert state["escalation_path"] == expected_branch
    assert expected_branch in state["agent_outputs"]


def test_degraded_reports_are_excluded_from_fusion():
    """A specialist that failed contributes no points and no false reassurance."""
    state = new_mission_state("x")
    state["agent_outputs"] = {
        "cyber_coordinator": {"threat_score": 90, "status": "OK"},
        "audio_analyst": {"threat_score": 0, "status": "DEGRADED"},
    }
    fused, contributions = fuse_threat_score(state)
    assert fused == 90.0
    assert "audio_analyst" not in contributions


def test_non_numeric_threat_score_does_not_crash_fusion():
    """Regression: a model returning "high" instead of 87 must not kill the run."""
    state = new_mission_state("x")
    state["agent_outputs"] = {"cyber_coordinator": {"threat_score": "high", "status": "OK"}}
    fused, _ = fuse_threat_score(state)
    assert fused == 0.0

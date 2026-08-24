"""Supervisor / dispatcher for the SENTINEL-4 MAS.

**Design declaration (required by the assessment brief): this router is
DYNAMIC.** It performs LLM-driven semantic intent classification over the raw
operator brief and selects a specialist set from that intent.

It is dynamic *with a deterministic safety net*. If the semantic pass fails —
endpoint down, unparseable reply, contract breach, or a selection the router
knows to be unsafe — control falls to a static keyword matcher. The system
therefore keeps the discriminating power of semantic routing without inheriting
its single point of failure, and the mode actually used is recorded on the
blackboard so the operator always knows which one produced the dispatch.
"""

from __future__ import annotations

from typing import Any

from src.llm_client import LLMError, ResilientLLMClient
from src.state import MissionState, trace_event
from src.tools.registry import ToolRegistry

#: Specialists the router is allowed to dispatch to. Anything outside this set
#: coming back from the LLM is treated as a hallucination and discarded.
DISPATCHABLE_AGENTS: frozenset[str] = frozenset(
    {"audio_analyst", "video_detector", "cyber_coordinator"}
)

#: Keyword table backing the static fallback path.
STATIC_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "audio_analyst": ("audio", "voice", "voicemail", "call", "recording", "speech",
                      "spoken", "caller", "phone", "dialled", "vishing"),
    "video_detector": ("video", "footage", "stream", "cctv", "broadcast", "clip",
                       "frame", "webcam", "recording of", "livestream"),
}

ROUTER_SYSTEM_PROMPT = """ROLE-TAG: ROUTER
You are the Supervisor of countermeasure unit SENTINEL-4. You do not analyse
evidence yourself. Your single job is to read an operator's incident brief and
decide which specialists must be activated.

AVAILABLE SPECIALISTS
- audio_analyst     : speech-audio authenticity. Activate when the brief mentions
                      a call, voicemail, recording, spoken instruction or voice.
- video_detector    : video manipulation. Activate when the brief mentions
                      footage, a stream, CCTV, a broadcast or a clip.
- cyber_coordinator : evidence fusion, attack-vector verification, containment.
                      ALWAYS activate. Every mission needs a fusion point, and
                      downstream planning depends on this agent's verdict.

ROUTING DOCTRINE
- Activate a media specialist only when that modality is actually present. Idle
  specialists cost latency and dilute the fused assessment with null findings.
- When a modality is merely implied ("they contacted the helpdesk"), do NOT
  activate a media specialist; the coordinator handles unspecified contact.
- Classify "intent" as one of: multi_vector_incident, media_deception_only,
  network_intrusion_only, reconnaissance_probe, ambiguous_report.

OUTPUT FORMAT — NON-NEGOTIABLE
Reply with a single raw JSON object and nothing else. Required keys:
["intent", "selected_agents", "confidence_score", "rationale", "next_step"].
"selected_agents" must be an array drawn ONLY from the three names above.
"confidence_score" must be an integer 0-100.
"""


class MissionRouter:
    """Dynamic semantic dispatcher with a deterministic static fallback.

    Attributes:
        client: Resilient LLM transport used for the semantic pass.
        registry: Tool registry (unused by the router itself, kept so the router
            has the same construction signature as the agents).
    """

    name = "router"

    def __init__(self, client: ResilientLLMClient, registry: ToolRegistry) -> None:
        """Create a router bound to a transport and registry.

        Args:
            client: The resilient LLM client.
            registry: The shared tool registry.
        """
        self.client = client
        self.registry = registry

    def static_route(self, brief: str) -> dict[str, Any]:
        """Select specialists using deterministic keyword rules.

        This is the fallback path. It is intentionally simple, dependency-free
        and incapable of failing, because it is what runs when everything else
        already has.

        Args:
            brief: The operator's incident brief.

        Returns:
            A routing decision dict with ``mode`` set to ``"STATIC_FALLBACK"``.
        """
        lowered = brief.lower()
        selected = [
            agent for agent, keywords in STATIC_KEYWORD_MAP.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        # The coordinator is unconditional: it is the fusion point on which the
        # escalation gate and the predictor both depend.
        selected.append("cyber_coordinator")

        return {
            "mode": "STATIC_FALLBACK",
            "intent": "multi_vector_incident" if len(selected) > 2 else "single_channel_incident",
            "selected_agents": selected,
            "confidence_score": 55,  # deliberately modest: keywords, not meaning
            "rationale": "Semantic routing unavailable; dispatched on keyword rules.",
            "next_step": "dispatch",
        }

    def sanitise(self, raw_selection: Any) -> list[str]:
        """Filter an LLM-proposed agent list down to dispatchable names.

        Args:
            raw_selection: Whatever the model put in ``selected_agents``.

        Returns:
            A de-duplicated list of valid agent names, always including
            ``cyber_coordinator``.
        """
        # [HUMAN-REVIEW] The AI trusted selected_agents verbatim and passed it
        # straight to the graph, so a hallucinated name such as
        # "malware_reverse_engineer" produced a KeyError deep inside dispatch.
        # We now intersect against DISPATCHABLE_AGENTS here, which turns a class
        # of hard crash into a silently-corrected routing decision.
        if not isinstance(raw_selection, list):
            return ["cyber_coordinator"]
        cleaned = [
            str(name).strip().lower()
            for name in raw_selection
            if str(name).strip().lower() in DISPATCHABLE_AGENTS
        ]
        if "cyber_coordinator" not in cleaned:
            cleaned.append("cyber_coordinator")
        return list(dict.fromkeys(cleaned))

    def route(self, state: MissionState) -> dict[str, Any]:
        """Produce the dispatch decision for a mission.

        Attempts semantic routing first and degrades to static rules on any
        failure. Never raises.

        Args:
            state: The blackboard, read for ``raw_input``.

        Returns:
            A blackboard patch carrying ``route_decision``, a trace entry, and
            any warning raised while degrading.
        """
        brief = state.get("raw_input", "")
        warnings: list[dict[str, Any]] = []

        try:
            result = self.client.complete_json(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=f"OPERATOR INCIDENT BRIEF\n{brief}\n\nProduce your routing decision.",
                required_keys=("intent", "selected_agents", "confidence_score",
                               "rationale", "next_step"),
                caller=self.name,
            )
            decision = dict(result.payload)
            decision["mode"] = "DYNAMIC_SEMANTIC"
            proposed = decision.get("selected_agents", [])
            decision["selected_agents"] = self.sanitise(proposed)

            if len(decision["selected_agents"]) != len(proposed or []):
                warnings.append({
                    "agent": self.name,
                    "type": "invalid_agent_name_discarded",
                    "detail": f"Router proposed {proposed}; dispatched "
                              f"{decision['selected_agents']}.",
                })
        except LLMError as exc:
            decision = self.static_route(brief)
            decision["degradation_cause"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            warnings.append({
                "agent": self.name,
                "type": "router_degraded_to_static",
                "detail": decision["degradation_cause"],
            })

        return {
            "route_decision": decision,
            "warnings": warnings,
            "trace": [trace_event(
                self.name,
                f"Routing mode {decision['mode']} -> {decision['selected_agents']}",
                intent=decision.get("intent"),
                confidence=decision.get("confidence_score"),
            )],
        }

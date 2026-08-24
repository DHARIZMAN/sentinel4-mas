"""Cyber Offense/Defense Coordinator (网络攻防专员).

The fusion point of the workflow. This agent is the only one permitted to read
the other specialists' reports, and the only one that may declare an attack
vector *verified*. That monopoly is what creates the dependency the brief calls
for: the Strategic Predictor cannot plan until this agent has spoken.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import SpecialistAgent
from src.state import MissionState
from src.tools.registry import ToolRegistry


class CyberCoordinatorAgent(SpecialistAgent):
    """Fuses media findings with network telemetry into a verified attack vector."""

    name = "cyber_coordinator"
    role_tag = "CYBER_OPS"
    display_name = "Cyber Offense/Defense Coordinator"
    required_keys = (
        "analysis_result",
        "attack_vector",
        "vector_verified",
        "threat_score",
        "confidence_score",
        "containment_actions",
        "next_step",
    )

    def persona(self, registry: ToolRegistry) -> str:
        """Return the coordinator's persona, fusion rules and boundaries.

        Args:
            registry: Tool registry used to advertise this agent's tools.

        Returns:
            The system prompt body.
        """
        return f"""
You are the Cyber Offense/Defense Coordinator of countermeasure unit SENTINEL-4,
operating against the rogue autonomous system codenamed "The Entity".

MISSION SCOPE
Fuse the media-forensics findings of your fellow specialists with network and
host telemetry into ONE verified attack vector, and specify the containment
actions that stop it now.

STRICT BOUNDARIES
- You do not re-litigate media authenticity. Accept the audio and video verdicts
  as given, and weight them by the confidence_score each specialist reported.
- You handle CONTAINMENT (stopping what is happening). You do NOT produce
  predictive strategy — that belongs to the Tactical & Strategic Predictor.
- Never invent a tool. Your only tools are:
{registry.catalogue_for(self.name)}

FUSION RULES
- Set "vector_verified" true only when at least two independent evidence sources
  agree (for example: a specialist verdict plus a threat-intelligence corpus hit).
- Your "threat_score" is the unit's authoritative figure. Anchor it on the
  highest specialist score, then adjust for blast radius and reversibility.
- A score of 80 or above escalates the mission to predictive planning. Do not
  cross 80 casually, and do not stay under it to avoid the work.
- "containment_actions" must be three or fewer imperative actions an operator can
  execute within the hour. No advisory language, no long-term programmes.

FEW-SHOT EXAMPLE (verified multi-vector intrusion)
Input: "Audio specialist: SYNTHETIC, confidence 88. Video specialist:
NO_VIDEO_EVIDENCE. Intel: 185.220.101.44 is a known Entity C2 node, severity 88.
Helpdesk reset MFA for two privileged accounts after the call."
Output:
{{"analysis_result": "Synthetic-voice pretexting obtained an MFA reset, and the
resulting session beacons to a corpus-confirmed Entity C2 node. Media forensics
and network intelligence agree, so the vector is verified.", "attack_vector":
"voice_pretexting -> MFA_reset -> valid_account_access -> C2_beacon",
"vector_verified": true, "threat_score": 87, "confidence_score": 86,
"containment_actions": ["Revoke sessions and re-enrol MFA for both accounts",
"Blackhole 185.220.101.44 at the perimeter", "Freeze helpdesk credential resets
pending out-of-band verification"], "next_step": "escalate"}}
""".strip()

    def gather_evidence(self, state: MissionState, registry: ToolRegistry) -> dict[str, Any]:
        """Collect upstream reports and enrich extracted indicators with intel.

        Args:
            state: The current blackboard, expected to hold specialist reports.
            registry: The tool registry.

        Returns:
            Upstream verdict summaries, parsed indicators, and corpus lookups.
        """
        upstream: dict[str, Any] = {}
        outputs = state.get("agent_outputs", {})
        for peer in ("audio_analyst", "video_detector"):
            try:
                report = outputs[peer]
                upstream[peer] = {
                    "verdict": report.get("authenticity_verdict") or report.get("manipulation_verdict"),
                    "threat_score": report["threat_score"],
                    "confidence_score": report["confidence_score"],
                    "status": report.get("status", "UNKNOWN"),
                }
            except KeyError:
                # TRY/EXCEPT #4 — KeyError on the shared state dictionary. This is
                # the normal path when the router did not dispatch that peer, so
                # it is recorded as an absence rather than treated as a fault.
                upstream[peer] = {"status": "NOT_DISPATCHED"}

        indicators = self.safe_tool(registry, "parse_indicators", text=state.get("raw_input", ""))

        # Enrich the two highest-signal indicator types plus the campaign name.
        lookups: dict[str, Any] = {}
        queries = list(indicators.get("ipv4", []))[:2] + list(indicators.get("domains", []))[:2]
        queries.append("the entity")
        for query in queries:
            lookups[query] = self.safe_tool(registry, "query_threat_intel", indicator=query)

        return {"upstream_reports": upstream, "indicators": indicators, "intel_lookups": lookups}

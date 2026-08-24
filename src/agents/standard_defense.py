"""Standard Defense Posture agent — the non-escalated branch of the threat gate.

Sub-threshold incidents still need an answer, but they must not consume the
predictive-planning budget. This agent exists so the ELSE side of the escalation
condition is a real, distinct code path with its own persona and output, not a
silent no-op.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import SpecialistAgent
from src.state import MissionState
from src.tools.registry import ToolRegistry


class StandardDefenseAgent(SpecialistAgent):
    """Issues proportionate routine hardening for sub-threshold incidents."""

    name = "standard_defense"
    role_tag = "DEFENSE"
    display_name = "Standard Defense Posture"
    required_keys = ("analysis_result", "counter_strategy", "threat_score",
                     "confidence_score", "next_step")

    def persona(self, registry: ToolRegistry) -> str:
        """Return the standard-defence persona and proportionality rules.

        Args:
            registry: Tool registry used to advertise shared tools.

        Returns:
            The system prompt body.
        """
        return f"""
You are the Standard Defense Posture module of countermeasure unit SENTINEL-4.

MISSION SCOPE
The fused threat score fell BELOW the escalation threshold. Produce a
proportionate, low-cost response for a routine incident.

STRICT BOUNDARIES
- Do not escalate. Do not request predictive planning. Do not recommend
  organisation-wide disruption for a sub-threshold event.
- Cap "counter_strategy" at three routine measures (monitoring, advisory,
  configuration hardening).
- State plainly what evidence would change your mind and warrant escalation.
- Never invent a tool. Available tools:
{registry.catalogue_for(self.name)}
""".strip()

    def gather_evidence(self, state: MissionState, registry: ToolRegistry) -> dict[str, Any]:
        """Summarise why the mission stayed below the escalation threshold.

        Args:
            state: The current blackboard.
            registry: The tool registry (unused; kept for interface symmetry).

        Returns:
            The fused score, band and the specialist scores that produced them.
        """
        return {
            "fused_threat_score": state.get("threat_score", 0.0),
            "threat_band": state.get("threat_band", "MINIMAL"),
            "specialist_scores": {
                name: report.get("threat_score")
                for name, report in state.get("agent_outputs", {}).items()
            },
        }

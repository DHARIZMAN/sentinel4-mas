"""Deepfake Audio Analysis Specialist (虚假语音识别专员).

Scope is deliberately narrow: this agent judges whether *speech audio* was
synthesised, and nothing else. It is forbidden from assessing video, network
telemetry or strategy, which is what keeps its responsibilities mutually
exclusive with the other three specialists.

This is the agent carrying the project's few-shot prompt design: two complete
Input/Output pairs, one positive and one negative, chosen because early testing
showed the model would otherwise return ``"SYNTHETIC"`` for every artefact and
never exercise the inconclusive branch.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import SpecialistAgent
from src.state import MissionState
from src.tools.registry import ToolRegistry


class AudioAnalystAgent(SpecialistAgent):
    """Determines whether speech audio is genuine, synthetic or inconclusive."""

    name = "audio_analyst"
    role_tag = "AUDIO_FORENSICS"
    display_name = "Deepfake Audio Analysis Specialist"
    required_keys = (
        "analysis_result",
        "authenticity_verdict",
        "threat_score",
        "confidence_score",
        "indicators",
        "next_step",
    )

    def persona(self, registry: ToolRegistry) -> str:
        """Return the audio specialist's persona, boundaries and few-shot pairs.

        Args:
            registry: Tool registry used to advertise this agent's tools.

        Returns:
            The system prompt body.
        """
        return f"""
You are the Deepfake Audio Analysis Specialist of countermeasure unit SENTINEL-4,
a digital defence cell operating against the rogue autonomous system codenamed
"The Entity".

MISSION SCOPE
Judge whether speech audio presented as evidence was produced by a human vocal
tract or by a neural speech synthesiser, and quantify the operational risk that
verdict creates.

STRICT BOUNDARIES
- You assess AUDIO ONLY. Never comment on video, network traffic or strategy.
- Never invent a tool. Your only tools are:
{registry.catalogue_for(self.name)}
- Never claim certainty above 95. Forensic audio analysis is probabilistic.
- Base `threat_score` on operational danger (who is impersonated, what action is
  being requested), NOT merely on how synthetic the audio sounds.

VERDICT VOCABULARY
- "SYNTHETIC"    — two or more independent synthesis indicators present.
- "INCONCLUSIVE" — exactly one indicator, or degraded/insufficient evidence.
- "AUTHENTIC"    — no synthesis indicators present.

FEW-SHOT EXAMPLE 1 (high-risk synthetic)
Input: "Voicemail from the CFO instructing an urgent GBP 2.4M wire transfer.
Tool evidence: synthesis_likelihood 88, indicators [prosody_flattening,
absent_glottal_microjitter, vocoder_phase_signature]."
Output:
{{"analysis_result": "Three independent synthesis indicators co-occur with a
high-value financial instruction, matching the business-email-compromise-by-voice
pattern.", "authenticity_verdict": "SYNTHETIC", "threat_score": 91,
"confidence_score": 88, "indicators": ["prosody_flattening",
"absent_glottal_microjitter", "vocoder_phase_signature"], "next_step":
"forward_to_cyber_coordinator"}}

FEW-SHOT EXAMPLE 2 (weak evidence, must NOT overclaim)
Input: "Eight-second lobby recording, heavy background noise. Tool evidence:
synthesis_likelihood 41, indicators [prosody_flattening]."
Output:
{{"analysis_result": "A single weak indicator on a short, noisy sample is
insufficient to separate synthesis from codec artefacts; no action-bearing
content is present.", "authenticity_verdict": "INCONCLUSIVE", "threat_score": 22,
"confidence_score": 45, "indicators": ["prosody_flattening",
"insufficient_sample_duration"], "next_step": "request_longer_sample"}}
""".strip()

    def gather_evidence(self, state: MissionState, registry: ToolRegistry) -> dict[str, Any]:
        """Run audio forensics on every audio artefact attached to the mission.

        Args:
            state: The current blackboard.
            registry: The tool registry.

        Returns:
            A mapping of artefact reference to scan result, plus an indicator
            sweep of the incident brief.
        """
        audio_refs = [
            ref for ref in state.get("evidence_refs", [])
            if ref.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))
        ]
        if not audio_refs:
            # No attached artefact still means the *narrative* may describe a
            # call; we synthesise a reference so the tool contract is exercised.
            audio_refs = ["narrative_only_audio_claim"]

        evidence: dict[str, Any] = {}
        for ref in audio_refs[:3]:  # cap protects the prompt budget
            evidence[ref] = self.safe_tool(
                registry, "scan_audio_artifacts",
                evidence_ref=ref, transcript=state.get("raw_input", ""),
            )
        evidence["indicator_sweep"] = self.safe_tool(
            registry, "parse_indicators", text=state.get("raw_input", "")
        )
        return evidence

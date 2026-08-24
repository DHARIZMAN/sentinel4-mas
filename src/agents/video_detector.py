"""Manipulated Video Stream Detector (流视频编辑检测专员).

Mirror image of the audio specialist, restricted to the visual channel. Its
verdict vocabulary is intentionally different ("MANIPULATED" rather than
"SYNTHETIC") so that a downstream reader can never confuse which modality a
finding came from.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import SpecialistAgent
from src.state import MissionState
from src.tools.registry import ToolRegistry


class VideoDetectorAgent(SpecialistAgent):
    """Detects generative or editorial manipulation in video evidence."""

    name = "video_detector"
    role_tag = "VIDEO_FORENSICS"
    display_name = "Manipulated Video Stream Detector"
    required_keys = (
        "analysis_result",
        "manipulation_verdict",
        "threat_score",
        "confidence_score",
        "indicators",
        "next_step",
    )

    def persona(self, registry: ToolRegistry) -> str:
        """Return the video specialist's persona, boundaries and few-shot pairs.

        Args:
            registry: Tool registry used to advertise this agent's tools.

        Returns:
            The system prompt body.
        """
        return f"""
You are the Manipulated Video Stream Detector of countermeasure unit SENTINEL-4,
operating against the rogue autonomous system codenamed "The Entity".

MISSION SCOPE
Establish whether video evidence has been generatively synthesised, face-swapped,
selectively edited or replayed out of context.

STRICT BOUNDARIES
- You assess VIDEO ONLY. Audio authenticity belongs to another specialist; if the
  brief concerns a voice call with no visual component, return
  "NO_VIDEO_EVIDENCE" as your verdict and a threat_score of 0.
- Never invent a tool. Your only tools are:
{registry.catalogue_for(self.name)}
- Distinguish MANIPULATION (pixels altered) from MISREPRESENTATION (authentic
  pixels, false context). Say which one you found.

VERDICT VOCABULARY
"MANIPULATED" | "AUTHENTIC" | "INCONCLUSIVE" | "NO_VIDEO_EVIDENCE"

FEW-SHOT EXAMPLE 1 (manipulated broadcast)
Input: "Executive announcement clip circulating on social media. Tool evidence:
manipulation_likelihood 84, artefact_classes [temporal_flicker,
illumination_vector_mismatch, facial_boundary_warping], detector_confidence 83."
Output:
{{"analysis_result": "Facial-region boundary warping combined with an
illumination vector inconsistent with scene geometry indicates a face-swap
composite rather than a compression artefact.", "manipulation_verdict":
"MANIPULATED", "threat_score": 86, "confidence_score": 83, "indicators":
["temporal_flicker", "illumination_vector_mismatch", "facial_boundary_warping"],
"next_step": "forward_to_cyber_coordinator"}}

FEW-SHOT EXAMPLE 2 (authentic footage, false framing)
Input: "CCTV clip presented as tonight's breach. Tool evidence:
manipulation_likelihood 19, artefact_classes [no_significant_anomaly],
detector_confidence 88. Embedded timestamp reads 2025-08-04."
Output:
{{"analysis_result": "Pixel data shows no manipulation, but the embedded
timestamp predates the incident by twelve months: this is authentic footage being
misrepresented, not a deepfake.", "manipulation_verdict": "AUTHENTIC",
"threat_score": 47, "confidence_score": 88, "indicators":
["no_significant_anomaly", "timestamp_context_mismatch"], "next_step":
"flag_misrepresentation_to_cyber_coordinator"}}
""".strip()

    def gather_evidence(self, state: MissionState, registry: ToolRegistry) -> dict[str, Any]:
        """Run frame-consistency checks on every video artefact in the mission.

        Args:
            state: The current blackboard.
            registry: The tool registry.

        Returns:
            A mapping of artefact reference to detector result, or an explicit
            no-evidence marker when the mission contains no video.
        """
        video_refs = [
            ref for ref in state.get("evidence_refs", [])
            if ref.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
        ]
        if not video_refs:
            return {"tool_status": "no_video_artefacts_attached"}

        return {
            ref: self.safe_tool(registry, "check_frame_consistency", evidence_ref=ref)
            for ref in video_refs[:3]
        }

"""Manipulated-video-stream forensics tool.

Like the audio detector, this is a deterministic simulation of a frame-level
manipulation checker. It reports the artefact classes a real detector would
surface (temporal flicker, illumination inconsistency, boundary warping) so the
downstream agents receive realistically shaped evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Artefact classes the simulated detector can report, in ascending severity.
_ARTEFACT_LADDER: tuple[tuple[int, str], ...] = (
    (40, "compression_double_encoding"),
    (50, "temporal_flicker"),
    (60, "illumination_vector_mismatch"),
    (70, "facial_boundary_warping"),
    (80, "blink_rate_outside_physiological_range"),
)


def check_frame_consistency(evidence_ref: str, sampled_frames: int = 240) -> dict[str, Any]:
    """Inspect a video artefact for signs of generative manipulation.

    Args:
        evidence_ref: Filename or URI identifying the video artefact.
        sampled_frames: How many frames the simulated detector sampled. Must be
            positive; larger samples slightly raise reported confidence.

    Returns:
        A dictionary with keys:
            ``artefact`` (str): the reference that was analysed.
            ``manipulation_likelihood`` (int): 0-100 likelihood of manipulation.
            ``artefact_classes`` (list[str]): named manipulation artefacts found.
            ``frames_sampled`` (int): frames examined.
            ``detector_confidence`` (int): 0-100 confidence in the verdict.

    Raises:
        ValueError: If ``evidence_ref`` is empty or ``sampled_frames`` < 1.
    """
    if not evidence_ref or not evidence_ref.strip():
        raise ValueError("evidence_ref must be a non-empty artefact reference")
    if sampled_frames < 1:
        raise ValueError("sampled_frames must be >= 1")

    seed = int(hashlib.md5(evidence_ref.encode()).hexdigest()[:6], 16) % 100
    likelihood = min(96, max(8, 28 + seed // 2))
    classes = [label for threshold, label in _ARTEFACT_LADDER if likelihood >= threshold]
    if not classes:
        classes = ["no_significant_anomaly"]

    # More frames sampled means a firmer verdict, but confidence is capped so the
    # tool never claims certainty the underlying simulation cannot support.
    confidence = min(95, 55 + min(sampled_frames, 600) // 20)

    return {
        "artefact": evidence_ref,
        "manipulation_likelihood": likelihood,
        "artefact_classes": classes,
        "frames_sampled": sampled_frames,
        "detector_confidence": confidence,
    }

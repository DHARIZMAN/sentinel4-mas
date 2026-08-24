"""Resilient LLM transport layer for the SENTINEL-4 MAS.

Everything that can go wrong between an agent and a language model is handled
here, in one place, rather than being re-implemented in each agent:

* **Timeouts** — every request carries a wall-clock ceiling (<= 30 s per brief).
* **Retries** — bounded exponential back-off (<= 3 attempts per brief).
* **Malformed JSON** — agents are prompted for strict JSON; this layer repairs
  fenced/prefixed output and raises a typed error if repair fails.
* **Schema drift** — a response missing a contracted key is rejected loudly
  instead of poisoning the blackboard with ``None``.
* **Offline operation** — a deterministic mock engine lets the entire MAS run,
  and be graded, with no API key and no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.config import Settings

# Matches a ```json ... ``` fenced block, the single most common way an
# instruct model violates a "raw JSON only" instruction.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Matches the outermost {...} span, used when the model prepends chatter.
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    """Base class for every recoverable failure raised by this transport."""


class LLMUnavailableError(LLMError):
    """The endpoint could not be reached, or timed out, on every attempt."""


class LLMParseError(LLMError):
    """The model replied, but its output could not be coerced into JSON."""


class LLMContractError(LLMError):
    """Valid JSON arrived, but a key promised by the agent's contract is absent."""


@dataclass
class CallResult:
    """Outcome of a single successful structured LLM call.

    Attributes:
        payload: The parsed JSON object returned by the model.
        attempts: How many attempts were needed (1 means first-try success).
        latency_ms: Wall-clock duration of the whole call, retries included.
        repaired: Whether the raw text had to be repaired before parsing.
        provider: Which backend served the call.
    """

    payload: dict[str, Any]
    attempts: int
    latency_ms: float
    repaired: bool
    provider: str


def extract_json(raw_text: str) -> dict[str, Any]:
    """Coerce a model's raw reply into a JSON object.

    Tries three strategies in order: parse as-is, unwrap a markdown fence, then
    grab the outermost brace span.

    Args:
        raw_text: The unmodified text returned by the model.

    Returns:
        The parsed JSON object.

    Raises:
        LLMParseError: If no strategy yields a JSON *object*.
    """
    candidates: list[str] = [raw_text.strip()]

    fenced = _FENCE_RE.search(raw_text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    braced = _BRACE_RE.search(raw_text)
    if braced:
        candidates.append(braced.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # TRY/EXCEPT #1 — JSONDecodeError. Expected often enough with small
            # local models that it is control flow, not an exceptional event:
            # we simply move on to the next repair strategy.
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMParseError(
        f"No JSON object recoverable from model output (first 160 chars): {raw_text[:160]!r}"
    )


class MockInferenceEngine:
    """Deterministic offline stand-in for a chat model.

    The engine reads the agent's role tag out of the system prompt and returns a
    schema-correct report whose numbers are derived from keywords in the
    incident text. It exists so that the workflow logic — routing, dependencies,
    escalation, evaluation, fallback — can be demonstrated and unit-tested
    without a GPU, an API key, or network access.
    """

    #: Keywords that raise the modelled severity, and by how much.
    SEVERITY_MARKERS: dict[str, int] = {
        "zero-day": 22, "zero day": 22, "ransomware": 20, "exfiltration": 18,
        "lateral movement": 15, "privilege escalation": 15, "c2": 12,
        "command-and-control": 12, "deepfake": 14, "spoof": 10, "cloned": 12,
        "critical": 12, "scada": 20, "ics": 18, "grid": 16, "hospital": 14,
        "credential": 10, "phishing": 8, "the entity": 10, "multi-vector": 14,
    }

    #: Phrases that *lower* modelled severity. Without these, a brief that
    #: explicitly rules a threat out still scored highly, because the keyword
    #: counter cannot read negation.
    BENIGN_MARKERS: dict[str, int] = {
        "no links": 12, "no attachments": 10, "passes dmarc": 14,
        "nothing anomalous": 14, "routine": 10, "no requests for credentials": 16,
        "internal address": 8, "proportionate handling": 6,
    }

    def __init__(self, seed_salt: str = "sentinel4") -> None:
        """Initialise the engine.

        Args:
            seed_salt: Salt mixed into the hash used for stable pseudo-random
                jitter, so repeated runs of the same scenario are identical.
        """
        self.seed_salt = seed_salt

    def _stable_jitter(self, text: str, spread: int = 7) -> int:
        """Derive a repeatable pseudo-random offset from a string.

        Args:
            text: Text to hash.
            spread: Exclusive upper bound of the returned offset.

        Returns:
            An integer in ``[0, spread)`` that is identical for identical input.
        """
        digest = hashlib.sha256((self.seed_salt + text).encode()).hexdigest()
        return int(digest[:8], 16) % spread

    def _severity(self, text: str, floor: int, ceiling: int) -> int:
        """Score an incident description against the severity keyword table.

        Args:
            text: The incident text to score.
            floor: Minimum score to return.
            ceiling: Maximum score to return.

        Returns:
            A clamped integer severity in ``[floor, ceiling]``.
        """
        lowered = text.lower()
        score = floor + sum(w for k, w in self.SEVERITY_MARKERS.items() if k in lowered)
        score -= sum(w for k, w in self.BENIGN_MARKERS.items() if k in lowered)
        # The floor is applied to the *raw* floor, not to the de-escalated score,
        # so an explicitly benign brief is allowed to fall below it.
        return max(5, min(ceiling, score + self._stable_jitter(text)))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Produce a schema-correct JSON reply for the requesting agent.

        Args:
            system_prompt: The agent's system prompt; its ``ROLE-TAG:`` line
                selects which report shape to emit.
            user_prompt: The task payload, scanned for severity keywords.

        Returns:
            A JSON string matching the requesting agent's output contract.
        """
        role = "generic"
        tag = re.search(r"ROLE-TAG:\s*([A-Z0-9_]+)", system_prompt)
        if tag:
            role = tag.group(1).lower()
        builder = getattr(self, f"_build_{role}", self._build_generic)
        return json.dumps(builder(user_prompt), indent=2)

    # -- per-role report builders -------------------------------------------

    def _build_router(self, text: str) -> dict[str, Any]:
        """Build a routing verdict from the incident text.

        Args:
            text: The operator's incident brief.

        Returns:
            A routing decision dict matching the router's contract.
        """
        lowered = text.lower()
        # Negation guard: a brief that says "no video" mentions the word "video"
        # but contains none. Keyword matching alone activates an idle specialist,
        # so each modality is checked for an explicit denial first.
        denied_audio = any(p in lowered for p in ("no audio", "no voice", "no call", "no recording"))
        denied_video = any(p in lowered for p in ("no video", "no footage", "no cctv",
                                                  "without video", "voice-only", "voice only"))
        agents: list[str] = []
        if not denied_audio and any(
            k in lowered for k in ("audio", "voice", "call", "recording", "speech", "voicemail")
        ):
            agents.append("audio_analyst")
        if not denied_video and any(
            k in lowered for k in ("video", "footage", "stream", "cctv", "broadcast", "frame")
        ):
            agents.append("video_detector")
        # The Cyber Coordinator is always dispatched: it is the fusion point that
        # the Strategic Predictor depends upon.
        agents.append("cyber_coordinator")
        return {
            "intent": "multi_vector_incident" if len(agents) > 2 else "single_vector_incident",
            "selected_agents": agents,
            "confidence_score": min(95, 62 + 8 * len(agents) + self._stable_jitter(text, 5)),
            "rationale": (
                "Semantic scan matched "
                f"{len(agents)} specialist competency areas in the operator brief."
            ),
            "next_step": "dispatch",
        }

    def _build_audio_forensics(self, text: str) -> dict[str, Any]:
        """Build a deepfake-audio analysis report.

        Args:
            text: The task payload handed to the audio specialist.

        Returns:
            A report dict matching the audio analyst's contract.
        """
        score = self._severity(text, 30, 96)
        return {
            "analysis_result": (
                "Synthetic speech signature detected: prosody flattening and absent "
                "glottal micro-jitter consistent with a neural vocoder."
            ),
            "authenticity_verdict": "SYNTHETIC" if score >= 60 else "INCONCLUSIVE",
            "threat_score": score,
            "confidence_score": min(97, score + 3),
            "indicators": [
                "spectral_discontinuity@4.2kHz",
                "absent_breath_events",
                "phase_coherence_anomaly",
            ],
            "next_step": "forward_to_cyber_coordinator",
        }

    def _build_video_forensics(self, text: str) -> dict[str, Any]:
        """Build a manipulated-video detection report.

        Args:
            text: The task payload handed to the video specialist.

        Returns:
            A report dict matching the video detector's contract.
        """
        score = self._severity(text, 26, 94)
        return {
            "analysis_result": (
                "Frame-level inconsistency detected: illumination vector diverges "
                "from scene geometry across the facial region."
            ),
            "manipulation_verdict": "MANIPULATED" if score >= 55 else "AUTHENTIC",
            "threat_score": score,
            "confidence_score": min(96, score + 2),
            "indicators": ["temporal_flicker", "warped_boundary_artifacts", "eye_blink_rate_anomaly"],
            "next_step": "forward_to_cyber_coordinator",
        }

    def _build_cyber_ops(self, text: str) -> dict[str, Any]:
        """Build a fused cyber offence/defence assessment.

        Args:
            text: The task payload, including upstream specialist findings.

        Returns:
            A report dict matching the cyber coordinator's contract.
        """
        score = self._severity(text, 28, 97)
        return {
            "analysis_result": (
                "Correlated media-deception and network telemetry indicate a "
                "coordinated multi-vector intrusion attributable to The Entity."
            ),
            "attack_vector": "social_engineering_to_credential_access_to_lateral_movement",
            "vector_verified": True,
            "threat_score": score,
            "confidence_score": min(95, score),
            "containment_actions": [
                "Isolate affected VLAN segment",
                "Force credential rotation for privileged accounts",
                "Enable enhanced egress filtering on suspected C2 ranges",
            ],
            "next_step": "escalate" if score >= 80 else "standard_defense",
        }

    def _build_strategy(self, text: str) -> dict[str, Any]:
        """Build a predictive counter-strategy.

        Args:
            text: The task payload, including the verified attack vector.

        Returns:
            A report dict matching the strategic predictor's contract.
        """
        score = self._severity(text, 50, 98)
        return {
            "analysis_result": (
                "Adversary is expected to pivot to backup infrastructure within "
                "6-12 hours once primary C2 is severed."
            ),
            "predicted_next_moves": [
                "Fallback C2 activation over DNS tunnelling",
                "Destructive action against backup catalogues",
                "Secondary deepfake targeting incident-response leadership",
            ],
            "counter_strategy": [
                "Pre-position DNS sinkhole before severing primary C2",
                "Air-gap and verify backup catalogue integrity now",
                "Institute out-of-band verbal challenge codes for all IR comms",
            ],
            "threat_score": score,
            "confidence_score": min(93, score - 2),
            "next_step": "self_evaluate",
        }

    def _build_defense(self, text: str) -> dict[str, Any]:
        """Build a standard (non-escalated) defensive posture.

        Args:
            text: The task payload for the low-severity branch.

        Returns:
            A report dict matching the standard defence contract.
        """
        return {
            "analysis_result": "Sub-threshold incident; routine hardening is sufficient.",
            "counter_strategy": [
                "Log and monitor the flagged indicators for 72 hours",
                "Issue an awareness advisory to the affected business unit",
            ],
            "threat_score": self._severity(text, 15, 55),
            "confidence_score": 74,
            "next_step": "self_evaluate",
        }

    def _build_evaluator(self, text: str) -> dict[str, Any]:
        """Build a self-evaluation verdict on the assembled mission product.

        Args:
            text: The evaluation prompt containing the draft mission product.

        Returns:
            A verdict dict matching the evaluator's contract.
        """
        coverage = 70 + self._stable_jitter(text, 26)
        return {
            "analysis_result": "Draft product reviewed against the original operator brief.",
            "coverage_score": coverage,
            "unmet_requirements": [] if coverage >= 75 else ["attack_vector_attribution_thin"],
            "verdict": "ACCEPT" if coverage >= 75 else "REFINE",
            "confidence_score": coverage,
            "next_step": "finalise" if coverage >= 75 else "refine",
        }

    def _build_generic(self, text: str) -> dict[str, Any]:
        """Build a minimal contract-satisfying report for unrecognised roles.

        Args:
            text: The task payload.

        Returns:
            A generic report dict.
        """
        return {
            "analysis_result": "Generic assessment produced by the offline engine.",
            "threat_score": self._severity(text, 20, 70),
            "confidence_score": 60,
            "next_step": "continue",
        }


class ResilientLLMClient:
    """Structured-output LLM client with timeouts, retries and JSON repair.

    Attributes:
        settings: The active runtime configuration.
        call_log: Chronological record of every call, used by the report and the
            live demonstration.
    """

    def __init__(self, settings: Settings) -> None:
        """Create a client bound to a configuration.

        Args:
            settings: Runtime configuration, including provider and budgets.
        """
        self.settings = settings
        self.call_log: list[dict[str, Any]] = []
        # Failure-injection channels. These exist so the live demonstration and
        # the test suite can trigger each failure class on demand, without
        # unplugging a network cable in front of an audience.
        self.inject_unavailable: set[str] = set()
        self.inject_malformed: set[str] = set()
        self.inject_contract_breach: set[str] = set()
        self._mock = MockInferenceEngine()
        self._client: Any = None

        if settings.provider != "mock":
            try:
                from openai import OpenAI

                # Timeout and retries are set on the client itself so that *no*
                # call site can accidentally issue an unbounded request.
                self._client = OpenAI(
                    base_url=settings.base_url,
                    api_key=settings.api_key,
                    timeout=settings.request_timeout,
                    max_retries=0,  # retry loop is ours, so we can log each attempt
                )
            except ImportError as exc:  # pragma: no cover
                raise LLMUnavailableError(
                    "openai package is required for local/remote providers"
                ) from exc

    def _raw_call(self, system_prompt: str, user_prompt: str) -> str:
        """Issue one un-retried request to the configured backend.

        Args:
            system_prompt: The agent persona and output contract.
            user_prompt: The task payload.

        Returns:
            The model's raw text reply.

        Raises:
            LLMUnavailableError: If the transport fails or returns empty content.
        """
        if self.settings.provider == "mock":
            time.sleep(0.01)  # keeps latency numbers in the trace non-zero
            return self._mock.generate(system_prompt, user_prompt)

        # resolve() re-checks that we are not about to send an inference request
        # to an embedding model — the specific hazard called out in the brief.
        model_id = self.settings.model_registry.resolve("chat")
        try:
            response = self._client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=self.settings.request_timeout,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # TRY/EXCEPT #2 — transport failure (timeout, refused connection,
            # rate limit, 5xx). Normalised into one typed error so the retry loop
            # and the fallback handler have exactly one exception to reason about.
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMUnavailableError("Endpoint returned an empty completion body")
        return content

    def _injected_or_real(self, caller: str, system_prompt: str, user_prompt: str) -> str:
        """Return an injected failure for this caller, or a real model reply.

        Args:
            caller: Name of the calling agent.
            system_prompt: The agent persona and output contract.
            user_prompt: The task payload.

        Returns:
            The model's raw text reply, or deliberately corrupted text when a
            malformed/contract-breach injection is armed for this caller.

        Raises:
            LLMUnavailableError: When an endpoint-outage injection is armed.
        """
        if caller in self.inject_unavailable:
            raise LLMUnavailableError(
                f"[INJECTED] Simulated endpoint outage for '{caller}'"
            )
        if caller in self.inject_malformed:
            return "Certainly! Here is my analysis: threat looks HIGH {not: valid json,,,"
        if caller in self.inject_contract_breach:
            return json.dumps({"analysis_result": "[INJECTED] contract breach: keys withheld"})
        return self._raw_call(system_prompt, user_prompt)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: Sequence[str],
        caller: str = "unknown",
    ) -> CallResult:
        """Call the model and return a validated JSON object.

        Retries transport failures up to the configured ceiling with exponential
        back-off, repairs fenced or chatty JSON, then enforces the caller's
        output contract.

        Args:
            system_prompt: Agent persona, few-shot examples and format contract.
            user_prompt: The task payload for this turn.
            required_keys: Keys the caller guarantees to its downstream consumers.
            caller: Name of the calling agent, recorded in the call log.

        Returns:
            A :class:`CallResult` holding the validated payload and call metrics.

        Raises:
            LLMUnavailableError: Every attempt failed at the transport layer.
            LLMParseError: A reply arrived but no JSON could be recovered.
            LLMContractError: JSON arrived but a required key was missing.
        """
        started = time.monotonic()
        attempts = 0
        last_error: Exception | None = None
        # settings.max_retries counts *retries*, so total attempts is one more.
        total_attempts = self.settings.max_retries + 1

        while attempts < total_attempts:
            attempts += 1
            try:
                raw = self._injected_or_real(caller, system_prompt, user_prompt)
            except LLMUnavailableError as exc:
                last_error = exc
                if attempts < total_attempts:
                    # [HUMAN-REVIEW] The AI proposed a flat 1-second sleep. We
                    # changed it to capped exponential back-off (0.4s, 0.8s,
                    # 1.6s) because a flat delay against a rate-limited endpoint
                    # simply burns the retry budget without letting the limiter
                    # recover, and the cap keeps us inside the 30 s ceiling.
                    time.sleep(min(0.4 * (2 ** (attempts - 1)), 4.0))
                    continue
                break

            try:
                payload = extract_json(raw)
            except LLMParseError as exc:
                last_error = exc
                if attempts < total_attempts:
                    # A parse failure is retried once with a blunt corrective
                    # suffix; small models usually comply on the second pass.
                    user_prompt += (
                        "\n\nCRITICAL: Your previous reply was not valid JSON. "
                        "Reply with a single raw JSON object and nothing else."
                    )
                    continue
                self._log(caller, attempts, started, "parse_failed")
                raise

            missing = [k for k in required_keys if k not in payload]
            if missing:
                # TRY/EXCEPT #3's counterpart: a contract breach is *detected*
                # here and raised as a typed error that agents catch below.
                self._log(caller, attempts, started, f"contract_missing:{','.join(missing)}")
                raise LLMContractError(
                    f"{caller}: response missing required key(s) {missing}; got {list(payload)}"
                )

            repaired = raw.strip() != json.dumps(payload)
            self._log(caller, attempts, started, "ok")
            return CallResult(
                payload=payload,
                attempts=attempts,
                latency_ms=(time.monotonic() - started) * 1000.0,
                repaired=repaired,
                provider=self.settings.provider,
            )

        self._log(caller, attempts, started, "unavailable")
        raise LLMUnavailableError(
            f"{caller}: all {attempts} attempt(s) failed. Last error: {last_error}"
        )

    def _log(self, caller: str, attempts: int, started: float, outcome: str) -> None:
        """Append one entry to the call log.

        Args:
            caller: Name of the calling agent.
            attempts: Number of attempts consumed.
            started: ``time.monotonic()`` value captured before the first attempt.
            outcome: Short status token, e.g. ``"ok"`` or ``"unavailable"``.
        """
        self.call_log.append(
            {
                "caller": caller,
                "attempts": attempts,
                "outcome": outcome,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                "provider": self.settings.provider,
            }
        )

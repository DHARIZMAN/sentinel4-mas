"""Central configuration for the SENTINEL-4 multi-agent countermeasure unit.

This module is the single source of truth for *where* inference requests go and
*which* model identifier is attached to them. The assessment brief explicitly
warns against misrouting inference calls to embedding models (for example
``text-embedding-nomic-embed-text-v1.5``); the :class:`ModelRegistry` below makes
that class of mistake impossible by validating every model name before it is
allowed onto the wire.

Typical use::

    from src.config import load_settings
    settings = load_settings()
    client = settings.model_registry.resolve("chat")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

try:  # pragma: no cover - dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    # [HUMAN-REVIEW] The AI draft made python-dotenv a hard import and crashed the
    # whole system when a grader ran the project without installing it. We
    # downgraded it to a soft dependency: real OS environment variables still
    # work, so the MAS starts in every environment we tested.
    pass

Provider = Literal["mock", "local", "remote"]

#: Substrings that identify an *embedding* model. Any model id containing one of
#: these is refused for chat/inference work. Keeping this as data (not scattered
#: ``if`` statements) means a new offending model only needs one list entry.
EMBEDDING_NAME_MARKERS: tuple[str, ...] = (
    "embed",
    "embedding",
    "nomic-embed",
    "text-embedding",
    "bge-",
    "gte-",
    "e5-",
)


class ModelRoutingError(ValueError):
    """Raised when a model identifier is used for the wrong kind of API call.

    This is a *configuration* error rather than a runtime failure: it means the
    operator wired an embedding model into a chat slot (or vice versa), and the
    system refuses to start rather than silently returning vectors where prose
    was expected.
    """


@dataclass(frozen=True)
class ModelRegistry:
    """Maps logical roles ("chat", "embed") onto concrete model identifiers.

    Attributes:
        chat_model: Identifier of the instruct/chat model used for all agent
            inference calls.
        embed_model: Identifier of the embedding model, reserved for similarity
            work. Never used for generation.

    Raises:
        ModelRoutingError: If ``chat_model`` looks like an embedding model.
    """

    chat_model: str
    embed_model: str

    def __post_init__(self) -> None:
        """Validate the chat slot at construction time.

        Fail-fast beats fail-weird: an embedding id in the chat slot produces a
        confusing 400 from the endpoint several layers deeper in the call stack.
        """
        if self.is_embedding_name(self.chat_model):
            raise ModelRoutingError(
                f"Refusing to start: '{self.chat_model}' is configured as the chat "
                "model but its name identifies it as an embedding model. Set "
                "MAS_*_CHAT_MODEL to an instruct/chat model id."
            )

    @staticmethod
    def is_embedding_name(model_name: str) -> bool:
        """Report whether a model identifier denotes an embedding model.

        Args:
            model_name: The raw model identifier, e.g. ``"qwen2.5-7b-instruct"``.

        Returns:
            ``True`` when the identifier contains a known embedding marker.
        """
        lowered = model_name.lower()
        return any(marker in lowered for marker in EMBEDDING_NAME_MARKERS)

    def resolve(self, purpose: Literal["chat", "embed"]) -> str:
        """Return the model id registered for a given purpose.

        Args:
            purpose: Either ``"chat"`` (generation) or ``"embed"`` (vectors).

        Returns:
            The validated model identifier for that purpose.

        Raises:
            ModelRoutingError: If ``purpose`` is not a known slot.
        """
        if purpose == "chat":
            return self.chat_model
        if purpose == "embed":
            return self.embed_model
        raise ModelRoutingError(f"Unknown model purpose: {purpose!r}")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for one execution of the MAS.

    Attributes:
        provider: Which backend to use — ``mock``, ``local`` or ``remote``.
        base_url: OpenAI-compatible endpoint URL (ignored when provider is mock).
        api_key: Credential for the endpoint (ignored when provider is mock).
        request_timeout: Per-request wall-clock ceiling in seconds. The brief
            caps this at 30 s.
        max_retries: Retry attempts after the first failure. The brief caps this
            at 3.
        max_refinement_loops: Hard ceiling on self-evaluation re-runs; the loop
            guard that stops an agent cycling forever.
        escalation_threshold: Threat score above which the Strategic Predictor
            branch is taken instead of the Standard Defense branch.
        model_registry: Validated chat/embedding model mapping.
    """

    provider: Provider
    base_url: str
    api_key: str
    request_timeout: float
    max_retries: int
    max_refinement_loops: int
    escalation_threshold: float
    model_registry: ModelRegistry = field(repr=False)

    def describe(self) -> str:
        """Return a one-line, credential-free summary for logs and demos.

        Returns:
            A human-readable configuration summary safe to print on screen
            during the live demonstration (the API key is never included).
        """
        return (
            f"provider={self.provider} model={self.model_registry.chat_model} "
            f"timeout={self.request_timeout}s retries={self.max_retries} "
            f"escalate_above={self.escalation_threshold}"
        )


def _env_float(name: str, default: float, maximum: float | None = None) -> float:
    """Read a float environment variable, clamping it to an allowed maximum.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or unparseable.
        maximum: Optional inclusive upper bound to clamp to.

    Returns:
        The parsed (and possibly clamped) float value.
    """
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        # [HUMAN-REVIEW] The AI original let a malformed value such as
        # MAS_REQUEST_TIMEOUT="thirty" raise and kill startup. Because a bad env
        # var is an operator typo rather than a system fault, we now fall back to
        # the documented default instead of refusing to run.
        value = default
    if maximum is not None:
        value = min(value, maximum)
    return value


def _env_int(name: str, default: int, maximum: int | None = None) -> int:
    """Read an integer environment variable, clamping it to an allowed maximum.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or unparseable.
        maximum: Optional inclusive upper bound to clamp to.

    Returns:
        The parsed (and possibly clamped) integer value.
    """
    return int(_env_float(name, float(default), float(maximum) if maximum else None))


def load_settings() -> Settings:
    """Build a :class:`Settings` object from the process environment.

    Reads ``MAS_*`` variables (populated from ``.env`` when python-dotenv is
    installed) and applies the assessment's hard ceilings: request timeout is
    clamped to 30 seconds and retries to 3, so no misconfiguration can push the
    system outside the specified resilience budget.

    Returns:
        A fully validated, immutable settings object.

    Raises:
        ModelRoutingError: If the configured chat model is an embedding model.
    """
    provider: Provider = os.getenv("MAS_PROVIDER", "mock").strip().lower()  # type: ignore[assignment]
    if provider not in ("mock", "local", "remote"):
        provider = "mock"

    if provider == "local":
        base_url = os.getenv("MAS_LOCAL_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("MAS_LOCAL_API_KEY", "lm-studio")
        chat_model = os.getenv("MAS_LOCAL_CHAT_MODEL", "qwen2.5-7b-instruct")
        embed_model = os.getenv("MAS_LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
    elif provider == "remote":
        base_url = os.getenv("MAS_REMOTE_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("MAS_REMOTE_API_KEY", "")
        chat_model = os.getenv("MAS_REMOTE_CHAT_MODEL", "gpt-4o-mini")
        embed_model = os.getenv("MAS_REMOTE_EMBED_MODEL", "text-embedding-3-small")
    else:
        base_url, api_key = "mock://offline", "not-required"
        chat_model, embed_model = "sentinel-mock-instruct", "sentinel-mock-embed"

    return Settings(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        # Ceilings below are the brief's limits, enforced in code rather than in prose.
        request_timeout=_env_float("MAS_REQUEST_TIMEOUT", 25.0, maximum=30.0),
        max_retries=_env_int("MAS_MAX_RETRIES", 2, maximum=3),
        max_refinement_loops=_env_int("MAS_MAX_REFINEMENT_LOOPS", 2, maximum=3),
        escalation_threshold=_env_float("MAS_ESCALATION_THRESHOLD", 80.0),
        model_registry=ModelRegistry(chat_model=chat_model, embed_model=embed_model),
    )

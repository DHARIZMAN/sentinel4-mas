"""Common machinery shared by every SENTINEL-4 specialist agent.

Each concrete agent supplies three things — a persona, an output contract, and
an evidence-gathering routine — and inherits everything else: prompt assembly,
resilient invocation, tool-error absorption and blackboard patching.

The invariant enforced here is that **an agent never raises into the graph**. A
specialist that fails returns a degraded report carrying ``"status":
"DEGRADED"`` plus a structured error record. The graph therefore always has
something to work with, which is precisely what makes the partial-response
fallback possible.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

from src.llm_client import (
    LLMContractError,
    LLMError,
    LLMParseError,
    LLMUnavailableError,
    ResilientLLMClient,
)
from src.state import MissionState, trace_event
from src.tools.registry import HallucinatedToolError, ToolExecutionError, ToolRegistry

#: Appended to every agent persona. Centralising the format contract guarantees
#: that all agents are held to the same strict-JSON discipline.
FORMAT_CONTRACT_TEMPLATE = """
OUTPUT FORMAT — NON-NEGOTIABLE
You must reply with a single raw JSON object and nothing else.
Do not wrap it in markdown fences. Do not add commentary before or after it.
The object must contain exactly these keys: {keys}.
Numeric fields must be plain integers between 0 and 100 with no units or symbols.
List fields must be JSON arrays of short strings.
If evidence is insufficient for a field, still emit the key and state the
limitation inside the string value — never omit a key and never return null.
"""


class SpecialistAgent(ABC):
    """Abstract base for the four domain specialists and the evaluator.

    Attributes:
        name: Blackboard key under which this agent files its report.
        role_tag: Uppercase tag embedded in the system prompt; the offline mock
            engine reads it to choose a report shape.
        display_name: Human-readable name used in traces and the final report.
        required_keys: Keys this agent contracts to return.
    """

    name: str = "unnamed_agent"
    role_tag: str = "GENERIC"
    display_name: str = "Unnamed Specialist"
    required_keys: Sequence[str] = ("analysis_result", "threat_score", "confidence_score")

    @abstractmethod
    def persona(self, registry: ToolRegistry) -> str:
        """Return the agent's system prompt body, excluding the format contract.

        Args:
            registry: Tool registry, so the persona can advertise its own tools.

        Returns:
            The persona, mission scope, boundaries and few-shot examples.
        """

    @abstractmethod
    def gather_evidence(self, state: MissionState, registry: ToolRegistry) -> dict[str, Any]:
        """Invoke this agent's tools and return their combined findings.

        Args:
            state: The current blackboard.
            registry: The tool registry to invoke through.

        Returns:
            A dictionary of tool results, or ``{"tool_status": ...}`` describing
            why tools were unavailable.
        """

    def system_prompt(self, registry: ToolRegistry) -> str:
        """Assemble the complete system prompt for this agent.

        Args:
            registry: Tool registry used to render the tool catalogue.

        Returns:
            Persona plus the shared strict-JSON format contract.
        """
        contract = FORMAT_CONTRACT_TEMPLATE.format(keys=list(self.required_keys))
        return f"ROLE-TAG: {self.role_tag}\n{self.persona(registry)}\n{contract}"

    def build_user_prompt(self, state: MissionState, evidence: dict[str, Any]) -> str:
        """Assemble the task message for this turn.

        Args:
            state: The current blackboard.
            evidence: Output of :meth:`gather_evidence`.

        Returns:
            The user-role prompt text.
        """
        return (
            f"INCIDENT BRIEF\n{state.get('raw_input', '(no brief supplied)')}\n\n"
            f"TOOL EVIDENCE\n{evidence}\n\n"
            "Produce your report now."
        )

    def safe_tool(self, registry: ToolRegistry, tool: str, **kwargs: Any) -> dict[str, Any]:
        """Invoke a tool, converting any failure into a structured result.

        Args:
            registry: The tool registry.
            tool: Name of the tool to invoke.
            **kwargs: Arguments forwarded to the tool.

        Returns:
            The tool's result, or a dict with ``tool_error`` describing the
            failure. Never raises.
        """
        try:
            return registry.invoke(tool, **kwargs)
        except HallucinatedToolError as exc:
            # TRY/EXCEPT #3 — hallucinated tool. Downgraded to evidence-quality
            # loss rather than a crash, exactly as the brief's risk-mitigation
            # clause requires.
            return {"tool_error": "hallucinated_tool", "detail": str(exc)}
        except ToolExecutionError as exc:
            return {"tool_error": "execution_failed", "detail": str(exc)}

    def degraded_report(self, reason: str, detail: str) -> dict[str, Any]:
        """Build the report this agent files when it cannot complete normally.

        Args:
            reason: Short machine-readable failure class.
            detail: Human-readable explanation for the operator.

        Returns:
            A contract-shaped report flagged ``status: DEGRADED`` with a
            deliberately low confidence so the fusion step down-weights it.
        """
        report: dict[str, Any] = {
            "analysis_result": f"[DEGRADED] {self.display_name} could not complete: {detail}",
            "threat_score": 0,
            "confidence_score": 0,
            "status": "DEGRADED",
            "failure_reason": reason,
        }
        # Fill any contract keys this agent promised but the degraded path lacks,
        # so consumers can index the report without defensive checks everywhere.
        for key in self.required_keys:
            report.setdefault(key, "unavailable_due_to_degradation")
        return report

    def run(
        self,
        state: MissionState,
        client: ResilientLLMClient,
        registry: ToolRegistry,
    ) -> dict[str, Any]:
        """Execute the agent and return a blackboard patch.

        Args:
            state: The current blackboard.
            client: The resilient LLM transport.
            registry: The tool registry.

        Returns:
            A partial :class:`MissionState` update containing this agent's
            report, its trace entry, and any error or warning records. This
            method does not raise.
        """
        started = time.monotonic()
        evidence = self.gather_evidence(state, registry)
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if any(isinstance(v, dict) and "tool_error" in v for v in evidence.values()):
            warnings.append({
                "agent": self.name,
                "type": "tool_degradation",
                "detail": "One or more tools failed; analysis proceeds on partial evidence.",
            })

        try:
            result = client.complete_json(
                system_prompt=self.system_prompt(registry),
                user_prompt=self.build_user_prompt(state, evidence),
                required_keys=self.required_keys,
                caller=self.name,
            )
            report = dict(result.payload)
            report["status"] = "OK"
            report["_meta"] = {
                "attempts": result.attempts,
                "latency_ms": round(result.latency_ms, 1),
                "json_repaired": result.repaired,
            }
        except (LLMUnavailableError, LLMParseError, LLMContractError, LLMError) as exc:
            # [HUMAN-REVIEW] The AI generated a bare `except Exception` here. We
            # narrowed it to the transport's own error hierarchy so that genuine
            # programming bugs (TypeError, AttributeError) still surface loudly in
            # development instead of being silently laundered into a degraded
            # report that looks like a normal API outage.
            reason = type(exc).__name__
            report = self.degraded_report(reason, str(exc)[:200])
            errors.append({
                "agent": self.name,
                "type": reason,
                "detail": str(exc)[:300],
                "fatal": False,  # one specialist failing is survivable; the fusion step decides
            })

        report["tool_evidence"] = evidence
        duration_ms = round((time.monotonic() - started) * 1000.0, 1)

        return {
            "agent_outputs": {self.name: report},
            "activated_agents": [self.name],
            "trace": [trace_event(
                self.name,
                f"{self.display_name} filed report ({report.get('status')})",
                duration_ms=duration_ms,
                threat_score=report.get("threat_score"),
            )],
            "errors": errors,
            "warnings": warnings,
        }

"""LangGraph workflow assembly for the SENTINEL-4 MAS.

COLLABORATION PATTERN — declared for the assessment
---------------------------------------------------
This is a **hierarchical supervisor over a shared-state blackboard**, not a
linear pipeline. A dynamic supervisor (:class:`~src.router.MissionRouter`) fans
work out to media specialists that run *concurrently*; their reports land on one
shared :class:`~src.state.MissionState`; a fusion agent reads the whole board and
verifies an attack vector; and only then does a conditional gate decide whether
predictive planning is warranted.

The pattern was chosen because the brief's dependency requirement — "the
Strategic Predictor cannot formulate a plan until the Cyber Coordinator verifies
the attack vector" — is a *data* dependency, not merely an ordering one. A
blackboard expresses it directly: the predictor reads the coordinator's
``vector_verified`` flag off shared state and refuses to plan without it.

CONDITIONAL ROUTING PATHS (four, against a required minimum of two)
-------------------------------------------------------------------
1. ``dispatch_gate``    — router intent decides *which* specialists activate.
2. ``health_gate``      — fatal error or mission-deadline breach diverts to fallback.
3. ``escalation_gate``  — IF fused threat_score >= 80 THEN Strategic Predictor
                          ELSE Standard Defense.
4. ``evaluation_gate``  — REFINE (with budget) loops back to fusion; ACCEPT
                          finalises; failure diverts to fallback.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.audio_analyst import AudioAnalystAgent
from src.agents.cyber_coordinator import CyberCoordinatorAgent
from src.agents.standard_defense import StandardDefenseAgent
from src.agents.strategic_predictor import StrategicPredictorAgent
from src.agents.video_detector import VideoDetectorAgent
from src.config import Settings, load_settings
from src.evaluation import SelfEvaluator, fusion_node
from src.fallback import fallback_node
from src.llm_client import ResilientLLMClient
from src.router import MissionRouter
from src.state import MissionState, elapsed_ms, new_mission_state, trace_event
from src.tools.registry import ToolRegistry, build_default_registry

#: Hard ceiling on graph super-steps. LangGraph raises GraphRecursionError past
#: this, which the orchestrator converts into a fallback rather than a crash.
RECURSION_LIMIT = 24

#: Wall-clock budget for a whole mission. Exceeding it diverts to fallback so a
#: slow endpoint degrades the answer instead of hanging the operator.
MISSION_DEADLINE_SECONDS = 90.0


class SentinelWorkflow:
    """Builds and runs the compiled countermeasure graph.

    Attributes:
        settings: Runtime configuration.
        client: Resilient LLM transport shared by every node.
        registry: Tool registry shared by every agent.
        graph: The compiled LangGraph application.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: ResilientLLMClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        """Assemble the workflow.

        Args:
            settings: Runtime configuration; loaded from the environment when
                omitted.
            client: Pre-built LLM client, chiefly so tests and the failure demo
                can arm injection channels before the run.
            registry: Pre-built tool registry, for the same reason.
        """
        self.settings = settings or load_settings()
        self.client = client or ResilientLLMClient(self.settings)
        self.registry = registry or build_default_registry()

        self.router = MissionRouter(self.client, self.registry)
        self.evaluator = SelfEvaluator(self.client, self.settings)
        self.specialists = {
            "audio_analyst": AudioAnalystAgent(),
            "video_detector": VideoDetectorAgent(),
            "cyber_coordinator": CyberCoordinatorAgent(),
            "strategic_predictor": StrategicPredictorAgent(),
            "standard_defense": StandardDefenseAgent(),
        }
        self.graph = self._build()

    # -- nodes ---------------------------------------------------------------

    def _ingest(self, state: MissionState) -> dict[str, Any]:
        """Record mission start and normalise the operator brief.

        Args:
            state: The incoming blackboard.

        Returns:
            A patch containing the ingestion trace entry.
        """
        return {"trace": [trace_event(
            "ingest",
            f"Mission {state.get('run_id')} accepted "
            f"({len(state.get('raw_input', ''))} chars, "
            f"{len(state.get('evidence_refs', []))} artefact(s))",
            config=self.settings.describe(),
        )]}

    def _run_specialist(self, name: str):
        """Create a graph-node callable for one specialist.

        Args:
            name: Key of the specialist in ``self.specialists``.

        Returns:
            A callable of the shape LangGraph expects, ``(state) -> patch``.
        """
        def node(state: MissionState) -> dict[str, Any]:
            """Execute the bound specialist against the current blackboard.

            Args:
                state: The current blackboard.

            Returns:
                The specialist's blackboard patch.
            """
            return self.specialists[name].run(state, self.client, self.registry)

        node.__name__ = f"node_{name}"
        return node

    def _evaluate(self, state: MissionState) -> dict[str, Any]:
        """Run the adversarial self-evaluation module.

        Args:
            state: The current blackboard.

        Returns:
            The evaluator's blackboard patch.
        """
        return self.evaluator.evaluate(state)

    def _synthesise(self, state: MissionState) -> dict[str, Any]:
        """Assemble the final, non-degraded mission product.

        Args:
            state: The completed blackboard.

        Returns:
            A patch containing ``final_report`` and a closing trace entry.
        """
        outputs = state.get("agent_outputs", {})
        strategy_source = outputs.get("strategic_predictor") or outputs.get("standard_defense", {})
        coordinator = outputs.get("cyber_coordinator", {})

        report = {
            "product_type": "FULL_ASSESSMENT",
            "run_id": state.get("run_id"),
            "threat_score": state.get("threat_score"),
            "threat_band": state.get("threat_band"),
            "escalation_path": state.get("escalation_path"),
            "routing": {
                "mode": state.get("route_decision", {}).get("mode"),
                "intent": state.get("route_decision", {}).get("intent"),
                "agents_dispatched": state.get("route_decision", {}).get("selected_agents"),
            },
            "attack_vector": coordinator.get("attack_vector", "not_established"),
            "vector_verified": coordinator.get("vector_verified", False),
            "containment_actions": coordinator.get("containment_actions", []),
            "predicted_next_moves": strategy_source.get("predicted_next_moves", []),
            "counter_strategy": strategy_source.get("counter_strategy", []),
            "specialist_reports": {
                name: {k: v for k, v in rep.items() if k not in ("tool_evidence", "_meta")}
                for name, rep in outputs.items()
            },
            "self_evaluation": state.get("evaluation", {}),
            "warnings": state.get("warnings", []),
            "errors": state.get("errors", []),
            "elapsed_ms": round(elapsed_ms(state), 1),
        }
        return {
            "final_report": report,
            "trace": [trace_event("synthesise", "Full assessment released")],
        }

    # -- conditional edges ---------------------------------------------------

    def _dispatch_gate(self, state: MissionState) -> list[str]:
        """CONDITIONAL PATH 1 — fan out to the specialists the router selected.

        Args:
            state: The blackboard, read for ``route_decision``.

        Returns:
            A list of node names to run concurrently in the next super-step.
        """
        selected = state.get("route_decision", {}).get("selected_agents") or ["cyber_coordinator"]
        media = [a for a in selected if a in ("audio_analyst", "video_detector")]
        # If no media specialist applies, jump straight to fusion. Returning the
        # coordinator here instead would double-run it, since every media branch
        # already terminates at the coordinator.
        return media if media else ["cyber_coordinator"]

    def _health_gate(self, state: MissionState) -> str:
        """CONDITIONAL PATH 2 — divert to fallback on a fatal fault or timeout.

        Args:
            state: The current blackboard.

        Returns:
            ``"fallback"`` or ``"threat_fusion"``.
        """
        if any(error.get("fatal") for error in state.get("errors", [])):
            return "fallback"
        if elapsed_ms(state) > MISSION_DEADLINE_SECONDS * 1000.0:
            return "fallback"

        coordinator = state.get("agent_outputs", {}).get("cyber_coordinator", {})
        if coordinator.get("status") == "DEGRADED":
            # The fusion point failed. Every downstream stage reads its output,
            # so continuing would produce a confident-looking empty assessment —
            # strictly worse than an honest partial one.
            return "fallback"
        return "threat_fusion"

    def _escalation_gate(self, state: MissionState) -> str:
        """CONDITIONAL PATH 3 — the threat-score escalation branch.

        Implements the brief's worked example directly: IF fused threat score
        >= the configured threshold (default 80) THEN Strategic Predictor, ELSE
        Standard Defense.

        Args:
            state: The blackboard, read for ``threat_score``.

        Returns:
            ``"strategic_predictor"`` or ``"standard_defense"``.
        """
        score = float(state.get("threat_score", 0.0))
        return ("strategic_predictor" if score >= self.settings.escalation_threshold
                else "standard_defense")

    def _record_escalation(self, state: MissionState) -> dict[str, Any]:
        """Write the chosen escalation branch onto the blackboard.

        Kept as its own node so the decision is *recorded* as state, not only
        implied by which node ran — the report and the demo both read it back.

        Args:
            state: The current blackboard.

        Returns:
            A patch containing ``escalation_path`` and a trace entry.
        """
        branch = self._escalation_gate(state)
        label = ("ESCALATED -> predictive planning" if branch == "strategic_predictor"
                 else "STANDARD -> routine hardening")
        return {
            "escalation_path": branch,
            "trace": [trace_event(
                "escalation_gate",
                f"score {state.get('threat_score')} vs threshold "
                f"{self.settings.escalation_threshold}: {label}",
            )],
        }

    def _evaluation_gate(self, state: MissionState) -> str:
        """CONDITIONAL PATH 4 — refine, finalise, or fall back.

        Args:
            state: The blackboard, read for ``evaluation`` and budgets.

        Returns:
            ``"threat_fusion"`` (refine loop), ``"synthesise"``, or ``"fallback"``.
        """
        if any(error.get("fatal") for error in state.get("errors", [])):
            return "fallback"

        verdict = str(state.get("evaluation", {}).get("verdict", "ACCEPT")).upper()
        loops_used = state.get("refinement_count", 0)

        if verdict == "REFINE" and loops_used <= self.settings.max_refinement_loops:
            if elapsed_ms(state) > MISSION_DEADLINE_SECONDS * 1000.0:
                # [HUMAN-REVIEW] The AI's gate checked only the loop counter. We
                # added the deadline check because a slow endpoint can burn the
                # entire wall-clock budget inside a *legal* refinement loop —
                # the counter says "keep going" while the operator is left
                # waiting. Time and iterations are separate exhaustion modes and
                # both have to be able to stop the loop.
                return "synthesise"
            return "threat_fusion"
        return "synthesise"

    # -- assembly ------------------------------------------------------------

    def _build(self):
        """Wire every node and edge, then compile the graph.

        Returns:
            The compiled LangGraph application.
        """
        builder = StateGraph(MissionState)

        builder.add_node("ingest", self._ingest)
        builder.add_node("router", lambda s: self.router.route(s))
        for name in self.specialists:
            builder.add_node(name, self._run_specialist(name))
        builder.add_node("threat_fusion", fusion_node)
        builder.add_node("escalation_gate", self._record_escalation)
        builder.add_node("self_evaluation", self._evaluate)
        builder.add_node("synthesise", self._synthesise)
        builder.add_node("fallback", fallback_node)

        builder.add_edge(START, "ingest")
        builder.add_edge("ingest", "router")

        # PATH 1: dynamic fan-out to the selected media specialists.
        builder.add_conditional_edges(
            "router", self._dispatch_gate,
            ["audio_analyst", "video_detector", "cyber_coordinator"],
        )
        # Media specialists converge on the coordinator. LangGraph's super-step
        # semantics make the coordinator wait for every branch that actually
        # started, which is what enforces the fusion dependency.
        builder.add_edge("audio_analyst", "cyber_coordinator")
        builder.add_edge("video_detector", "cyber_coordinator")

        # PATH 2: health check after fusion point.
        builder.add_conditional_edges(
            "cyber_coordinator", self._health_gate, ["threat_fusion", "fallback"],
        )
        builder.add_edge("threat_fusion", "escalation_gate")

        # PATH 3: threat-score escalation branch.
        builder.add_conditional_edges(
            "escalation_gate", self._escalation_gate,
            ["strategic_predictor", "standard_defense"],
        )
        builder.add_edge("strategic_predictor", "self_evaluation")
        builder.add_edge("standard_defense", "self_evaluation")

        # PATH 4: refine / finalise / fall back.
        builder.add_conditional_edges(
            "self_evaluation", self._evaluation_gate,
            ["threat_fusion", "synthesise", "fallback"],
        )
        builder.add_edge("synthesise", END)
        builder.add_edge("fallback", END)

        return builder.compile()

    # -- execution -----------------------------------------------------------

    def run(self, brief: str, evidence_refs: list[str] | None = None) -> MissionState:
        """Execute one mission end to end.

        Wraps graph invocation so that *no* exception escapes: a recursion-limit
        breach or an unforeseen error is converted into the degraded partial
        product, which is the behaviour the brief's risk-mitigation clause
        requires.

        Args:
            brief: The operator's incident description.
            evidence_refs: Optional artefact filenames or URIs.

        Returns:
            The final blackboard, always carrying a populated ``final_report``.
        """
        initial = new_mission_state(brief, evidence_refs)
        # [HUMAN-REVIEW] The AI used graph.invoke() here. We switched to stream()
        # and kept the newest emitted state, because invoke() discards everything
        # the graph produced when it raises — so a loop breach at step 23 threw
        # away 22 steps of good analysis and the fallback had nothing to salvage.
        # Streaming makes the partial response genuinely partial rather than empty.
        last_state: MissionState = initial
        try:
            for emitted in self.graph.stream(
                initial, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values"
            ):
                last_state = emitted  # type: ignore[assignment]
            return last_state
        except Exception as exc:
            # TRY/EXCEPT #5 — the outermost safety net. GraphRecursionError (a
            # genuine infinite loop) and anything else unforeseen land here and
            # are converted into a partial response instead of a stack trace.
            salvage = dict(last_state)
            salvage.setdefault("errors", [])
            salvage["errors"] = list(salvage["errors"]) + [{
                "agent": "graph",
                "type": type(exc).__name__,
                "detail": str(exc)[:300],
                "fatal": True,
            }]
            patch = fallback_node(salvage)  # type: ignore[arg-type]
            salvage.update(patch)
            salvage["trace"] = list(salvage.get("trace", [])) + list(patch.get("trace", []))
            return salvage  # type: ignore[return-value]


def run_mission(brief: str, evidence_refs: list[str] | None = None) -> MissionState:
    """Convenience helper: build a default workflow and run one mission.

    Args:
        brief: The operator's incident description.
        evidence_refs: Optional artefact filenames or URIs.

    Returns:
        The final blackboard for the mission.
    """
    return SentinelWorkflow().run(brief, evidence_refs)

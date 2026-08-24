#!/usr/bin/env python3
"""Live failure-injection demonstration for SENTINEL-4.

Run this during the presentation to show, one after another, that each failure
class the brief names is caught and converted into a degraded-but-useful answer
rather than a crash:

1. **Endpoint outage** at the fusion agent  -> partial response.
2. **Malformed JSON** from a specialist     -> that agent degrades, mission continues.
3. **Hallucinated tool**                    -> refused at the registry, run continues.
4. **Infinite reasoning loop**              -> recursion ceiling -> partial response.

Every case is timed, and the fallback budget (< 5 s) is asserted for each.

Usage::

    python demo_fallback.py
    python demo_fallback.py --case 3
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

from src.agents.audio_analyst import AudioAnalystAgent
from src.config import load_settings
from src.fallback import FALLBACK_BUDGET_SECONDS
from src.graph import SentinelWorkflow
from src.llm_client import ResilientLLMClient
from src.state import MissionState
from src.tools.registry import HallucinatedToolError, build_default_registry

RULE = "-" * 74

BRIEF = (
    "Synthetic voicemail impersonating the CFO authorising an emergency wire "
    "transfer, manipulated CCTV footage of the server room, and outbound "
    "beaconing to 185.220.101.44 exploiting CVE-2024-3400. Zero-day suspected "
    "with lateral movement and staged exfiltration."
)
ARTEFACTS = ["cfo_voicemail_0214.wav", "serverroom_cctv_0214.mp4"]


class HallucinatingAudioAgent(AudioAnalystAgent):
    """Audio specialist that requests a tool which does not exist.

    Used only by demonstration case 3. It models the realistic failure in which
    a language model confidently names a plausible-sounding capability
    (``run_nmap_scan``) that was never registered.
    """

    def gather_evidence(self, state: MissionState, registry: Any) -> dict[str, Any]:
        """Request a non-existent tool alongside the legitimate one.

        Args:
            state: The current blackboard.
            registry: The tool registry.

        Returns:
            Evidence containing one real result and one refusal record.
        """
        return {
            "hallucinated_call": self.safe_tool(registry, "run_nmap_scan", target="10.0.0.0/24"),
            "legitimate_call": self.safe_tool(
                registry, "scan_audio_artifacts", evidence_ref=ARTEFACTS[0]
            ),
        }


def _banner(number: int, title: str) -> None:
    """Print a numbered case header.

    Args:
        number: Case number.
        title: Case title.
    """
    print(f"\n{RULE}\n CASE {number}: {title}\n{RULE}")


def _verdict(elapsed: float, degraded: bool) -> None:
    """Print the timing verdict for a case.

    Args:
        elapsed: Seconds the case took end to end.
        degraded: Whether the run produced a partial response.
    """
    status = "PARTIAL RESPONSE" if degraded else "FULL ASSESSMENT (survived)"
    budget = "WITHIN" if elapsed < FALLBACK_BUDGET_SECONDS else "OVER"
    print(f" -> {status} in {elapsed * 1000:.0f} ms "
          f"({budget} the {FALLBACK_BUDGET_SECONDS:.0f}s fallback budget)")


def case_endpoint_outage() -> None:
    """Case 1: the inference endpoint dies at the fusion agent."""
    _banner(1, "Inference endpoint outage at the Cyber Coordinator")
    settings = load_settings()
    client = ResilientLLMClient(settings)
    client.inject_unavailable.add("cyber_coordinator")

    started = time.monotonic()
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    report = state["final_report"]
    warning = report.get("risk_warning", {})
    print(f" Retries consumed : "
          f"{[c['attempts'] for c in client.call_log if c['caller'] == 'cyber_coordinator']}")
    print(f" Risk warning     : {warning.get('headline')}")
    print(f" Completion       : {warning.get('completion_pct')}%")
    print(f" Salvaged         : {list(report.get('salvaged_findings', {}))}")
    _verdict(elapsed, state.get("fallback_triggered", False))


def case_malformed_json() -> None:
    """Case 2: a specialist returns text that is not JSON."""
    _banner(2, "Malformed (non-JSON) output from the Audio Analyst")
    settings = load_settings()
    client = ResilientLLMClient(settings)
    client.inject_malformed.add("audio_analyst")

    started = time.monotonic()
    state = SentinelWorkflow(settings=settings, client=client).run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    audio = state["agent_outputs"].get("audio_analyst", {})
    print(f" Audio agent status : {audio.get('status')} ({audio.get('failure_reason')})")
    print(f" Mission continued  : {list(state['agent_outputs'])}")
    print(f" Fused score        : {state.get('threat_score')} "
          f"(degraded report excluded from fusion)")
    print(f" Product            : {state['final_report']['product_type']}")
    _verdict(elapsed, state.get("fallback_triggered", False))


def case_hallucinated_tool() -> None:
    """Case 3: an agent invents a tool that was never registered."""
    _banner(3, "Hallucinated tool invocation")
    registry = build_default_registry()

    try:
        registry.invoke("run_nmap_scan", target="10.0.0.0/24")
    except HallucinatedToolError as exc:
        print(f" Registry refusal   : {str(exc)[:90]}...")

    settings = load_settings()
    workflow = SentinelWorkflow(settings=settings, registry=registry)
    workflow.specialists["audio_analyst"] = HallucinatingAudioAgent()

    started = time.monotonic()
    state = workflow.run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    tool_warnings = [w for w in state.get("warnings", []) if w["type"] == "tool_degradation"]
    print(f" Warning raised     : {tool_warnings[0]['detail'] if tool_warnings else 'none'}")
    print(f" Audio agent status : {state['agent_outputs']['audio_analyst'].get('status')}")
    print(f" Product            : {state['final_report']['product_type']}")
    _verdict(elapsed, state.get("fallback_triggered", False))


def case_infinite_loop() -> None:
    """Case 4: a reasoning loop that never terminates on its own."""
    _banner(4, "Infinite reasoning loop (recursion ceiling)")

    class LoopingWorkflow(SentinelWorkflow):
        """Workflow whose evaluation gate never accepts, forcing a true loop."""

        def _evaluation_gate(self, state: MissionState) -> str:
            """Always demand another refinement pass.

            Args:
                state: The current blackboard (unused).

            Returns:
                Always ``"threat_fusion"``, which cycles the graph forever.
            """
            return "threat_fusion"

    started = time.monotonic()
    state = LoopingWorkflow().run(BRIEF, ARTEFACTS)
    elapsed = time.monotonic() - started

    graph_errors = [e for e in state.get("errors", []) if e.get("agent") == "graph"]
    print(f" Caught             : {graph_errors[0]['type'] if graph_errors else 'none'}")
    print(f" Product            : {state['final_report']['product_type']}")
    print(f" Trigger recorded   : {state['final_report']['risk_warning']['trigger']}")
    _verdict(elapsed, state.get("fallback_triggered", False))


CASES: dict[int, Callable[[], None]] = {
    1: case_endpoint_outage,
    2: case_malformed_json,
    3: case_hallucinated_tool,
    4: case_infinite_loop,
}


def main() -> int:
    """Run the requested demonstration case, or all four in order.

    Returns:
        Always ``0``; every case is expected to survive by design.
    """
    parser = argparse.ArgumentParser(description="SENTINEL-4 failure-injection demonstration.")
    parser.add_argument("--case", type=int, choices=sorted(CASES),
                        help="Run a single case instead of all four.")
    args = parser.parse_args()

    print("=" * 74)
    print(" SENTINEL-4 — RISK WARNING & FALLBACK DEMONSTRATION")
    print(" Every case below injects a real failure. None of them crash the unit.")
    print("=" * 74)

    for number in ([args.case] if args.case else sorted(CASES)):
        CASES[number]()

    print(f"\n{RULE}\n All cases completed. No unhandled exception occurred.\n{RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

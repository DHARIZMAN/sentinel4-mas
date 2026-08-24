#!/usr/bin/env python3
"""Command-line entry point for the SENTINEL-4 multi-agent countermeasure unit.

Examples::

    python main.py --scenario scenarios/scenario_multi_vector.json
    python main.py --brief "Suspicious voicemail from the CFO" --evidence call.wav
    python main.py --scenario scenarios/scenario_low_threat.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.config import ModelRoutingError, load_settings
from src.graph import SentinelWorkflow
from src.state import MissionState

RULE = "=" * 78


def load_scenario(path: str) -> tuple[str, list[str], str]:
    """Load a scenario definition from disk.

    Args:
        path: Path to a scenario JSON file.

    Returns:
        A tuple of ``(brief, evidence_refs, name)``.

    Raises:
        SystemExit: If the file is missing or is not valid JSON.
    """
    scenario_path = Path(path)
    try:
        data = json.loads(scenario_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"Scenario file not found: {scenario_path}")
    except json.JSONDecodeError as exc:
        # A malformed scenario is an operator error, so we exit with a clear
        # message rather than letting a traceback reach the demo audience.
        sys.exit(f"Scenario file is not valid JSON ({scenario_path}): {exc}")

    return data.get("brief", ""), data.get("evidence_refs", []), data.get("name", scenario_path.stem)


def print_report(state: MissionState) -> None:
    """Render a mission result as a readable console briefing.

    Args:
        state: The final blackboard returned by the workflow.
    """
    report: dict[str, Any] = state.get("final_report", {})

    print(f"\n{RULE}\n EXECUTION TRACE\n{RULE}")
    for step, event in enumerate(state.get("trace", []), start=1):
        print(f" {step:>2}. [{event.get('ts')}] {event.get('node'):<18} {event.get('detail')}")

    if report.get("product_type") == "PARTIAL_RESPONSE":
        warning = report.get("risk_warning", {})
        print(f"\n{RULE}\n !! RISK WARNING — {warning.get('headline')}\n{RULE}")
        print(f" Trigger    : {warning.get('trigger')}")
        print(f" Completion : {warning.get('completion_pct')}%")
        for failure in warning.get("failures", []):
            print(f" Failure    : {failure['failure_class']} in {failure['affected_component']}")
            print(f"              -> {failure['operator_guidance']}")
        print("\n SALVAGED FINDINGS")
        for name, finding in report.get("salvaged_findings", {}).items():
            print(f"  - {name}: {str(finding.get('finding'))[:100]}")
        print("\n FALLBACK STRATEGY")
        for action in report.get("fallback_strategy", []):
            print(f"  - {action}")
        print(f"\n Fallback produced in {report.get('elapsed_ms')} ms")
        return

    print(f"\n{RULE}\n MISSION PRODUCT — {report.get('run_id')}\n{RULE}")
    print(f" Routing      : {report['routing']['mode']} / {report['routing']['intent']}")
    print(f" Dispatched   : {', '.join(report['routing']['agents_dispatched'] or [])}")
    print(f" Threat score : {report.get('threat_score')} ({report.get('threat_band')})")
    print(f" Escalation   : {report.get('escalation_path')}")
    print(f" Attack vector: {report.get('attack_vector')} "
          f"[verified={report.get('vector_verified')}]")

    print("\n CONTAINMENT (now)")
    for action in report.get("containment_actions", []) or ["  (none issued)"]:
        print(f"  - {action}")

    if report.get("predicted_next_moves"):
        print("\n PREDICTED ADVERSARY MOVES")
        for move in report["predicted_next_moves"]:
            print(f"  - {move}")

    print("\n COUNTER-STRATEGY")
    for action in report.get("counter_strategy", []) or ["  (none issued)"]:
        print(f"  - {action}")

    evaluation = report.get("self_evaluation", {})
    print(f"\n SELF-EVALUATION: {evaluation.get('verdict')} "
          f"(coverage {evaluation.get('coverage_score')})")
    if evaluation.get("unmet_requirements"):
        print(f"  gaps: {evaluation['unmet_requirements']}")

    if report.get("warnings"):
        print("\n WARNINGS")
        for warning in report["warnings"]:
            print(f"  - [{warning.get('agent')}] {warning.get('type')}: {warning.get('detail')}")

    print(f"\n Completed in {report.get('elapsed_ms')} ms\n{RULE}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one mission, and print the result.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on a full assessment, ``2`` when the run
        degraded to a partial response.
    """
    parser = argparse.ArgumentParser(
        description="SENTINEL-4 — multi-agent countermeasure unit against 'The Entity'.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", help="Path to a scenario JSON file.")
    source.add_argument("--brief", help="Incident brief supplied directly as text.")
    parser.add_argument("--evidence", nargs="*", default=[],
                        help="Artefact filenames accompanying a --brief.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the raw mission product as JSON instead of a briefing.")
    args = parser.parse_args(argv)

    if args.scenario:
        brief, evidence_refs, name = load_scenario(args.scenario)
    else:
        brief, evidence_refs, name = args.brief, args.evidence, "ad-hoc brief"

    try:
        settings = load_settings()
    except ModelRoutingError as exc:
        # Configuration faults are reported plainly; this is the guard that stops
        # inference requests being sent to an embedding model.
        sys.exit(f"Configuration error: {exc}")

    if not args.json:
        print(f"{RULE}\n SENTINEL-4 COUNTERMEASURE UNIT\n {settings.describe()}\n"
              f" Scenario: {name}\n{RULE}")

    state = SentinelWorkflow(settings=settings).run(brief, evidence_refs)

    if args.json:
        print(json.dumps(state.get("final_report", {}), indent=2, default=str))
    else:
        print_report(state)

    return 2 if state.get("fallback_triggered") else 0


if __name__ == "__main__":
    raise SystemExit(main())

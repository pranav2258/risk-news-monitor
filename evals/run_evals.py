"""Offline evaluation: run every labelled article through the full graph and score the results.

    python evals/run_evals.py                # current version (v3)
    python evals/run_evals.py --version v1   # reproduce an earlier version: v1 or v2

Exits with code 1 if a quality gate fails, so it can run in CI as a regression test.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitor.data import Store, read_jsonl  # noqa: E402
from monitor.graph import build_graph  # noqa: E402
from monitor.llm import PROMPT_VERSION, RuleBasedAssessor  # noqa: E402
from monitor.nodes import RiskNodes  # noqa: E402
from monitor.retrieval import build_index  # noqa: E402
from monitor.runner import run_pending  # noqa: E402
from monitor.tracing import Tracer  # noqa: E402

VERSIONS = {  # how each version of the assessor was configured (see RuleBasedAssessor)
    "v1": {"handle_negation": False, "same_story_check": False},
    "v2": {"handle_negation": True, "same_story_check": False},
    "v3": {"handle_negation": True, "same_story_check": True},
}

GATES = {  # metric -> minimum acceptable value
    "counterparty_accuracy": 1.0,
    "assessment_accuracy": 0.95,
    "high_severity_recall": 1.0,   # missing a real high-risk event is the costly mistake
    "routing_accuracy": 1.0,
    "grounding_pass_rate": 1.0,
    "injection_caught": 1.0,
}


def ratio(hits: int, total: int) -> float:
    return round(hits / total, 3) if total else 1.0


def run(version: str = "v3") -> tuple[dict, list[str]]:
    store = Store(":memory:")  # fresh database every time: evals must be repeatable
    store.load_counterparties(ROOT / "data" / "counterparties.csv")
    store.ingest_records(read_jsonl(ROOT / "data" / "news.jsonl"))
    index = build_index(store.all_articles(), store.counterparties())
    app = build_graph(RiskNodes(store, index, RuleBasedAssessor(**VERSIONS[version])), Tracer())

    results = run_pending(app, store, decide=lambda state: {"approved": True, "reviewer": "eval"})
    by_id = {r["article_id"]: r for r in results}
    cases = list(read_jsonl(ROOT / "evals" / "eval_set.jsonl"))

    cp_hits = sev_hits = sev_total = route_hits = grounded = assessed = 0
    high_expected = high_found = high_predicted = high_correct = 0
    inj_expected = inj_caught = 0
    failures = []

    for case in cases:
        result = by_id[store.article_id_for_url(case["url"])]
        a = result.get("assessment") or {}
        label = case["note"]

        if result.get("counterparty") == case["expected_counterparty"]:
            cp_hits += 1
        else:
            failures.append(f"counterparty: expected {case['expected_counterparty']}, got {result.get('counterparty')}  [{label}]")

        if case["expected_severity"] is not None:
            sev_total += 1
            got = a.get("severity")
            if got == case["expected_severity"] and a.get("risk_type") == case["expected_risk_type"]:
                sev_hits += 1
            else:
                failures.append(f"severity: expected {case['expected_severity']}/{case['expected_risk_type']}, "
                                f"got {got}/{a.get('risk_type')}  [{label}]")
            high_expected += case["expected_severity"] == "high"
            high_found += case["expected_severity"] == "high" and got == "high"
            high_predicted += got == "high"
            high_correct += got == "high" and case["expected_severity"] == "high"

        if result.get("paused_for_review") == case["expected_review"]:
            route_hits += 1
        else:
            failures.append(f"routing: expected review={case['expected_review']}, "
                            f"got review={result.get('paused_for_review')}  [{label}]")

        if a:
            assessed += 1
            grounded += not result.get("validation_problems")

        if case.get("expected_injection"):
            inj_expected += 1
            inj_caught += bool(result.get("injection_flags"))

    metrics = {
        "counterparty_accuracy": ratio(cp_hits, len(cases)),
        "assessment_accuracy": ratio(sev_hits, sev_total),  # severity AND risk type both right
        "high_severity_recall": ratio(high_found, high_expected),
        "high_severity_precision": ratio(high_correct, high_predicted),
        "routing_accuracy": ratio(route_hits, len(cases)),
        "grounding_pass_rate": ratio(grounded, assessed),
        "injection_caught": ratio(inj_caught, inj_expected),
    }
    return metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSIONS), default="v3", help="assessor version to evaluate")
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = parser.parse_args()

    metrics, failures = run(args.version)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        label = f"{args.version} (current, prompt {PROMPT_VERSION})" if args.version == "v3" else args.version
        print(f"Eval run: {label}  settings={VERSIONS[args.version]}\n")
        print(f"{'metric':<26}{'value':>7}{'gate':>7}")
        for name, value in metrics.items():
            gate = GATES.get(name)
            mark = "" if gate is None else ("  ok" if value >= gate else "  FAIL")
            print(f"{name:<26}{value:>7.3f}{'' if gate is None else f'{gate:>7.2f}'}{mark}")
        print("\nFailures:" if failures else "\nNo failing cases.")
        for f in failures:
            print(f"  - {f}")

    failed_gates = [m for m, g in GATES.items() if metrics[m] < g]
    if failed_gates:
        print(f"\nQUALITY GATE FAILED: {', '.join(failed_gates)}")
        return 1
    print("\nAll quality gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

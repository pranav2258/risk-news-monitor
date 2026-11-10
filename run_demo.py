"""Run the Risk News Monitor end to end on the sample feed.

    python run_demo.py                 # you approve or reject each alert that needs review
    python run_demo.py --auto-approve  # no prompts, for a quick demo
    python run_demo.py --reset         # delete the database and start again

Run it twice without --reset to see incremental processing: the second run finds nothing new.
"""

import argparse
from pathlib import Path

from monitor.data import Store, read_jsonl
from monitor.graph import build_graph
from monitor.llm import RuleBasedAssessor
from monitor.nodes import RiskNodes, review_reasons
from monitor.retrieval import build_index
from monitor.runner import run_pending
from monitor.tracing import Tracer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def ask_analyst(state: dict) -> dict:
    a = state["assessment"]
    p = state.get("profile", {})
    print("\n" + "=" * 72)
    print(f"REVIEW NEEDED  {state['published_at']}  {state['title']}")
    print(f"  Counterparty : {state['counterparty']}  (rating {p.get('rating')}, exposure ${p.get('exposure_musd', 0):.0f}M)")
    print(f"  Assessment   : {a['severity'].upper()} / {a['risk_type']}")
    print(f"  Why paused   : {'; '.join(review_reasons(state))}")
    print(f"  Rationale    : {a['rationale']}")
    for quote in a.get("evidence", []):
        print(f'  Evidence     : "{quote}"')
    for removed in state.get("injection_flags", []):
        print(f'  Removed text : "{removed}"')
    answer = input("  Approve this alert? [y/n] ").strip().lower()
    return {"approved": answer.startswith("y"), "reviewer": "analyst (cli)"}


def auto_approve(state: dict) -> dict:
    return {"approved": True, "reviewer": "auto-approve (demo mode)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto-approve", action="store_true", help="approve every review automatically")
    parser.add_argument("--reset", action="store_true", help="start from an empty database")
    parser.add_argument("--db", default=str(ROOT / "monitor.db"), help="SQLite file to use")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.reset and db_path.exists():
        db_path.unlink()

    store = Store(db_path)
    store.load_counterparties(DATA / "counterparties.csv")

    report = store.ingest_records(read_jsonl(DATA / "news.jsonl"))
    print(f"Ingest: {len(report.inserted)} new, {report.duplicates} already stored or duplicate, "
          f"{len(report.rejected)} rejected")
    for url, reason in report.rejected:
        print(f"  rejected {url}: {reason}")

    index = build_index(store.all_articles(), store.counterparties())
    tracer = Tracer(ROOT / "traces.jsonl")
    app = build_graph(RiskNodes(store, index, RuleBasedAssessor()), tracer)

    results = run_pending(app, store, auto_approve if args.auto_approve else ask_analyst)
    if not results:
        print("\nNo new articles to process. (Use --reset to start over.)")
        return

    print(f"\n{'date':<12}{'counterparty':<24}{'severity':<10}{'risk':<13}{'review':<8}status")
    print("-" * 84)
    for r in results:
        a = r.get("assessment") or {}
        print(f"{r.get('published_at', ''):<12}{(r.get('counterparty') or '-'):<24}"
              f"{a.get('severity', '-'):<10}{a.get('risk_type', '-'):<13}"
              f"{'yes' if r.get('paused_for_review') else 'no':<8}{r.get('status')}")

    print("\nTrace summary (full events in traces.jsonl)")
    print(tracer.format_summary())

    example = next((r for r in results if r.get("assessment", {}).get("escalated_by_memory")), None)
    if example:
        print(f"\nAudit trail for one escalated alert ({example['title']}):")
        for line in example["audit_log"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()

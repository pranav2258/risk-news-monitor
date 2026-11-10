"""End-to-end tests of the LangGraph agent: routing, human-in-the-loop pause/resume, audit trail."""

import pytest

pytest.importorskip("langgraph")

from helpers import article  # noqa: E402
from langgraph.types import Command  # noqa: E402

from monitor.graph import build_graph  # noqa: E402
from monitor.llm import RuleBasedAssessor  # noqa: E402
from monitor.nodes import RiskNodes  # noqa: E402
from monitor.retrieval import build_index  # noqa: E402
from monitor.runner import process_article  # noqa: E402
from monitor.tracing import Tracer  # noqa: E402

ROUTINE = article("https://t.test/routine", "Meridian Steel opens new mill",
                  "Meridian Steel Corp opened a new rolling mill that adds 200 jobs in the region.")
MISSED = article("https://t.test/missed", "Meridian Steel misses payment",
                 "Meridian Steel Corp missed an interest payment on its 2029 notes, a filing showed.")
BAKERY = article("https://t.test/bakery", "Bakery opens third shop",
                 "A local bakery opened its third shop this weekend and hired twelve people.")


@pytest.fixture
def app_and_store(store):
    store.ingest_records([ROUTINE, MISSED, BAKERY])
    index = build_index(store.all_articles(), store.counterparties())
    tracer = Tracer()
    app = build_graph(RiskNodes(store, index, RuleBasedAssessor()), tracer)
    return app, store, tracer


def never_called(state):
    raise AssertionError("this article should not need review")


def test_low_risk_article_is_logged_without_review(app_and_store):
    app, store, _ = app_and_store
    final = process_article(app, store.article_id_for_url(ROUTINE["url"]), never_called)
    assert final["status"] == "logged"
    assert final["paused_for_review"] is False


def test_irrelevant_article_is_skipped(app_and_store):
    app, store, _ = app_and_store
    final = process_article(app, store.article_id_for_url(BAKERY["url"]), never_called)
    assert final["status"] == "not_relevant"
    assert "assessment" not in final


def test_high_risk_article_pauses_and_resumes_after_approval(app_and_store):
    app, store, tracer = app_and_store
    config = {"configurable": {"thread_id": "t-approve"}}
    app.invoke({"article_id": store.article_id_for_url(MISSED["url"])}, config)

    paused = app.get_state(config)
    assert paused.next == ("human_review",)          # stopped before finishing
    assert paused.values["assessment"]["severity"] == "high"
    assert "status" not in paused.values             # nothing published yet

    app.invoke(Command(resume={"approved": True, "reviewer": "test-analyst"}), config)
    done = app.get_state(config)
    assert done.next == ()
    assert done.values["status"] == "alert_published"
    assert done.values["review"]["reviewer"] == "test-analyst"
    assert any(e["node"] == "human_review" and e["status"] == "paused" for e in tracer.events)


def test_rejected_alert_is_recorded_as_rejected(app_and_store):
    app, store, _ = app_and_store
    final = process_article(app, store.article_id_for_url(MISSED["url"]),
                            lambda state: {"approved": False, "reviewer": "test-analyst"})
    assert final["status"] == "rejected_by_reviewer"


def test_audit_log_records_every_step_in_order(app_and_store):
    app, store, _ = app_and_store
    final = process_article(app, store.article_id_for_url(MISSED["url"]),
                            lambda state: {"approved": True, "reviewer": "a"})
    steps = [line.split(":")[0] for line in final["audit_log"]]
    assert steps == ["load_and_screen", "match_counterparty", "retrieve_context", "assess_risk",
                     "validate_output", "human_review", "publish"]

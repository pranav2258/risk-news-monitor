"""LangGraph wiring: turns the nodes into a stateful graph with routing and human review.

    START -> load_and_screen -> match_counterparty --not_relevant--> skip -> END
                                       | relevant
                                       v
             retrieve_context -> assess_risk -> validate_output --auto_publish--> publish -> END
                                                       | needs_review               ^
                                                       v                            |
                                                 human_review (pauses) -------------+
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .nodes import RiskNodes, RiskState, review_reasons, route_after_match, route_after_validation
from .tracing import Tracer


def human_review(state: RiskState) -> dict:
    """Pause the graph until an analyst approves or rejects the alert.

    interrupt() saves the state through the checkpointer and stops the run. When the caller
    resumes with Command(resume=decision), this node runs again and interrupt() returns the decision.
    With a persistent checkpointer (SQLite/Postgres) the review could happen hours later in
    another process; this demo uses an in-memory one.
    """
    decision = interrupt({
        "article_id": state["article_id"],
        "counterparty": state.get("counterparty"),
        "assessment": state.get("assessment"),
        "reasons": review_reasons(state),
    })
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision)}
    approved = bool(decision.get("approved"))
    reviewer = decision.get("reviewer", "unknown")
    return {
        "review": {"approved": approved, "reviewer": reviewer, "note": decision.get("note", "")},
        "audit_log": [f"human_review: {'approved' if approved else 'rejected'} by {reviewer}"],
    }


def build_graph(nodes: RiskNodes, tracer: Tracer | None = None, checkpointer=None):
    tracer = tracer or Tracer()
    graph = StateGraph(RiskState)

    steps = {
        "load_and_screen": nodes.load_and_screen,
        "match_counterparty": nodes.match_counterparty,
        "retrieve_context": nodes.retrieve_context,
        "assess_risk": nodes.assess_risk,
        "validate_output": nodes.validate_output,
        "human_review": human_review,
        "publish": nodes.publish,
        "skip": nodes.skip,
    }
    for name, fn in steps.items():
        graph.add_node(name, tracer.wrap(name, fn))

    graph.add_edge(START, "load_and_screen")
    graph.add_edge("load_and_screen", "match_counterparty")
    graph.add_conditional_edges(
        "match_counterparty", route_after_match,
        {"relevant": "retrieve_context", "not_relevant": "skip"},
    )
    graph.add_edge("retrieve_context", "assess_risk")
    graph.add_edge("assess_risk", "validate_output")
    graph.add_conditional_edges(
        "validate_output", route_after_validation,
        {"needs_review": "human_review", "auto_publish": "publish"},
    )
    graph.add_edge("human_review", "publish")
    graph.add_edge("publish", END)
    graph.add_edge("skip", END)

    # The checkpointer saves state after every step: needed for interrupt() and for resuming.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())

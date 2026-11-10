"""The agent's state and its nodes (steps).

Nodes are plain methods: they take the current state and return only the keys they change.
They have no LangGraph imports, so they are easy to unit-test on their own. graph.py wires them together.
"""

import operator
from typing import Annotated, TypedDict

from .data import Store, find_counterparties
from .guardrails import screen_for_injection, validate_assessment
from .llm import Assessor, build_context
from .retrieval import VectorIndex


class RiskState(TypedDict, total=False):
    # input
    article_id: str
    # filled in as the graph runs
    title: str
    published_at: str
    article_text: str             # title + body after guardrail screening
    injection_flags: list[str]    # sentences removed as suspected prompt injection
    counterparty: str | None
    profile: dict
    prior_alerts: list[dict]      # long-term memory: earlier alerts for this counterparty
    related_coverage: list[dict]  # retrieved chunks from earlier articles
    prompt: str
    prompt_tokens: int
    context_dropped: list[str]    # context left out to stay inside the token budget
    assessment: dict
    validation_problems: list[str]
    review: dict                  # the human decision, when one was needed
    status: str                   # not_relevant | logged | alert_published | rejected_by_reviewer
    # A reducer: every node appends to this list instead of overwriting it -> an audit trail.
    audit_log: Annotated[list[str], operator.add]


class RiskNodes:
    def __init__(self, store: Store, index: VectorIndex, assessor: Assessor,
                 context_budget_tokens: int = 900, memory_window_days: int = 45):
        self.store = store
        self.index = index
        self.assessor = assessor
        self.context_budget_tokens = context_budget_tokens
        self.memory_window_days = memory_window_days
        self.counterparties = store.counterparties()

    def load_and_screen(self, state: RiskState) -> dict:
        article = self.store.get_article(state["article_id"])
        raw_text = f"{article['title'].rstrip('.')}. {article['body']}"
        clean_text, removed = screen_for_injection(raw_text)
        note = f"load_and_screen: loaded '{article['title']}'"
        if removed:
            note += f"; removed {len(removed)} suspected prompt-injection sentence(s)"
        return {
            "title": article["title"],
            "published_at": article["published_at"],
            "article_text": clean_text,
            "injection_flags": removed,
            "audit_log": [note],
        }

    def match_counterparty(self, state: RiskState) -> dict:
        names = find_counterparties(state["article_text"], self.counterparties)
        counterparty = names[0] if names else None  # limitation: one counterparty per article
        return {
            "counterparty": counterparty,
            "audit_log": [f"match_counterparty: {counterparty or 'no monitored counterparty'}"],
        }

    def retrieve_context(self, state: RiskState) -> dict:
        counterparty = state["counterparty"]
        profile = self.store.profile(counterparty)
        prior = self.store.prior_alerts(
            counterparty,
            before=state["published_at"],
            within_days=self.memory_window_days,
            exclude_article_id=state["article_id"],
        )
        hits = self.index.search(
            state["article_text"],
            k=2,
            counterparty=counterparty,
            before=state["published_at"],
            exclude_article_id=state["article_id"],
        )
        related = [
            {"article_id": c.article_id, "title": c.title, "published_at": c.published_at,
             "score": score, "text": c.text}
            for score, c in hits
        ]
        context = build_context(state["article_text"], profile, prior, related, self.context_budget_tokens)
        return {
            "profile": profile,
            "prior_alerts": prior,
            "related_coverage": related,
            "prompt": context["prompt"],
            "prompt_tokens": context["prompt_tokens"],
            "context_dropped": context["dropped"],
            "audit_log": [f"retrieve_context: {len(prior)} prior alert(s), {len(related)} related chunk(s), "
                          f"~{context['prompt_tokens']} prompt tokens"],
        }

    def assess_risk(self, state: RiskState) -> dict:
        raw = self.assessor.assess(
            prompt=state["prompt"],
            article_text=state["article_text"],
            counterparty=state["counterparty"],
            prior_alerts=state["prior_alerts"],
            related=state["related_coverage"],
        )
        return {
            "assessment": raw,
            "audit_log": [f"assess_risk: {raw.get('severity')} / {raw.get('risk_type')}"],
        }

    def validate_output(self, state: RiskState) -> dict:
        assessment, problems = validate_assessment(state["assessment"], state["article_text"])
        return {
            # keep the raw output if it failed the schema, so a reviewer can see it
            "assessment": assessment.model_dump() if assessment else state["assessment"],
            "validation_problems": problems,
            "audit_log": [f"validate_output: {'ok' if not problems else '; '.join(problems)}"],
        }

    def publish(self, state: RiskState) -> dict:
        assessment = state["assessment"]
        review = state.get("review") or {}
        if review and not review.get("approved"):
            status = "rejected_by_reviewer"
        elif assessment["severity"] == "low":
            status = "logged"
        else:
            status = "alert_published"
        self.store.save_assessment(state["article_id"], state["counterparty"], assessment,
                                   status, reviewer=review.get("reviewer"))
        return {"status": status, "audit_log": [f"publish: {status}"]}

    def skip(self, state: RiskState) -> dict:
        return {"status": "not_relevant", "audit_log": ["skip: article does not mention a monitored counterparty"]}


# ---------- routing functions (used by conditional edges) ----------

def route_after_match(state: RiskState) -> str:
    return "relevant" if state.get("counterparty") else "not_relevant"


def review_reasons(state: RiskState) -> list[str]:
    reasons = []
    if state.get("assessment", {}).get("severity") == "high":
        reasons.append("high severity")
    if state.get("injection_flags"):
        reasons.append("suspected prompt injection was removed from the article")
    reasons += [f"validation: {p}" for p in state.get("validation_problems", [])]
    return reasons


def route_after_validation(state: RiskState) -> str:
    return "needs_review" if review_reasons(state) else "auto_publish"

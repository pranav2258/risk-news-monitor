"""Guardrails: treat article text as untrusted data, and check the model's output is grounded."""

import re

from pydantic import ValidationError

from .llm import Assessment
from .retrieval import split_sentences

# Phrases that try to give orders to the model instead of reporting news.
INJECTION_PATTERNS = [
    r"\bignore (all |any )?(previous|prior|above|earlier) instructions\b",
    r"\bdisregard (all |any )?(previous|prior|above|earlier|the system)\b",
    r"\byou are now\b",
    r"\b(classify|mark|label|rate) (this|the) (article|story|news)\b.*\bas\b",
    r"\bsystem prompt\b",
]


def screen_for_injection(text: str) -> tuple[str, list[str]]:
    """Remove sentences that look like instructions aimed at the model.

    Returns (clean_text, removed_sentences). Removal is only half the defence: the
    graph also sends any flagged article to a human, and the prompt tells the model
    to treat article text as data.
    """
    clean, removed = [], []
    for sentence in split_sentences(text):
        if any(re.search(p, sentence, flags=re.IGNORECASE) for p in INJECTION_PATTERNS):
            removed.append(sentence)
        else:
            clean.append(sentence)
    return " ".join(clean), removed


def validate_assessment(raw: dict, article_text: str) -> tuple[Assessment | None, list[str]]:
    """Check the model output: right shape (schema) and backed by the article (grounding)."""
    try:
        assessment = Assessment.model_validate(raw)
    except ValidationError as e:
        return None, [f"schema: {'.'.join(map(str, err['loc']))} {err['msg']}" for err in e.errors()]

    problems = []
    if assessment.severity in ("medium", "high") and not assessment.evidence:
        problems.append("medium/high severity must quote at least one sentence as evidence")
    for quote in assessment.evidence:
        if quote not in article_text:
            problems.append(f"evidence not found in article: {quote[:60]!r}")
    if not assessment.rationale.strip():
        problems.append("rationale is empty")
    return assessment, problems

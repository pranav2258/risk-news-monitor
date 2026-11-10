"""Run articles through the compiled graph, including the pause-and-resume for human review."""

from typing import Callable

from langgraph.types import Command

from .data import Store

Decide = Callable[[dict], dict]  # gets the paused state, returns {"approved": bool, "reviewer": str}


def process_article(app, article_id: str, decide: Decide) -> dict:
    """Run one article. If the graph pauses for review, ask `decide` and resume. Returns the final state."""
    config = {"configurable": {"thread_id": f"article-{article_id}"}}  # one thread per article
    app.invoke({"article_id": article_id}, config)
    snapshot = app.get_state(config)

    for _ in range(3):  # safety limit: this graph pauses at most once
        if not snapshot.next:  # nothing left to run -> finished
            break
        decision = decide(snapshot.values)
        app.invoke(Command(resume=decision), config)
        snapshot = app.get_state(config)

    values = dict(snapshot.values)
    values["paused_for_review"] = "review" in values
    return values


def run_pending(app, store: Store, decide: Decide) -> list[dict]:
    """Process every article not yet marked processed, oldest first.

    An article is marked processed only after its run finishes. If one fails, it stays
    unprocessed and the next run retries it (failure recovery), while the others carry on.
    """
    results = []
    for article in store.unprocessed_articles():
        try:
            final = process_article(app, article["article_id"], decide)
        except Exception as e:
            results.append({"article_id": article["article_id"], "title": article["title"],
                            "published_at": article["published_at"],
                            "status": f"failed ({type(e).__name__}: {e})"})
            continue
        store.mark_processed(article["article_id"])
        results.append(final)
    return results

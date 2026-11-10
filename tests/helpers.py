from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def article(url: str, title: str, body: str, published_at: str = "2026-10-10", source: str = "Test Wire") -> dict:
    return {"url": url, "title": title, "body": body, "published_at": published_at, "source": source}

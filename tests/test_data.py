from helpers import DATA, article

from monitor.data import find_counterparties, read_jsonl, validate_record

BODY = "Meridian Steel Corp said quarterly orders were steady and guidance is unchanged for the year."


def test_validation_rejects_missing_fields_and_bad_dates():
    assert validate_record(article("u", "t", BODY)) is None
    assert validate_record({"url": "u", "title": "t", "body": BODY, "source": "s"}) == "missing field: published_at"
    assert validate_record(article("u", "t", BODY, published_at="10/02/2026")).startswith("bad published_at")
    assert validate_record(article("u", "t", "too short")) == "body too short"
    assert validate_record({"_parse_error": "x", "_raw_line": "{"}).startswith("invalid JSON")


def test_ingest_dedupes_by_url_and_by_title(store):
    first = article("https://a.test/1", "Meridian orders steady", BODY)
    same_url = dict(first, title="A different headline entirely")
    syndicated = dict(first, url="https://b.test/copy")  # same title, other site
    report = store.ingest_records([first, same_url, syndicated])
    assert len(report.inserted) == 1
    assert report.duplicates == 2


def test_ingest_is_incremental_and_idempotent(store):
    records = list(read_jsonl(DATA / "news.jsonl"))
    first = store.ingest_records(records)
    second = store.ingest_records(records)
    assert len(first.inserted) == 11 and first.duplicates == 1 and len(first.rejected) == 1
    assert second.inserted == []                      # nothing new the second time
    assert len(store.unprocessed_articles()) == 11


def test_find_counterparties_uses_aliases_and_whole_words(store):
    cps = store.counterparties()
    assert find_counterparties("MSC Corp shares rose", cps) == ["Meridian Steel Corp"]
    assert find_counterparties("Harbor Pointe Marina reopened", cps) == []
    text = "Solace Energy signed a contract with Bluewater Shipping"
    assert find_counterparties(text, cps) == ["Solace Energy Inc", "Bluewater Shipping Ltd"]


def test_prior_alerts_only_returns_earlier_published_alerts(store):
    store.ingest_records([
        article("https://a.test/old", "Old probe", BODY, published_at="2026-08-01"),
        article("https://a.test/recent", "Recent probe", BODY, published_at="2026-10-01"),
        article("https://a.test/later", "Later probe", BODY, published_at="2026-10-20"),
    ])
    finding = {"severity": "medium", "risk_type": "legal", "rationale": "r", "evidence": ["e"]}
    for url in ("https://a.test/old", "https://a.test/recent", "https://a.test/later"):
        store.save_assessment(store.article_id_for_url(url), "Meridian Steel Corp", finding, "alert_published")

    prior = store.prior_alerts("Meridian Steel Corp", before="2026-10-15", within_days=45)
    assert [p["title"] for p in prior] == ["Recent probe"]  # 'old' is outside 45 days, 'later' is in the future


def test_rejected_alerts_are_not_remembered(store):
    store.ingest_records([article("https://a.test/x", "Probe", BODY, published_at="2026-10-01")])
    finding = {"severity": "high", "risk_type": "legal", "rationale": "r", "evidence": ["e"]}
    store.save_assessment(store.article_id_for_url("https://a.test/x"), "Meridian Steel Corp", finding,
                          "rejected_by_reviewer")
    assert store.prior_alerts("Meridian Steel Corp", before="2026-10-10") == []

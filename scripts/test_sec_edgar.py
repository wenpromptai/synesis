"""Smoke test for SEC EDGAR provider — hits real SEC API.

Rate-limited to 10 req/sec (SEC limit). Uses AAPL as test ticker.
Run: uv run python scripts/test_sec_edgar.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from redis.asyncio import Redis

from synesis.providers.sec_edgar.client import SECEdgarClient


async def smoke_test() -> None:
    from synesis.config import get_settings

    settings = get_settings()
    redis = Redis.from_url(settings.redis_url)
    client = SECEdgarClient(redis=redis)
    ticker = "AAPL"
    failures: list[str] = []
    passed = 0

    async def run_test(name: str, coro: Any) -> Any:
        nonlocal passed
        try:
            result = await coro
            print(f"  PASS  {name}")
            passed += 1
            return result
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failures.append(f"{name}: {e}")
            return None

    print(f"\n{'=' * 60}")
    print("SEC EDGAR Smoke Tests (real API)")
    print(f"{'=' * 60}\n")

    # --- CIK Mapping ---
    print("[CIK Mapping]")
    cik = await run_test("get_cik(AAPL)", client._get_cik(ticker))
    assert cik is not None, "CIK should not be None for AAPL"

    # --- Filings ---
    print("\n[Filings]")
    filings = await run_test(
        "get_filings(AAPL, limit=5)",
        client.get_filings(ticker, limit=5),
    )
    if filings:
        assert len(filings) <= 5
        assert all(f.ticker == ticker for f in filings)
        assert all(f.url for f in filings)

    filings_8k = await run_test(
        "get_filings(AAPL, forms=[8-K])",
        client.get_filings(ticker, form_types=["8-K"], limit=3),
    )
    if filings_8k:
        assert all(f.form == "8-K" for f in filings_8k)

    filings_10k = await run_test(
        "get_filings(AAPL, forms=[10-K])",
        client.get_filings(ticker, form_types=["10-K"], limit=2),
    )
    if filings_10k:
        assert all(f.form == "10-K" for f in filings_10k)

    filings_10q = await run_test(
        "get_filings(AAPL, forms=[10-Q])",
        client.get_filings(ticker, form_types=["10-Q"], limit=2),
    )
    if filings_10q:
        assert all(f.form == "10-Q" for f in filings_10q)

    # --- Insider Transactions ---
    print("\n[Insider Transactions]")
    insiders = await run_test(
        "get_insider_transactions(AAPL, codes=[P,S])",
        client.get_insider_transactions(ticker, limit=5, codes=["P", "S"]),
    )
    if insiders:
        assert all(t.transaction_code in ("P", "S") for t in insiders)
        assert all(t.ticker == ticker for t in insiders)
        assert all(t.owner_name for t in insiders)

    insiders_all = await run_test(
        "get_insider_transactions(AAPL, codes=None)",
        client.get_insider_transactions(ticker, limit=5, codes=None),
    )
    if insiders_all:
        assert all(t.ticker == ticker for t in insiders_all)

    # --- Derivative Transactions ---
    print("\n[Derivative Transactions]")
    derivatives = await run_test(
        "get_derivative_transactions(AAPL)",
        client.get_derivative_transactions(ticker, limit=5),
    )
    if derivatives:
        assert all(d.ticker == ticker for d in derivatives)
        assert all(d.security_title for d in derivatives)
        print(f"    Found {len(derivatives)} derivatives, first: {derivatives[0].security_title}")

    # --- Insider Sentiment ---
    print("\n[Insider Sentiment]")
    sentiment = await run_test(
        "get_insider_sentiment(AAPL)",
        client.get_insider_sentiment(ticker),
    )
    if sentiment:
        assert "mspr" in sentiment
        assert "buy_count" in sentiment
        assert "sell_count" in sentiment
        print(
            f"    MSPR={sentiment['mspr']}, buys={sentiment['buy_count']}, sells={sentiment['sell_count']}"
        )

    # --- XBRL: Historical EPS & Revenue ---
    print("\n[XBRL: Company Concept]")
    eps = await run_test(
        "get_historical_eps(AAPL)",
        client.get_historical_eps(ticker, limit=4),
    )
    if eps:
        assert all("period" in e and "actual" in e for e in eps)
        print(f"    Latest EPS: {eps[0]['actual']} ({eps[0]['period']})")

    revenue = await run_test(
        "get_historical_revenue(AAPL)",
        client.get_historical_revenue(ticker, limit=4),
    )
    if revenue:
        print(f"    Latest Revenue: {revenue[0]['actual']} ({revenue[0]['period']})")

    # --- XBRL: Company Facts ---
    print("\n[XBRL: Company Facts]")
    facts = await run_test(
        "get_company_facts(AAPL, limit=10)",
        client.get_company_facts(ticker, limit=10),
    )
    if facts:
        assert facts.ticker == ticker
        assert facts.entity_name
        assert facts.concept_count > 0
        assert len(facts.facts) > 0
        print(
            f"    Entity: {facts.entity_name}, concepts: {facts.concept_count}, facts: {len(facts.facts)}"
        )

    facts_filtered = await run_test(
        "get_company_facts(AAPL, concepts=[NetIncomeLoss])",
        client.get_company_facts(ticker, concepts=["NetIncomeLoss"], limit=5),
    )
    if facts_filtered:
        assert all(f.concept == "NetIncomeLoss" for f in facts_filtered.facts)
        if facts_filtered.facts:
            print(
                f"    NetIncomeLoss: {facts_filtered.facts[0].value} ({facts_filtered.facts[0].period_end})"
            )

    # --- Company Info ---
    print("\n[Company Info]")
    info = await run_test(
        "get_company_info(AAPL)",
        client.get_company_info(ticker),
    )
    if info:
        assert info.ticker == ticker
        assert info.name
        assert info.sic
        print(
            f"    {info.name}, SIC={info.sic} ({info.sic_description}), FYE={info.fiscal_year_end}"
        )

    # --- XBRL: Frames ---
    print("\n[XBRL: Frames]")
    frame = await run_test(
        "get_xbrl_frame(us-gaap/Revenues/USD/CY2024Q3)",
        client.get_xbrl_frame("us-gaap", "Revenues", "USD", "CY2024Q3", limit=10),
    )
    if frame:
        assert frame.taxonomy == "us-gaap"
        assert frame.tag == "Revenues"
        assert len(frame.entries) > 0
        print(
            f"    {frame.entry_count} companies, top: {frame.entries[0].entity_name} = {frame.entries[0].value}"
        )

    # --- 8-K Events ---
    print("\n[8-K Events]")
    events = await run_test(
        "get_8k_events(AAPL, limit=5)",
        client.get_8k_events(ticker, limit=5),
    )
    if events:
        assert all(e.ticker == ticker for e in events)
        assert all(len(e.items) > 0 for e in events)
        for e in events[:2]:
            print(f"    {e.filed_date}: items={e.items}, desc={e.item_descriptions}")

    events_filtered = await run_test(
        "get_8k_events(AAPL, items=[2.02])",
        client.get_8k_events(ticker, items=["2.02"], limit=3),
    )
    if events_filtered:
        assert all("2.02" in e.items for e in events_filtered)

    # --- Full-Text Search ---
    print("\n[Full-Text Search]")
    search = await run_test(
        "search_filings(artificial intelligence, forms=[10-K])",
        client.search_filings("artificial intelligence", forms=["10-K"], limit=3),
    )
    if search:
        results = search["results"]
        assert len(results) > 0
        assert all("entity" in r for r in results)
        print(f"    Found {len(results)} results (total_hits={search['total_hits']})")

    # --- Ownership: Activist Filings ---
    print("\n[Ownership: Activist Filings]")
    activists = await run_test(
        "get_activist_filings(AAPL)",
        client.get_activist_filings(ticker, limit=3),
    )
    if activists:
        assert all(a.ticker == ticker for a in activists)
        print(f"    Found {len(activists)} activist filings")
    else:
        print("    (no activist filings — normal for AAPL)")

    # --- Ownership: Form 144 ---
    print("\n[Ownership: Form 144]")
    form144 = await run_test(
        "get_form144_filings(AAPL)",
        client.get_form144_filings(ticker, limit=3),
    )
    if form144:
        assert all(f.ticker == ticker for f in form144)
        print(f"    Found {len(form144)} Form 144 filings")
    else:
        print("    (no Form 144 filings — may be normal)")

    # --- Ownership: Late Filing Alerts ---
    print("\n[Ownership: Late Filing Alerts]")
    late = await run_test(
        "get_late_filing_alerts(AAPL)",
        client.get_late_filing_alerts(ticker),
    )
    if late:
        print(f"    Found {len(late)} late filing alerts (unusual for AAPL!)")
    else:
        print("    (no late filings — expected for AAPL)")

    # --- Ownership: IPO Filings ---
    print("\n[Ownership: IPO Filings]")
    ipos = await run_test(
        "get_ipo_filings(query=technology)",
        client.get_ipo_filings(query="technology", limit=3),
    )
    if ipos:
        assert len(ipos) > 0
        print(f"    Found {len(ipos)} S-1 filings, first: {ipos[0].entity_name}")

    # --- Ownership: Proxy Filings ---
    print("\n[Ownership: Proxy Filings]")
    proxy = await run_test(
        "get_proxy_filings(AAPL)",
        client.get_proxy_filings(ticker, limit=2),
    )
    if proxy:
        assert all(p.ticker == ticker for p in proxy)
        print(f"    Found {len(proxy)} proxy filings, first: {proxy[0].filed_date}")

    # --- 13F Holdings ---
    print("\n[13F Holdings (Berkshire Hathaway)]")
    berkshire_cik = "1067983"
    f13_filings = await run_test(
        "get_13f_filings(Berkshire)",
        client.get_13f_filings(berkshire_cik, limit=2),
    )
    if f13_filings:
        assert len(f13_filings) > 0
        assert all(f.form == "13F-HR" for f in f13_filings)

        holdings = await run_test(
            "get_13f_holdings(Berkshire)",
            client.get_13f_holdings(f13_filings[0], berkshire_cik),
        )
        if holdings:
            assert len(holdings.holdings) > 0
            print(
                f"    {len(holdings.holdings)} holdings, total ${holdings.total_value_thousands}K"
            )
            print(f"    Top holding: {holdings.holdings[0].name_of_issuer}")

    if len(f13_filings) >= 2:
        comparison = await run_test(
            "compare_13f_quarters(Berkshire)",
            client.compare_13f_quarters(berkshire_cik, "Berkshire Hathaway"),
        )
        if comparison:
            print(f"    New positions: {len(comparison['new_positions'])}")
            print(f"    Exited positions: {len(comparison['exited_positions'])}")

    # --- Tender Offers ---
    print("\n[Tender Offers]")
    tenders = await run_test(
        "get_tender_offers(AAPL)",
        client.get_tender_offers(ticker, limit=3),
    )
    print(f"    Found {len(tenders) if tenders else 0} tender offer filings")

    # --- Foreign Filings ---
    print("\n[Foreign Filings (TSM)]")
    foreign = await run_test(
        "get_foreign_filings(TSM)",
        client.get_foreign_filings("TSM", limit=3),
    )
    if foreign:
        print(f"    Found {len(foreign)} foreign filings, first: {foreign[0].form_type}")

    # --- Filing Feed ---
    print("\n[Filing Feed]")
    feed = await run_test(
        "get_filing_feed(form_type=8-K)",
        client.get_filing_feed(form_type="8-K", count=5),
    )
    if feed:
        assert len(feed) > 0
        print(f"    {len(feed)} entries, first: {feed[0].title[:60]}...")

    # --- Summary ---
    await client.close()
    await redis.close()

    print(f"\n{'=' * 60}")
    total = passed + len(failures)
    print(f"Results: {passed}/{total} passed")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All smoke tests passed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(smoke_test())

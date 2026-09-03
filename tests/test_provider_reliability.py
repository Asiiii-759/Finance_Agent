from __future__ import annotations

import httpx

from mas_finance.harness import ExecutionPolicy, ToolContext, ToolHarness
from mas_finance.macro import FREDClient, FREDEvidenceAdapter, fred_series_harness_tool
from mas_finance.sec import SECCompanyFactsAdapter, SECCompanyFactsClient, sec_company_facts_harness_tool
from mas_finance.web_search import BochaWebSearchClient, WebSearchEvidenceAdapter, web_search_harness_tool


def _context(run_id: str, capability: str) -> ToolContext:
    return ToolContext(
        run_id=run_id,
        thread_id="provider-reliability",
        policy=ExecutionPolicy(
            allowed_capabilities=frozenset({capability}),
            allow_network=True,
            max_network_calls=2,
        ),
    )


def test_fred_retries_transient_http_status_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        if request.url.path.endswith("/fred/series"):
            return httpx.Response(
                200,
                json={
                    "seriess": [
                        {
                            "id": "UNRATE",
                            "title": "Unemployment Rate",
                            "units": "Percent",
                            "frequency": "Monthly",
                            "last_updated": "2026-08-01",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"observations": [{"date": "2026-08-01", "value": "4.2"}]},
        )

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = FREDClient("test-key", transport=httpx.MockTransport(handler))
    harness.register(fred_series_harness_tool(FREDEvidenceAdapter(client)))
    result = harness.invoke("macro.fred_series", {"series_id": "UNRATE"}, _context("fred-503", "macro.read"))
    assert result.ok
    assert result.attempts == 2
    assert calls == 3


def test_fred_access_denial_is_stable_and_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = FREDClient("test-key", transport=httpx.MockTransport(handler))
    harness.register(fred_series_harness_tool(FREDEvidenceAdapter(client)))
    result = harness.invoke("macro.fred_series", {"series_id": "UNRATE"}, _context("fred-403", "macro.read"))
    assert result.error_code == "provider_access_denied"
    assert result.error_details == {"http_status": 403, "model_action": "report_unavailable"}
    assert result.attempts == 1
    assert calls == 1


def test_sec_retries_transient_http_status_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"ticker": "ACME", "cik_str": 1234}})
        return httpx.Response(
            200,
            json={
                "cik": 1234,
                "entityName": "ACME Corp",
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 120,
                                        "start": "2026-01-01",
                                        "end": "2026-03-31",
                                        "filed": "2026-05-01",
                                        "form": "10-Q",
                                        "accn": "0002",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
        )

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = SECCompanyFactsClient(
        "MAS Finance test test@example.com",
        minimum_request_interval=0,
        transport=httpx.MockTransport(handler),
    )
    harness.register(sec_company_facts_harness_tool(SECCompanyFactsAdapter(client)))
    result = harness.invoke(
        "sec.company_facts",
        {"company": "ACME Corp", "symbol": "ACME"},
        _context("sec-503", "regulatory.read"),
    )
    assert result.ok
    assert result.attempts == 2
    assert calls == 3


def test_sec_access_denial_is_stable_and_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = SECCompanyFactsClient(
        "MAS Finance test test@example.com",
        minimum_request_interval=0,
        transport=httpx.MockTransport(handler),
    )
    harness.register(sec_company_facts_harness_tool(SECCompanyFactsAdapter(client)))
    result = harness.invoke(
        "sec.company_facts",
        {"company": "ACME Corp", "symbol": "ACME"},
        _context("sec-403", "regulatory.read"),
    )
    assert result.error_code == "provider_access_denied"
    assert result.error_details == {"http_status": 403, "model_action": "report_unavailable"}
    assert result.attempts == 1
    assert calls == 1


def test_bocha_retries_transient_http_status_once() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={
                "code": 200,
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "Federal Reserve",
                                "url": "https://www.federalreserve.gov/",
                                "snippet": "Official source.",
                            }
                        ]
                    }
                },
            },
        )

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = BochaWebSearchClient("test-key", transport=httpx.MockTransport(handler))
    harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(client)))
    result = harness.invoke("web.search", {"query": "Federal Reserve"}, _context("bocha-500", "web.search"))
    assert result.ok
    assert result.attempts == 2
    assert calls == 2


def test_bocha_access_denial_is_stable_and_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    harness = ToolHarness(sleeper=lambda _seconds: None)
    client = BochaWebSearchClient("test-key", transport=httpx.MockTransport(handler))
    harness.register(web_search_harness_tool(WebSearchEvidenceAdapter(client)))
    result = harness.invoke("web.search", {"query": "Federal Reserve"}, _context("bocha-403", "web.search"))
    assert result.error_code == "provider_access_denied"
    assert result.error_details == {"http_status": 403, "model_action": "report_unavailable"}
    assert result.attempts == 1
    assert calls == 1

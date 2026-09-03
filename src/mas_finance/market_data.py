from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import httpx

from .rate_limit import RateLimit, RateLimiter


class MarketDataClient:
    def __init__(
        self,
        provider: str = "offline",
        alphavantage_api_key: str | None = None,
        *,
        rate_limiter: RateLimiter | None = None,
        rate_limit: RateLimit | None = None,
    ) -> None:
        self.provider = provider
        self.alphavantage_api_key = alphavantage_api_key
        self.rate_limiter = rate_limiter
        self.rate_limit = rate_limit or RateLimit(6)

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        ticker = _resolve_ticker(company, symbol)
        if self.provider not in {"offline", "disabled", "none"} and self.rate_limiter is not None:
            self.rate_limiter.acquire("market", self.rate_limit, timeout_seconds=30.0)
        if self.provider in {"offline", "disabled", "none"}:
            return self._unavailable_snapshot(company, ticker, ["provider_disabled"])
        if self.provider == "alphavantage":
            if not self.alphavantage_api_key:
                return self._unavailable_snapshot(company, ticker, ["alphavantage_api_key_missing"])
            return self._fetch_from_alphavantage(company, ticker)
        if self.provider == "yahoo":
            return self._fetch_from_yahoo(company, ticker)
        return self._unavailable_snapshot(company, ticker, ["unsupported_provider"])

    def fetch_price_history(
        self,
        company: str,
        symbol: str | None = None,
        *,
        range_name: str = "1y",
        interval: str = "1d",
    ) -> dict[str, Any]:
        ticker = _resolve_ticker(company, symbol)
        if range_name not in {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"}:
            raise ValueError("unsupported market history range")
        if interval not in {"1d", "1wk", "1mo"}:
            raise ValueError("unsupported market history interval")
        if self.provider not in {"offline", "disabled", "none"} and self.rate_limiter is not None:
            self.rate_limiter.acquire("market", self.rate_limit, timeout_seconds=30.0)
        if self.provider in {"offline", "disabled", "none"}:
            return self._unavailable_history(company, ticker, range_name, interval, ["provider_disabled"])
        if self.provider == "alphavantage":
            if not self.alphavantage_api_key:
                return self._unavailable_history(
                    company,
                    ticker,
                    range_name,
                    interval,
                    ["alphavantage_api_key_missing"],
                )
            return self._history_from_alphavantage(company, ticker, range_name, interval)
        if self.provider == "yahoo":
            return self._history_from_yahoo(company, ticker, range_name, interval)
        return self._unavailable_history(company, ticker, range_name, interval, ["unsupported_provider"])

    def _unavailable_snapshot(self, company: str, ticker: str, errors: list[str]) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "symbol": ticker,
            "company": company,
            "status": "unavailable",
            "error_codes": errors,
            "retrieved_at": _utc_now(),
            "as_of": None,
            "current_price": None,
            "monthly_return": None,
            "market_cap": None,
            "trailing_pe": None,
            "price_to_book": None,
            "price_to_sales": None,
            "enterprise_to_ebitda": None,
            "currency": None,
            "sector": None,
            "industry": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
        }

    def _unavailable_history(
        self,
        company: str,
        ticker: str,
        range_name: str,
        interval: str,
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "symbol": ticker,
            "company": company,
            "status": "unavailable",
            "error_codes": errors,
            "retrieved_at": _utc_now(),
            "as_of": None,
            "currency": None,
            "range": range_name,
            "interval": interval,
            "price_basis": None,
            "points": [],
        }

    def _fetch_from_yahoo(self, company: str, ticker: str) -> dict[str, Any]:
        with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as client:
            quote_resp = client.get("https://query1.finance.yahoo.com/v7/finance/quote", params={"symbols": ticker})
            quote_resp.raise_for_status()
            quote_data = quote_resp.json()["quoteResponse"]["result"]
            chart_resp = client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"range": "1mo", "interval": "1d"},
            )
            chart_resp.raise_for_status()
            chart_data = chart_resp.json()["chart"]["result"][0]

        quote = quote_data[0] if quote_data else {}
        raw_closes = chart_data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        timestamps = chart_data.get("timestamp", [])
        close_points = [
            (int(timestamp), float(value))
            for timestamp, value in zip(timestamps, raw_closes, strict=False)
            if value is not None
        ]
        closes = [value for _, value in close_points]
        latest_close = round(closes[-1], 4) if closes else None
        monthly_change = round((closes[-1] / closes[0]) - 1, 4) if len(closes) >= 2 else None
        return {
            "provider": "yahoo",
            "symbol": ticker,
            "company": company,
            "status": "ok",
            "retrieved_at": _utc_now(),
            "as_of": datetime.fromtimestamp(close_points[-1][0], tz=UTC).isoformat() if close_points else None,
            "exchange": quote.get("fullExchangeName") or quote.get("exchange"),
            "current_price": latest_close,
            "monthly_return": monthly_change,
            "market_cap": quote.get("marketCap"),
            "trailing_pe": quote.get("trailingPE"),
            "price_to_book": quote.get("priceToBook"),
            "price_to_sales": quote.get("priceToSalesTrailing12Months"),
            "enterprise_to_ebitda": quote.get("enterpriseToEbitda"),
            "currency": quote.get("currency"),
            "sector": quote.get("sector"),
            "industry": quote.get("industry"),
            "fifty_two_week_high": quote.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": quote.get("fiftyTwoWeekLow"),
        }

    def _history_from_yahoo(
        self,
        company: str,
        ticker: str,
        range_name: str,
        interval: str,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as client:
            response = client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"range": range_name, "interval": interval, "events": "div,splits"},
            )
            response.raise_for_status()
            chart = response.json()["chart"]["result"][0]
        timestamps = chart.get("timestamp") or []
        indicators = chart.get("indicators") or {}
        adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []
        closes = adjusted or (indicators.get("quote") or [{}])[0].get("close") or []
        price_basis = "adjusted_close" if adjusted else "close"
        points: list[dict[str, Any]] = [
            {
                "date": datetime.fromtimestamp(int(timestamp), tz=UTC).date().isoformat(),
                "close": float(value),
            }
            for timestamp, value in zip(timestamps, closes, strict=False)
            if value is not None
        ]
        metadata = chart.get("meta") or {}
        return {
            "provider": "yahoo",
            "symbol": ticker,
            "company": company,
            "status": "ok" if points else "unavailable",
            "retrieved_at": _utc_now(),
            "as_of": points[-1]["date"] if points else None,
            "currency": metadata.get("currency"),
            "exchange": metadata.get("exchangeName"),
            "range": range_name,
            "interval": interval,
            "price_basis": price_basis,
            "points": points,
        }

    def _fetch_from_alphavantage(self, company: str, ticker: str) -> dict[str, Any]:
        overview_url = "https://www.alphavantage.co/query"
        with httpx.Client(timeout=30) as client:
            overview = client.get(
                overview_url,
                params={"function": "OVERVIEW", "symbol": ticker, "apikey": self.alphavantage_api_key},
            ).json()
            quote = client.get(
                overview_url,
                params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": self.alphavantage_api_key},
            ).json()
        global_quote = quote.get("Global Quote", {})
        return {
            "provider": "alphavantage",
            "symbol": ticker,
            "company": company,
            "status": "ok" if global_quote else "unavailable",
            "retrieved_at": _utc_now(),
            "as_of": global_quote.get("07. latest trading day"),
            "exchange": overview.get("Exchange"),
            "current_price": _safe_float(global_quote.get("05. price")),
            "monthly_return": None,
            "market_cap": _safe_float(overview.get("MarketCapitalization")),
            "trailing_pe": _safe_float(overview.get("PERatio")),
            "price_to_book": _safe_float(overview.get("PriceToBookRatio")),
            "price_to_sales": _safe_float(overview.get("PriceToSalesRatioTTM")),
            "enterprise_to_ebitda": _safe_float(overview.get("EVToEBITDA")),
            "currency": overview.get("Currency"),
            "sector": overview.get("Sector"),
            "industry": overview.get("Industry"),
            "fifty_two_week_high": _safe_float(overview.get("52WeekHigh")),
            "fifty_two_week_low": _safe_float(overview.get("52WeekLow")),
        }

    def _history_from_alphavantage(
        self,
        company: str,
        ticker: str,
        range_name: str,
        interval: str,
    ) -> dict[str, Any]:
        if interval != "1d":
            raise ValueError("AlphaVantage history currently supports daily intervals only")
        with httpx.Client(timeout=30) as client:
            response = client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "outputsize": "full" if range_name in {"2y", "5y", "10y"} else "compact",
                    "apikey": self.alphavantage_api_key,
                },
            )
            response.raise_for_status()
            payload = response.json()
        series = payload.get("Time Series (Daily)") or {}
        points = [
            {"date": str(date), "close": float(values["4. close"])}
            for date, values in series.items()
            if isinstance(values, dict) and _safe_float(values.get("4. close")) is not None
        ]
        points.sort(key=lambda item: str(item["date"]))
        limits = {"1mo": 23, "3mo": 66, "6mo": 132, "1y": 264, "2y": 528, "5y": 1320, "10y": 2640}
        points = points[-limits[range_name] :]
        metadata = payload.get("Meta Data") or {}
        return {
            "provider": "alphavantage",
            "symbol": ticker,
            "company": company,
            "status": "ok" if points else "unavailable",
            "retrieved_at": _utc_now(),
            "as_of": points[-1]["date"] if points else None,
            "currency": None,
            "exchange": None,
            "range": range_name,
            "interval": interval,
            "price_basis": "close",
            "points": points,
            "provider_metadata": {"last_refreshed": metadata.get("3. Last Refreshed")},
        }


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_ticker(company: str, symbol: str | None) -> str:
    del company
    candidate = (symbol or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9.^=_:-]{1,64}", candidate):
        raise ValueError("a valid market symbol is required for this company")
    return candidate


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

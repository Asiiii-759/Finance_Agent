"""AllTick / 必盈行情 MCP server：stdio JSON-RPC，返回 canonical EvidenceBundle。"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from mas_finance.market import MarketEvidenceAdapter, MarketHistoryEvidenceAdapter
from mas_finance.rate_limit import RateLimit, RateLimiter

_ALLTICK_TICK = "https://quote.alltick.co/quote-stock-b-api/trade-tick"
_ALLTICK_KLINE = "https://quote.alltick.co/quote-stock-b-api/kline"
_BIYING_REALTIME = "https://api.biyingapi.com/hsstock/real/time/{code}/{licence}"
_BIYING_HISTORY = "https://api.biyingapi.com/hsstock/history/{code}/{period}/{licence}"
_KLINE_TYPE = {"1d": 8, "1wk": 9, "1mo": 10}
_BIYING_PERIOD = {"1d": "d", "1wk": "w", "1mo": "n"}
_RANGE_BARS = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252, "2y": 500, "5y": 500, "10y": 500}
_ASHARE_RE = re.compile(r"^(?:\d{6}(?:\.(?:SZ|SH|SS|BJ))?)$", re.IGNORECASE)
_limiter = RateLimiter()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def is_ashare_symbol(symbol: str) -> bool:
    return bool(_ASHARE_RE.fullmatch(symbol.strip()))


def alltick_code(symbol: str) -> str:
    raw = symbol.strip().upper()
    if "." in raw:
        return raw
    return f"{raw}.US"


def biying_code(symbol: str) -> str:
    raw = symbol.strip().upper()
    if re.fullmatch(r"\d{6}", raw):
        return f"{raw}.SZ"
    if raw.endswith(".SS"):
        return raw[:-3] + ".SH"
    return raw


def parse_alltick_tick(payload: Mapping[str, Any], *, company: str, symbol: str) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    ticks = data.get("tick_list") if isinstance(data, Mapping) else None
    tick: Mapping[str, Any] = {}
    if isinstance(ticks, list) and ticks and isinstance(ticks[0], Mapping):
        tick = ticks[0]
    elif isinstance(data, Mapping):
        tick = data
    price = _optional_float(tick.get("price") or tick.get("last_price") or tick.get("px"))
    as_of = _optional_text(tick.get("tick_time") or tick.get("time") or payload.get("as_of"))
    return {
        "provider": "alltick",
        "symbol": symbol,
        "company": company,
        "status": "ok" if price is not None else "unavailable",
        "error_codes": [] if price is not None else ["alltick_price_missing"],
        "retrieved_at": _utc_now(),
        "as_of": as_of or _utc_now(),
        "current_price": price,
        "monthly_return": None,
        "market_cap": None,
        "trailing_pe": None,
        "price_to_book": None,
        "price_to_sales": None,
        "enterprise_to_ebitda": None,
        "currency": "USD",
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
    }


def parse_alltick_kline(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    rows = data.get("kline_list") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        rows = data.get("kline") if isinstance(data, Mapping) else []
    points: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return points
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        close = _optional_float(item.get("close_price") or item.get("close"))
        if close is None:
            continue
        timestamp = item.get("timestamp") or item.get("time")
        point_date = _timestamp_to_date(timestamp)
        if point_date is None:
            continue
        points.append(
            {
                "date": point_date,
                "open": _optional_float(item.get("open_price") or item.get("open")) or close,
                "high": _optional_float(item.get("high_price") or item.get("high")) or close,
                "low": _optional_float(item.get("low_price") or item.get("low")) or close,
                "close": close,
                "volume": _optional_float(item.get("volume")) or 0.0,
            }
        )
    points.sort(key=lambda item: str(item["date"]))
    return points


def parse_biying_realtime(payload: Any, *, company: str, symbol: str) -> dict[str, Any]:
    row: Mapping[str, Any]
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        row = payload[0]
    elif isinstance(payload, Mapping):
        nested = payload.get("data")
        if isinstance(nested, list) and nested and isinstance(nested[0], Mapping):
            row = nested[0]
        elif isinstance(nested, Mapping):
            row = nested
        else:
            row = payload
    else:
        row = {}
    price = _optional_float(row.get("price") or row.get("p") or row.get("now") or row.get("current"))
    return {
        "provider": "biying",
        "symbol": symbol,
        "company": company,
        "status": "ok" if price is not None else "unavailable",
        "error_codes": [] if price is not None else ["biying_price_missing"],
        "retrieved_at": _utc_now(),
        "as_of": _optional_text(row.get("time") or row.get("date")) or _utc_now(),
        "current_price": price,
        "monthly_return": None,
        "market_cap": _optional_float(row.get("sz") or row.get("market_cap")),
        "trailing_pe": _optional_float(row.get("pe") or row.get("trailing_pe")),
        "price_to_book": _optional_float(row.get("pb") or row.get("price_to_book")),
        "price_to_sales": None,
        "enterprise_to_ebitda": None,
        "currency": "CNY",
        "fifty_two_week_high": _optional_float(row.get("high") or row.get("h")),
        "fifty_two_week_low": _optional_float(row.get("low") or row.get("l")),
    }


def parse_biying_history(payload: Any) -> list[dict[str, Any]]:
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        nested = payload.get("data") or payload.get("list") or payload.get("history")
        rows = nested if isinstance(nested, list) else []
    else:
        rows = []
    points: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        close = _optional_float(item.get("close") or item.get("c") or item.get("spj"))
        point_date = _optional_text(item.get("date") or item.get("t") or item.get("time"))
        if close is None or not point_date:
            continue
        points.append(
            {
                "date": point_date[:10],
                "open": _optional_float(item.get("open") or item.get("o") or item.get("kpj")) or close,
                "high": _optional_float(item.get("high") or item.get("h") or item.get("zgj")) or close,
                "low": _optional_float(item.get("low") or item.get("l") or item.get("zdj")) or close,
                "close": close,
                "volume": _optional_float(item.get("volume") or item.get("v") or item.get("cjl")) or 0.0,
            }
        )
    points.sort(key=lambda item: str(item["date"]))
    return points


class ExternalMarketClient:
    """按代码路由 AllTick / 必盈；可选 yfinance、akshare 仅作显式 fallback。"""

    def __init__(
        self,
        *,
        alltick_token: str | None = None,
        biying_licence: str | None = None,
        enable_yfinance: bool = False,
        enable_akshare: bool = False,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: RateLimiter | None = None,
        max_calls_per_minute: int = 6,
    ) -> None:
        self.alltick_token = (alltick_token or "").strip() or None
        self.biying_licence = (biying_licence or "").strip() or None
        self.enable_yfinance = enable_yfinance
        self.enable_akshare = enable_akshare
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.rate_limiter = rate_limiter or _limiter
        self.rate_limit = RateLimit(max_calls_per_minute, 60.0)

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> Mapping[str, Any]:
        ticker = (symbol or company).strip()
        self._acquire(ticker)
        try:
            if is_ashare_symbol(ticker) and self.biying_licence:
                return self._biying_snapshot(company, ticker)
            if self.alltick_token:
                return self._alltick_snapshot(company, ticker)
            return self._fallback_snapshot(company, ticker, ["primary_provider_missing"])
        except Exception as exc:
            return self._fallback_snapshot(company, ticker, [type(exc).__name__])

    def fetch_price_history(
        self,
        company: str,
        symbol: str | None = None,
        *,
        range_name: str = "1y",
        interval: str = "1d",
    ) -> Mapping[str, Any]:
        ticker = (symbol or company).strip()
        self._acquire(ticker)
        try:
            if is_ashare_symbol(ticker) and self.biying_licence:
                return self._biying_history(company, ticker, range_name, interval)
            if self.alltick_token:
                return self._alltick_history(company, ticker, range_name, interval)
            return self._fallback_history(company, ticker, range_name, interval, ["primary_provider_missing"])
        except Exception as exc:
            return self._fallback_history(company, ticker, range_name, interval, [type(exc).__name__])

    def _acquire(self, ticker: str) -> None:
        key = "biying" if is_ashare_symbol(ticker) else "alltick"
        self.rate_limiter.acquire(key, self.rate_limit, timeout_seconds=self.timeout_seconds)

    def _alltick_snapshot(self, company: str, ticker: str) -> dict[str, Any]:
        if not self.alltick_token:
            raise RuntimeError("ALLTICK_TOKEN is missing")
        query = {"trace": uuid4().hex, "data": {"symbol_list": [{"code": alltick_code(ticker)}]}}
        payload = self._get_json(_ALLTICK_TICK, {"token": self.alltick_token, "query": json.dumps(query)})
        snapshot = parse_alltick_tick(payload, company=company, symbol=alltick_code(ticker))
        if snapshot["status"] == "ok":
            return snapshot
        return self._fallback_snapshot(company, ticker, list(snapshot.get("error_codes") or ["alltick_empty"]))

    def _alltick_history(self, company: str, ticker: str, range_name: str, interval: str) -> dict[str, Any]:
        if not self.alltick_token:
            raise RuntimeError("ALLTICK_TOKEN is missing")
        query = {
            "trace": uuid4().hex,
            "data": {
                "code": alltick_code(ticker),
                "kline_type": _KLINE_TYPE.get(interval, 8),
                "kline_timestamp_end": 0,
                "query_kline_num": _RANGE_BARS.get(range_name, 252),
                "adjust_type": 0,
            },
        }
        payload = self._get_json(_ALLTICK_KLINE, {"token": self.alltick_token, "query": json.dumps(query)})
        points = parse_alltick_kline(payload)
        if len(points) >= 3:
            return {
                "provider": "alltick",
                "symbol": alltick_code(ticker),
                "company": company,
                "status": "ok",
                "retrieved_at": _utc_now(),
                "as_of": points[-1]["date"],
                "currency": "USD",
                "range": range_name,
                "interval": interval,
                "price_basis": "close",
                "points": points,
            }
        return self._fallback_history(company, ticker, range_name, interval, ["alltick_history_empty"])

    def _biying_snapshot(self, company: str, ticker: str) -> dict[str, Any]:
        if not self.biying_licence:
            raise RuntimeError("BIYING_LICENCE is missing")
        url = _BIYING_REALTIME.format(code=biying_code(ticker), licence=self.biying_licence)
        payload = self._get_json(url, {})
        snapshot = parse_biying_realtime(payload, company=company, symbol=biying_code(ticker))
        if snapshot["status"] == "ok":
            return snapshot
        return self._fallback_snapshot(company, ticker, list(snapshot.get("error_codes") or ["biying_empty"]))

    def _biying_history(self, company: str, ticker: str, range_name: str, interval: str) -> dict[str, Any]:
        if not self.biying_licence:
            raise RuntimeError("BIYING_LICENCE is missing")
        period = _BIYING_PERIOD.get(interval, "d")
        url = _BIYING_HISTORY.format(code=biying_code(ticker), period=period, licence=self.biying_licence)
        payload = self._get_json(url, {})
        points = parse_biying_history(payload)[-_RANGE_BARS.get(range_name, 252) :]
        if len(points) >= 3:
            return {
                "provider": "biying",
                "symbol": biying_code(ticker),
                "company": company,
                "status": "ok",
                "retrieved_at": _utc_now(),
                "as_of": points[-1]["date"],
                "currency": "CNY",
                "range": range_name,
                "interval": interval,
                "price_basis": "close",
                "points": points,
            }
        return self._fallback_history(company, ticker, range_name, interval, ["biying_history_empty"])

    def _fallback_snapshot(self, company: str, ticker: str, errors: list[str]) -> dict[str, Any]:
        if self.enable_yfinance and not is_ashare_symbol(ticker):
            try:
                return _yfinance_snapshot(company, ticker)
            except Exception as exc:
                errors.append(type(exc).__name__)
        if self.enable_akshare and is_ashare_symbol(ticker):
            try:
                return _akshare_snapshot(company, ticker)
            except Exception as exc:
                errors.append(type(exc).__name__)
        return {
            "provider": "extmarket",
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
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
        }

    def _fallback_history(
        self,
        company: str,
        ticker: str,
        range_name: str,
        interval: str,
        errors: list[str],
    ) -> dict[str, Any]:
        if self.enable_yfinance and not is_ashare_symbol(ticker):
            try:
                return _yfinance_history(company, ticker, range_name, interval)
            except Exception as exc:
                errors.append(type(exc).__name__)
        if self.enable_akshare and is_ashare_symbol(ticker):
            try:
                return _akshare_history(company, ticker, range_name, interval)
            except Exception as exc:
                errors.append(type(exc).__name__)
        return {
            "provider": "extmarket",
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

    def _get_json(self, url: str, params: Mapping[str, Any]) -> Any:
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            response = client.get(url, params=dict(params) if params else None)
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f"market provider transient HTTP status: {response.status_code}")
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("market provider response exceeds the byte limit")
        return response.json()


def snapshot_bundle(client: ExternalMarketClient, arguments: Mapping[str, Any]) -> dict[str, Any]:
    raw_required = arguments.get("required_fields") or []
    if raw_required is None:
        raw_required = []
    if not isinstance(raw_required, list):
        raise ValueError("required_fields must be a list")
    return MarketEvidenceAdapter(client).fetch(
        company=str(arguments.get("company") or ""),
        symbol=_optional_text(arguments.get("symbol")),
        required_fields=tuple(str(item) for item in raw_required),
    ).to_dict()


def history_bundle(client: ExternalMarketClient, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return MarketHistoryEvidenceAdapter(client).fetch(
        company=str(arguments.get("company") or ""),
        symbol=_optional_text(arguments.get("symbol")),
        range_name=str(arguments.get("range") or "1y"),
        interval=str(arguments.get("interval") or "1d"),
    ).to_dict()


def _yfinance_snapshot(company: str, ticker: str) -> dict[str, Any]:
    yf = importlib.import_module("yfinance")

    info = yf.Ticker(alltick_code(ticker).split(".", 1)[0]).fast_info
    price = _optional_float(getattr(info, "last_price", None) or getattr(info, "lastPrice", None))
    return {
        "provider": "yfinance",
        "symbol": ticker,
        "company": company,
        "status": "ok" if price is not None else "unavailable",
        "error_codes": [] if price is not None else ["yfinance_price_missing"],
        "retrieved_at": _utc_now(),
        "as_of": _utc_now(),
        "current_price": price,
        "monthly_return": None,
        "market_cap": _optional_float(getattr(info, "market_cap", None) or getattr(info, "marketCap", None)),
        "trailing_pe": None,
        "price_to_book": None,
        "price_to_sales": None,
        "enterprise_to_ebitda": None,
        "currency": str(getattr(info, "currency", None) or "USD"),
        "fifty_two_week_high": _optional_float(getattr(info, "year_high", None) or getattr(info, "yearHigh", None)),
        "fifty_two_week_low": _optional_float(getattr(info, "year_low", None) or getattr(info, "yearLow", None)),
    }


def _yfinance_history(company: str, ticker: str, range_name: str, interval: str) -> dict[str, Any]:
    yf = importlib.import_module("yfinance")

    period = {"1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y"}[range_name]
    yf_interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}[interval]
    frame = yf.Ticker(alltick_code(ticker).split(".", 1)[0]).history(period=period, interval=yf_interval)
    points: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        close = _optional_float(row.get("Close"))
        if close is None:
            continue
        points.append(
            {
                "date": str(index)[:10],
                "open": _optional_float(row.get("Open")) or close,
                "high": _optional_float(row.get("High")) or close,
                "low": _optional_float(row.get("Low")) or close,
                "close": close,
                "volume": _optional_float(row.get("Volume")) or 0.0,
            }
        )
    return {
        "provider": "yfinance",
        "symbol": ticker,
        "company": company,
        "status": "ok" if len(points) >= 3 else "unavailable",
        "retrieved_at": _utc_now(),
        "as_of": points[-1]["date"] if points else None,
        "currency": "USD",
        "range": range_name,
        "interval": interval,
        "price_basis": "close",
        "points": points,
        "error_codes": [] if len(points) >= 3 else ["yfinance_history_empty"],
    }


def _akshare_snapshot(company: str, ticker: str) -> dict[str, Any]:
    ak = importlib.import_module("akshare")

    code = re.sub(r"\.(?:SZ|SH|SS|BJ)$", "", ticker.strip(), flags=re.IGNORECASE)
    info = ak.stock_individual_info_em(symbol=code)
    values = {str(row.iloc[0]): row.iloc[1] for _, row in info.iterrows()} if hasattr(info, "iterrows") else {}
    price = _optional_float(values.get("最新") or values.get("最新价"))
    return {
        "provider": "akshare",
        "symbol": biying_code(ticker),
        "company": company,
        "status": "ok" if price is not None else "unavailable",
        "error_codes": [] if price is not None else ["akshare_price_missing"],
        "retrieved_at": _utc_now(),
        "as_of": _utc_now(),
        "current_price": price,
        "monthly_return": None,
        "market_cap": _optional_float(values.get("总市值")),
        "trailing_pe": None,
        "price_to_book": None,
        "price_to_sales": None,
        "enterprise_to_ebitda": None,
        "currency": "CNY",
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
    }


def _akshare_history(company: str, ticker: str, range_name: str, interval: str) -> dict[str, Any]:
    ak = importlib.import_module("akshare")

    code = re.sub(r"\.(?:SZ|SH|SS|BJ)$", "", ticker.strip(), flags=re.IGNORECASE)
    period = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}[interval]
    frame = ak.stock_zh_a_hist(symbol=code, period=period, adjust="")
    points: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        close = _optional_float(row.get("收盘") or row.get("close"))
        point_date = str(row.get("日期") or row.get("date") or "")[:10]
        if close is None or not point_date:
            continue
        points.append(
            {
                "date": point_date,
                "open": _optional_float(row.get("开盘") or row.get("open")) or close,
                "high": _optional_float(row.get("最高") or row.get("high")) or close,
                "low": _optional_float(row.get("最低") or row.get("low")) or close,
                "close": close,
                "volume": _optional_float(row.get("成交量") or row.get("volume")) or 0.0,
            }
        )
    points = points[-_RANGE_BARS.get(range_name, 252) :]
    return {
        "provider": "akshare",
        "symbol": biying_code(ticker),
        "company": company,
        "status": "ok" if len(points) >= 3 else "unavailable",
        "retrieved_at": _utc_now(),
        "as_of": points[-1]["date"] if points else None,
        "currency": "CNY",
        "range": range_name,
        "interval": interval,
        "price_basis": "close",
        "points": points,
        "error_codes": [] if len(points) >= 3 else ["akshare_history_empty"],
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _timestamp_to_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        text = str(value).strip()
        return text[:10] if text else None
    if number > 10_000_000_000:
        number //= 1000
    return datetime.fromtimestamp(number, tz=UTC).date().isoformat()


def _read_message() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    payload = json.loads(line.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stdio MCP message must be an object")
    return payload


def _write_message(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _tools() -> list[dict[str, Any]]:
    snapshot_schema = {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "公司或资产的规范展示名称，例如 贵州茅台、Apple。不要在此字段传供应商代码。",
                "examples": ["贵州茅台", "Apple"],
            },
            "symbol": {
                "type": "string",
                "description": "可选证券代码。A 股使用 600519.SH/000001.SZ，美股使用 AAPL；不确定时省略。",
                "examples": ["600519.SH", "AAPL"],
            },
            "required_fields": {
                "type": "array",
                "description": "本次必须返回的规范字段名列表。",
                "items": {"type": "string"},
                "examples": [["price", "market_cap"]],
            },
        },
        "required": ["company"],
        "additionalProperties": False,
    }
    history_schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "公司或资产的规范展示名称。"},
            "symbol": {"type": "string", "description": "可选证券代码；不确定时省略。"},
            "range": {
                "type": "string",
                "description": "历史区间。",
                "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
                "default": "1y",
            },
            "interval": {
                "type": "string",
                "description": "K 线周期。",
                "enum": ["1d", "1wk", "1mo"],
                "default": "1d",
            },
        },
        "required": ["company"],
        "additionalProperties": False,
    }
    return [
        {
            "name": "snapshot",
            "description": "通过 AllTick 或必盈读取实时行情。传规范公司名；只有确定证券代码时才传 symbol。",
            "inputSchema": snapshot_schema,
            "annotations": {"readOnlyHint": True},
            "_meta": {
                "mas_finance": {
                    "capability": "market.read",
                    "side_effect": "read_only",
                    "planner_category": "market",
                }
            },
        },
        {
            "name": "history",
            "description": "通过 AllTick 或必盈读取历史行情。range 和 interval 必须使用契约枚举值。",
            "inputSchema": history_schema,
            "annotations": {"readOnlyHint": True},
            "_meta": {
                "mas_finance": {
                    "capability": "market.read",
                    "side_effect": "read_only",
                    "planner_category": "market_history",
                }
            },
        },
    ]


def _client_from_env() -> ExternalMarketClient:
    raw_limit = os.getenv("MAS_MARKET_MAX_CALLS_PER_MINUTE", "6")
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 6
    return ExternalMarketClient(
        alltick_token=os.getenv("ALLTICK_TOKEN"),
        biying_licence=os.getenv("BIYING_LICENCE"),
        enable_yfinance=os.getenv("MAS_ENABLE_YFINANCE", "").strip().lower() in {"1", "true", "yes"},
        enable_akshare=os.getenv("MAS_ENABLE_AKSHARE", "").strip().lower() in {"1", "true", "yes"},
        max_calls_per_minute=max(1, min(limit, 60)),
    )


def main() -> None:
    client = _client_from_env()
    while True:
        message = _read_message()
        if message is None:
            return
        method = str(message.get("method") or "")
        call_id = message.get("id")
        if method == "initialize":
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mas-finance-extmarket", "version": "1"},
                    },
                }
            )
            continue
        if method == "notifications/initialized" or call_id is None:
            continue
        if method == "tools/list":
            _write_message({"jsonrpc": "2.0", "id": call_id, "result": {"tools": _tools()}})
            continue
        if method == "tools/call":
            raw_params = message.get("params")
            params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}
            name = str(params.get("name") or "")
            raw_arguments = params.get("arguments")
            arguments: Mapping[str, Any] = raw_arguments if isinstance(raw_arguments, Mapping) else {}
            try:
                if name == "snapshot":
                    payload = snapshot_bundle(client, arguments)
                elif name == "history":
                    payload = history_bundle(client, arguments)
                else:
                    raise ValueError(f"unknown tool: {name}")
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "result": {"structuredContent": payload, "content": [], "isError": False},
                    }
                )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError, httpx.HTTPError) as exc:
                if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)):
                    error_code = "provider_transient_error"
                    retryable = True
                    hint = "保持实体不变，稍后重试；不要通过猜测其他代码规避网络错误。"
                elif "missing" in str(exc).casefold():
                    error_code = "provider_configuration_error"
                    retryable = False
                    hint = "服务端缺少供应商凭据，停止修改参数并报告配置问题。"
                else:
                    error_code = "invalid_or_unresolved_arguments"
                    retryable = True
                    hint = "检查 symbol 的市场后缀和 range/interval 枚举；不确定 symbol 时省略该字段。"
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "result": {
                            "isError": True,
                            "structuredContent": {
                                "error_code": error_code,
                                "message": str(exc)[:1_000],
                                "retryable": retryable,
                                "received_arguments": dict(arguments),
                                "suggested_action": hint,
                            },
                            "content": [{"type": "text", "text": str(exc)[:1_000]}],
                        },
                    }
                )
            continue
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": call_id,
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }
        )


if __name__ == "__main__":
    main()

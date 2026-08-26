from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import AppConfig
from .service import FinanceAnalysisService


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the evidence-first financial research agent.")
    parser.add_argument("--query", required=True, help="Financial research question.")
    parser.add_argument("--entity", action="append", default=[], help="Company/issuer; repeat as needed.")
    parser.add_argument("--symbol", action="append", default=[], help="ENTITY=TICKER; repeat as needed.")
    parser.add_argument("--pdf", action="append", default=[], help="PDF document; repeat as needed.")
    parser.add_argument("--thread-id", default="demo-thread", help="Conversation thread identifier.")
    parser.add_argument("--output-dir", help="Artifact output directory.")
    parser.add_argument("--allow-network", action="store_true", help="Request server-authorized network tools.")
    parser.add_argument(
        "--macro-series",
        action="append",
        default=[],
        help="FRED series id; repeat as needed.",
    )
    parser.add_argument(
        "--calculate",
        action="append",
        default=[],
        help=(
            'Calculation JSON, e.g. {"operation":"cagr","inputs":{"beginning_value":100,"ending_value":150,"years":3}}'
        ),
    )
    parser.add_argument("--require-market", action="store_true", help="Force current market evidence.")
    parser.add_argument("--require-market-history", action="store_true", help="Force historical price/risk evidence.")
    parser.add_argument("--require-regulatory", action="store_true", help="Force SEC fundamental evidence.")
    parser.add_argument(
        "--market-range",
        default="1y",
        choices=("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"),
    )
    parser.add_argument("--market-interval", default="1d", choices=("1d", "1wk", "1mo"))
    parser.add_argument("--json", action="store_true", help="Print full state as JSON.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))
    response = FinanceAnalysisService(config).analyze(
        args.query,
        thread_id=args.thread_id,
        document_paths=args.pdf,
        entities=args.entity,
        symbols=_parse_symbols(args.symbol),
        allow_network=args.allow_network,
        macro_series=args.macro_series,
        calculations=_parse_calculations(args.calculate),
        require_market_data=True if args.require_market else None,
        require_market_history=True if args.require_market_history else None,
        require_regulatory_data=True if args.require_regulatory else None,
        market_history_range=args.market_range,
        market_history_interval=args.market_interval,
    )
    result = response["result"]
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(result["report"])
        print(f"\n[Status] {result['status']}")
        print(f"[Synthesis] backend={response['llm_backend']}")
        for name, path in response["artifacts"].items():
            print(f"[Artifact] {name}={path}")


def _parse_symbols(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --symbol {value!r}; expected ENTITY=TICKER")
        entity, symbol = (item.strip() for item in value.split("=", 1))
        if not entity or not symbol:
            raise SystemExit(f"invalid --symbol {value!r}; expected ENTITY=TICKER")
        result[entity] = symbol
    return result


def _parse_calculations(values: list[str]) -> list[dict]:
    result: list[dict] = []
    for raw in values:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --calculate JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit("--calculate must be a JSON object")
        result.append(value)
    return result

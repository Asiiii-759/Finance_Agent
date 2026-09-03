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
    parser.add_argument("--pdf", action="append", default=[], help="PDF document; repeat as needed.")
    parser.add_argument("--thread-id", default="demo-thread", help="Conversation thread identifier.")
    parser.add_argument("--output-dir", help="Artifact output directory.")
    parser.add_argument("--allow-network", action="store_true", help="Request server-authorized network tools.")
    parser.add_argument("--json", action="store_true", help="Print full state as JSON.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))
    response = FinanceAnalysisService(config).analyze(
        args.query,
        thread_id=args.thread_id,
        document_paths=args.pdf,
        allow_network=args.allow_network,
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

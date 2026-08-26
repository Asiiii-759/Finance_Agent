from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .security import safe_child, safe_identifier


def export_run_artifacts(result: dict[str, Any], output_dir: Path, thread_id: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    base_name = f"{safe_identifier(thread_id, fallback='thread')}_{timestamp}_{uuid4().hex[:8]}"

    report_path = safe_child(output_dir, f"{base_name}_report.md")
    audit_path = safe_child(output_dir, f"{base_name}_audit.json")
    state_path = safe_child(output_dir, f"{base_name}_state.json")

    report_path.write_text(result["report"], encoding="utf-8")
    audit_path.write_text(
        json.dumps(result.get("audit_events", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "report_path": str(report_path),
        "audit_path": str(audit_path),
        "state_path": str(state_path),
    }

from __future__ import annotations

import json
import os
import queue as queue_module
import socket
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import AppConfig
from .queueing import ReliableJobQueue
from .security import safe_child
from .service import FinanceAnalysisService


def execute_analysis_job(
    job_id: str,
    query: str,
    thread_id: str,
    export_artifacts: bool = True,
    document_paths: list[str] | None = None,
    cleanup_documents: bool = False,
) -> None:
    config = AppConfig.from_env()
    service = FinanceAnalysisService(config)
    try:
        submitted = service.submit_job(
            query=query,
            thread_id=thread_id,
            export_artifacts=export_artifacts,
            document_paths=document_paths or [],
            cleanup_documents=cleanup_documents,
            idempotency_key=job_id,
            tenant_id=config.local_tenant_id,
            user_id=config.local_user_id,
        )
        service.process_queued_job(f"direct-worker-{os.getpid()}", job_id=submitted["job_id"])
    finally:
        service.close()


def _run_claimed_job(
    config: AppConfig,
    payload: dict[str, Any],
    result_queue,
    llm_client,
    pdf_document_parser,
    pdf_parser_network_access,
    embedding_provider,
    conversation_summarizer,
    conversation_token_counter,
) -> None:
    service = FinanceAnalysisService(
        config,
        llm_client=llm_client,
        pdf_document_parser=pdf_document_parser,
        pdf_parser_network_access=pdf_parser_network_access,
        embedding_provider=embedding_provider,
        conversation_summarizer=conversation_summarizer,
        conversation_token_counter=conversation_token_counter,
    )
    try:
        service.run_job(
            payload["job_id"],
            payload["query"],
            payload["thread_id"],
            payload.get("export_artifacts", True),
            payload.get("document_paths") or [],
            tenant_id=payload["tenant_id"],
            user_id=payload["user_id"],
            allow_network=payload.get("allow_network", False),
            use_session_documents=payload.get("use_session_documents", False),
            use_personal_memory=payload.get("use_personal_memory", True),
            use_personal_knowledge=payload.get("use_personal_knowledge", True),
            retain_documents_for_session=payload.get("retain_documents_for_session", False),
            resume=payload["attempt_count"] > 1,
            finalize_status=False,
        )
        session_documents_path: str | None = None
        if payload.get("retain_documents_for_session"):
            documents = service._load_session_documents(
                payload["tenant_id"],
                payload["user_id"],
                payload["thread_id"],
            )
            if not documents:
                raise RuntimeError("retained session documents disappeared before transfer")
            transfer_path = safe_child(
                config.upload_dir,
                f"session-{payload['job_id']}-{uuid4().hex[:8]}.json",
            )
            transfer_path.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
            session_documents_path = str(transfer_path)
    except Exception as exc:
        result_queue.put({"error_type": type(exc).__name__})
    else:
        result_queue.put(
            {
                "completed": True,
                "session_documents_path": session_documents_path,
            }
        )
    finally:
        service.close()


def process_one_isolated_job(
    service: FinanceAnalysisService,
    worker_id: str,
    *,
    job_id: str | None = None,
) -> bool:
    queue = ReliableJobQueue(
        service.repository,
        lease_seconds=service.config.job_lease_seconds,
        retry_delay_seconds=service.config.job_retry_delay_seconds,
    )
    payload = queue.claim(worker_id, job_id=job_id)
    if payload is None:
        return False
    context = get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_claimed_job,
        args=(
            service.config,
            payload,
            result_queue,
            service.llm_client,
            service.pdf_document_parser,
            service.pdf_parser_network_access,
            service.embedding_provider,
            service.conversation_summarizer,
            service.conversation_token_counter,
        ),
        name=f"analysis-{payload['job_id']}",
    )
    process.start()
    interval = max(1, service.config.job_lease_seconds // 3)
    while process.is_alive():
        process.join(timeout=interval)
        queue_item = service.repository.get_queue_item(payload["job_id"])
        if queue_item is not None and queue_item["status"] == "cancel_requested":
            process.terminate()
            process.join(timeout=5)
            queue.complete_cancellation(payload["job_id"], payload["lease_token"])
            break
        if process.is_alive() and not queue.renew(payload["job_id"], payload["lease_token"]):
            process.terminate()
            process.join(timeout=5)
            raise RuntimeError("job worker lost its lease")
    current_item = service.repository.get_queue_item(payload["job_id"])
    if current_item is None:
        raise RuntimeError("claimed queue item disappeared")
    result: dict[str, Any]
    if current_item["status"] == "cancel_requested":
        queue.complete_cancellation(payload["job_id"], payload["lease_token"])
        result = {"cancelled": True}
    elif current_item["status"] == "cancelled":
        result = {"cancelled": True}
    else:
        try:
            result = result_queue.get(timeout=5)
        except queue_module.Empty:
            result = {"error_type": f"WorkerExit{process.exitcode}"}
        session_documents_path = result.get("session_documents_path")
        if result.get("completed") and session_documents_path:
            transfer_path = Path(str(session_documents_path)).resolve()
            try:
                transfer_path.relative_to(service.config.upload_dir.resolve())
                documents = json.loads(transfer_path.read_text(encoding="utf-8"))
                if not isinstance(documents, list):
                    raise ValueError("session document transfer must be a list")
                service._retain_session_documents(
                    payload["tenant_id"],
                    payload["user_id"],
                    payload["thread_id"],
                    documents,
                )
            except (OSError, ValueError) as exc:
                result = {"error_type": type(exc).__name__}
            finally:
                transfer_path.unlink(missing_ok=True)
        if result.get("completed"):
            if not queue.complete(payload["job_id"], payload["lease_token"]):
                raise RuntimeError("analysis completed after its job lease expired")
            service.repository.update_job_status(job_id=payload["job_id"], status="completed")
        else:
            result["queue_status"] = queue.fail(
                payload["job_id"], payload["lease_token"], str(result["error_type"])
            )
    if (
        result.get("completed") or result.get("cancelled") or result.get("queue_status") == "dead"
    ) and payload.get("cleanup_documents"):
        for document_path in payload.get("document_paths") or []:
            Path(document_path).unlink(missing_ok=True)
    result_queue.close()
    return True


def work_forever() -> None:
    config = AppConfig.from_env()
    service = FinanceAnalysisService(config)
    worker_id = f"worker-{socket.gethostname()}-{os.getpid()}"
    next_retention_at = 0.0
    try:
        while True:
            if time.monotonic() >= next_retention_at:
                service.run_retention()
                next_retention_at = time.monotonic() + 3_600
            if not process_one_isolated_job(service, worker_id):
                time.sleep(1)
    finally:
        service.close()

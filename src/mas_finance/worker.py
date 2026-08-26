from __future__ import annotations

from .config import AppConfig
from .queueing import RedisQueueManager
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
    service.run_job(
        job_id=job_id,
        query=query,
        thread_id=thread_id,
        export_artifacts=export_artifacts,
        document_paths=document_paths or [],
        cleanup_documents=cleanup_documents,
    )


def work_forever() -> None:
    config = AppConfig.from_env()
    if not config.redis_url:
        raise RuntimeError("MAS_REDIS_URL is required to start the Redis worker.")
    queue = RedisQueueManager(config.redis_url, config.redis_queue_name)
    while True:
        payload = queue.dequeue(timeout_seconds=5)
        if not payload:
            continue
        execute_analysis_job(
            job_id=payload["job_id"],
            query=payload["query"],
            thread_id=payload["thread_id"],
            export_artifacts=payload.get("export_artifacts", True),
            document_paths=payload.get("document_paths", []),
            cleanup_documents=payload.get("cleanup_documents", False),
        )

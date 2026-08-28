from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Integer, Text, UniqueConstraint, create_engine, inspect, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import NullPool


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Base(DeclarativeBase):
    pass


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    llm_backend: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifacts_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class AnalysisJobQueueItem(Base):
    __tablename__ = "analysis_job_queue"
    __table_args__ = (UniqueConstraint("idempotency_key"),)

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    available_at: Mapped[str] = mapped_column(Text, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_expires_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class JobRepository:
    def __init__(self, database_url: str, db_path: Path | None = None) -> None:
        if database_url.startswith("sqlite:///") and db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        engine_options = {"poolclass": NullPool} if database_url.startswith("sqlite:///") else {}
        self.engine = create_engine(database_url, future=True, **engine_options)
        Base.metadata.create_all(self.engine)
        self._migrate_principal_columns()

    def _migrate_principal_columns(self) -> None:
        inspector = inspect(self.engine)
        with self.engine.begin() as connection:
            for table_name in ("analysis_jobs", "analysis_job_queue"):
                columns = {str(column["name"]) for column in inspector.get_columns(table_name)}
                added_principal = False
                if "tenant_id" not in columns:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'")
                    )
                    added_principal = True
                if "user_id" not in columns:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'owner'")
                    )
                    added_principal = True
                if table_name == "analysis_job_queue" and added_principal:
                    rows = connection.execute(
                        text("SELECT job_id, tenant_id, user_id, idempotency_key FROM analysis_job_queue")
                    ).mappings()
                    for row in rows:
                        connection.execute(
                            text(
                                """
                                UPDATE analysis_job_queue
                                SET idempotency_key = :scoped_key
                                WHERE job_id = :job_id
                                """
                            ),
                            {
                                "job_id": row["job_id"],
                                "scoped_key": json.dumps(
                                    [row["tenant_id"], row["user_id"], row["idempotency_key"]],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        )

    def create_job(
        self,
        job_id: str,
        thread_id: str,
        query: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> None:
        now = utc_now()
        with Session(self.engine) as session:
            session.add(
                AnalysisJob(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    thread_id=thread_id,
                    query=query,
                    status="pending",
                    llm_backend=None,
                    result_json=None,
                    artifacts_json=None,
                    error_message=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()

    def submit_job(
        self,
        *,
        job_id: str,
        thread_id: str,
        query: str,
        payload: dict[str, Any],
        idempotency_key: str,
        max_attempts: int,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> tuple[dict[str, Any], bool]:
        if not idempotency_key.strip() or len(idempotency_key) > 200:
            raise ValueError("job idempotency key is invalid")
        if not 1 <= max_attempts <= 10:
            raise ValueError("job max attempts must be between 1 and 10")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(serialized) > 100_000:
            raise ValueError("job payload exceeds length limit")
        now = utc_now()
        with Session(self.engine) as session:
            scoped_idempotency_key = json.dumps(
                [tenant_id, user_id, idempotency_key], ensure_ascii=False, separators=(",", ":")
            )
            existing = session.scalar(
                select(AnalysisJobQueueItem).where(
                    AnalysisJobQueueItem.idempotency_key == scoped_idempotency_key
                )
            )
            if existing is not None:
                job = session.get(AnalysisJob, existing.job_id)
                if job is None:
                    raise RuntimeError("idempotent queue item has no analysis job")
                return self._row_to_dict(job), False
            job = AnalysisJob(
                job_id=job_id,
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=thread_id,
                query=query,
                status="pending",
                llm_backend=None,
                result_json=None,
                artifacts_json=None,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.add(
                AnalysisJobQueueItem(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    idempotency_key=scoped_idempotency_key,
                    payload_json=serialized,
                    status="pending",
                    attempt_count=0,
                    max_attempts=max_attempts,
                    available_at=now,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    last_error_type=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                existing = session.scalar(
                    select(AnalysisJobQueueItem).where(
                        AnalysisJobQueueItem.idempotency_key == scoped_idempotency_key
                    )
                )
                if existing is None:
                    raise
                job = session.get(AnalysisJob, existing.job_id)
                if job is None:
                    raise RuntimeError("idempotent queue item has no analysis job") from exc
                return self._row_to_dict(job), False
            return self._row_to_dict(job), True

    def claim_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        job_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not worker_id.strip() or len(worker_id) > 200:
            raise ValueError("queue worker id is invalid")
        if not 10 <= lease_seconds <= 3_600:
            raise ValueError("queue lease must be between 10 and 3600 seconds")
        now = datetime.now(UTC)
        now_text = now.isoformat()
        with Session(self.engine) as session:
            if self.engine.dialect.name == "sqlite":
                session.execute(text("BEGIN IMMEDIATE"))
            exhausted = session.scalars(
                select(AnalysisJobQueueItem).where(
                    AnalysisJobQueueItem.status == "leased",
                    AnalysisJobQueueItem.lease_expires_at <= now_text,
                    AnalysisJobQueueItem.attempt_count >= AnalysisJobQueueItem.max_attempts,
                )
            ).all()
            for exhausted_item in exhausted:
                exhausted_item.status = "dead"
                exhausted_item.lease_owner = None
                exhausted_item.lease_token = None
                exhausted_item.lease_expires_at = None
                exhausted_item.updated_at = now_text
                job = session.get(AnalysisJob, exhausted_item.job_id)
                if job is not None:
                    job.status = "failed"
                    job.error_message = "Analysis failed after exhausting queue attempts."
                    job.updated_at = now_text
            statement = (
                select(AnalysisJobQueueItem)
                .where(
                    AnalysisJobQueueItem.attempt_count < AnalysisJobQueueItem.max_attempts,
                    or_(
                        (
                            (AnalysisJobQueueItem.status == "pending")
                            & (AnalysisJobQueueItem.available_at <= now_text)
                        ),
                        (
                            (AnalysisJobQueueItem.status == "leased")
                            & (AnalysisJobQueueItem.lease_expires_at <= now_text)
                        ),
                    ),
                )
                .order_by(AnalysisJobQueueItem.available_at, AnalysisJobQueueItem.created_at)
                .with_for_update(skip_locked=True)
            )
            if job_id is not None:
                statement = statement.where(AnalysisJobQueueItem.job_id == job_id)
            claimed = session.scalar(statement)
            if claimed is None:
                session.commit()
                return None
            lease_token = uuid4().hex
            claimed.status = "leased"
            claimed.attempt_count += 1
            claimed.lease_owner = worker_id
            claimed.lease_token = lease_token
            claimed.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            claimed.updated_at = now_text
            job = session.get(AnalysisJob, claimed.job_id)
            if job is None:
                raise RuntimeError("queue item has no analysis job")
            job.status = "running"
            job.updated_at = now_text
            payload = json.loads(claimed.payload_json)
            session.commit()
            return {
                **payload,
                "job_id": claimed.job_id,
                "tenant_id": claimed.tenant_id,
                "user_id": claimed.user_id,
                "lease_token": lease_token,
                "attempt_count": claimed.attempt_count,
                "max_attempts": claimed.max_attempts,
            }

    def renew_job_lease(self, job_id: str, lease_token: str, lease_seconds: int) -> bool:
        now = datetime.now(UTC)
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None or item.status != "leased" or item.lease_token != lease_token:
                return False
            item.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            item.updated_at = now.isoformat()
            session.commit()
            return True

    def complete_job_lease(self, job_id: str, lease_token: str) -> bool:
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None or item.status != "leased" or item.lease_token != lease_token:
                return False
            item.status = "completed"
            item.lease_owner = None
            item.lease_token = None
            item.lease_expires_at = None
            item.updated_at = utc_now()
            session.commit()
            return True

    def fail_job_lease(
        self,
        job_id: str,
        lease_token: str,
        *,
        error_type: str,
        retry_delay_seconds: int,
    ) -> str:
        now = datetime.now(UTC)
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None or item.status != "leased" or item.lease_token != lease_token:
                raise ValueError("job lease is no longer owned by this worker")
            retry = item.attempt_count < item.max_attempts
            item.status = "pending" if retry else "dead"
            item.available_at = (now + timedelta(seconds=retry_delay_seconds)).isoformat()
            item.lease_owner = None
            item.lease_token = None
            item.lease_expires_at = None
            item.last_error_type = error_type[:200]
            item.updated_at = now.isoformat()
            job = session.get(AnalysisJob, job_id)
            if job is None:
                raise RuntimeError("queue item has no analysis job")
            job.status = "pending" if retry else "failed"
            job.error_message = None if retry else f"Analysis failed ({error_type[:100]})."
            job.updated_at = now.isoformat()
            session.commit()
            return item.status

    def get_queue_item(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None:
                return None
            return {
                "job_id": item.job_id,
                "idempotency_key": item.idempotency_key,
                "status": item.status,
                "attempt_count": item.attempt_count,
                "max_attempts": item.max_attempts,
                "available_at": item.available_at,
                "lease_owner": item.lease_owner,
                "lease_expires_at": item.lease_expires_at,
                "last_error_type": item.last_error_type,
            }

    def request_job_cancellation(self, job_id: str) -> str | None:
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None:
                return None
            if item.status == "pending":
                item.status = "cancelled"
            elif item.status == "leased":
                item.status = "cancel_requested"
            elif item.status not in {"cancel_requested", "cancelled"}:
                return item.status
            item.updated_at = utc_now()
            job = session.get(AnalysisJob, job_id)
            if job is not None:
                job.status = item.status
                job.updated_at = item.updated_at
            session.commit()
            return item.status

    def complete_job_cancellation(self, job_id: str, lease_token: str) -> bool:
        with Session(self.engine) as session:
            item = session.get(AnalysisJobQueueItem, job_id)
            if item is None or item.status != "cancel_requested" or item.lease_token != lease_token:
                return False
            item.status = "cancelled"
            item.lease_owner = None
            item.lease_token = None
            item.lease_expires_at = None
            item.updated_at = utc_now()
            job = session.get(AnalysisJob, job_id)
            if job is not None:
                job.status = "cancelled"
                job.updated_at = item.updated_at
            session.commit()
            return True

    def update_job_status(
        self,
        job_id: str,
        status: str,
        llm_backend: str | None = None,
        result: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        error_message: str | None = None,
    ) -> None:
        with Session(self.engine) as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return
            job.status = status
            if llm_backend is not None:
                job.llm_backend = llm_backend
            if result is not None:
                job.result_json = json.dumps(result, ensure_ascii=False)
            if artifacts is not None:
                job.artifacts_json = json.dumps(artifacts, ensure_ascii=False)
            job.error_message = error_message
            job.updated_at = utc_now()
            session.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            job = session.get(AnalysisJob, job_id)
            return self._row_to_dict(job) if job is not None else None

    def get_job_for_principal(self, job_id: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            job = session.scalar(
                select(AnalysisJob).where(
                    AnalysisJob.job_id == job_id,
                    AnalysisJob.tenant_id == tenant_id,
                    AnalysisJob.user_id == user_id,
                )
            )
            return self._row_to_dict(job) if job is not None else None

    def list_jobs(self, tenant_id: str, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(AnalysisJob)
                .where(AnalysisJob.tenant_id == tenant_id, AnalysisJob.user_id == user_id)
                .order_by(AnalysisJob.created_at.desc())
                .limit(limit)
            ).all()
            return [self._row_to_dict(row) for row in rows]

    def delete_terminal_jobs_before(self, cutoff: str) -> int:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(AnalysisJobQueueItem).where(
                    AnalysisJobQueueItem.status.in_(("completed", "dead")),
                    AnalysisJobQueueItem.updated_at < cutoff,
                )
            ).all()
            job_ids = [row.job_id for row in rows]
            for row in rows:
                session.delete(row)
            for job_id in job_ids:
                job = session.get(AnalysisJob, job_id)
                if job is not None:
                    session.delete(job)
            session.commit()
            return len(job_ids)

    def _row_to_dict(self, row: AnalysisJob) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "tenant_id": row.tenant_id,
            "user_id": row.user_id,
            "thread_id": row.thread_id,
            "query": row.query,
            "status": row.status,
            "llm_backend": row.llm_backend,
            "result": json.loads(row.result_json) if row.result_json else None,
            "artifacts": json.loads(row.artifacts_json) if row.artifacts_json else {},
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

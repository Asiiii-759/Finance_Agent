from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .database import JobRepository


@dataclass(frozen=True)
class ReliableJobQueue:
    repository: JobRepository
    lease_seconds: int = 300
    retry_delay_seconds: int = 30

    def claim(self, worker_id: str, *, job_id: str | None = None) -> dict[str, Any] | None:
        return self.repository.claim_job(
            worker_id=worker_id,
            lease_seconds=self.lease_seconds,
            job_id=job_id,
        )

    def renew(self, job_id: str, lease_token: str) -> bool:
        return self.repository.renew_job_lease(job_id, lease_token, self.lease_seconds)

    def complete(self, job_id: str, lease_token: str) -> bool:
        return self.repository.complete_job_lease(job_id, lease_token)

    def fail(self, job_id: str, lease_token: str, error_type: str) -> str:
        return self.repository.fail_job_lease(
            job_id,
            lease_token,
            error_type=error_type,
            retry_delay_seconds=self.retry_delay_seconds,
        )

    def request_cancellation(self, job_id: str) -> str | None:
        return self.repository.request_job_cancellation(job_id)

    def complete_cancellation(self, job_id: str, lease_token: str) -> bool:
        return self.repository.complete_job_cancellation(job_id, lease_token)

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str
    created_at: float
    result: Optional[dict] = None
    error: Optional[dict] = None


class JobStore:
    def __init__(self, ttl_seconds: int, clock: Callable[[], float] = time.monotonic):
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._jobs: Dict[str, Job] = {}

    def create(self) -> Job:
        self.purge_expired()
        job = Job(job_id=uuid.uuid4().hex, status="queued", created_at=self._clock())
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def mark_completed(self, job_id: str, result: dict) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "completed"
            job.result = result

    def mark_failed(self, job_id: str, error: dict) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "failed"
            job.error = error

    def purge_expired(self) -> None:
        cutoff = self._clock() - self._ttl_seconds
        expired = [job_id for job_id, job in self._jobs.items() if job.created_at < cutoff]
        for job_id in expired:
            del self._jobs[job_id]

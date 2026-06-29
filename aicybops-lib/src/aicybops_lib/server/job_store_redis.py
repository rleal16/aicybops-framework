import json
import time
from typing import Any, Dict, Optional, Tuple

import redis

TERMINAL_STATUSES = {"completed", "failed"}


class RedisJobStore:
    def __init__(self, redis_url: str, namespace: str = "aicybops") -> None:
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._namespace = namespace
        self._queue_key = f"{namespace}:jobs:queue"

    def ping(self) -> bool:
        return bool(self._redis.ping())

    def _job_key(self, job_type: str, job_id: str) -> str:
        return f"{self._namespace}:job:{job_type}:{job_id}"

    def create_job(self, job_type: str, job_id: str, payload: Dict[str, Any]) -> None:
        self._redis.set(self._job_key(job_type, job_id), json.dumps(payload))

    def get_job(self, job_type: str, job_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._job_key(job_type, job_id))
        if raw is None:
            return None
        return json.loads(raw)

    def update_job(self, job_type: str, job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._job_key(job_type, job_id)
        raw = self._redis.get(key)
        if raw is None:
            return None
        current = json.loads(raw)
        if current.get("status") in TERMINAL_STATUSES:
            return current
        merged = {**current, **updates, "updated_at": time.time()}
        self._redis.set(key, json.dumps(merged))
        return merged

    def enqueue_job(self, job_type: str, job_id: str, request_payload: Dict[str, Any]) -> None:
        envelope = {"job_type": job_type, "job_id": job_id, "request": request_payload}
        self._redis.lpush(self._queue_key, json.dumps(envelope))

    def dequeue_job(self, timeout_seconds: int = 5) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        item = self._redis.brpop(self._queue_key, timeout=timeout_seconds)
        if not item:
            return None
        _, payload = item
        parsed = json.loads(payload)
        return parsed["job_type"], parsed["job_id"], parsed["request"]

    def recover_interrupted_jobs(
        self,
        reason: str = "Worker process exited before the job finished",
    ) -> int:
        """
        Mark jobs left in 'running' as failed (e.g. after worker crash or restart).
        Returns the number of jobs updated.
        """
        prefix = f"{self._namespace}:job:"
        recovered = 0
        for key in self._redis.scan_iter(match=f"{prefix}*"):
            raw = self._redis.get(key)
            if not raw:
                continue
            job = json.loads(raw)
            if job.get("status") != "running":
                continue
            # key format: {namespace}:job:{job_type}:{job_id}
            parts = key.split(":")
            if len(parts) < 4:
                continue
            job_type, job_id = parts[-2], parts[-1]
            self.update_job(
                job_type,
                job_id,
                {"status": "failed", "error": reason},
            )
            recovered += 1
        return recovered

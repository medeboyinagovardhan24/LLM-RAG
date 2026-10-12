"""Response caching with TTL and request monitoring middleware."""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field


class ResponseCache:
    """In-memory response cache with configurable TTL and per-document invalidation."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache: dict[str, dict] = {}
        self._ttl = ttl_seconds
        self._max = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def cache_key(question: str, filter_filename: str | None = None) -> str:
        raw = f"{question.strip().lower()}|{filter_filename or ''}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry["data"]

    def set(self, key: str, data: dict) -> None:
        if len(self._cache) >= self._max:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest_key]
        self._cache[key] = {"data": data, "ts": time.time()}

    def invalidate(self, filename: str) -> int:
        """Remove all cache entries whose data references this filename."""
        to_remove = []
        for key, entry in self._cache.items():
            data = entry["data"]
            sources = data.get("sources", [])
            if any(s.get("filename") == filename for s in sources):
                to_remove.append(key)
        for key in to_remove:
            del self._cache[key]
        return len(to_remove)

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_size": self._max,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
        }


class RequestMonitor:
    """Tracks request counts, latencies, error rates, and slow requests."""

    # Maximum number of individual request records retained for slow-request
    # queries and detailed breakdowns.
    _MAX_RECORDS = 5_000

    def __init__(self):
        self._total = 0
        self._errors = 0
        self._latency_sum = 0.0
        self._records: list[dict] = []
        self._by_endpoint: dict[str, dict] = defaultdict(lambda: {
            "count": 0,
            "errors": 0,
            "latency_sum": 0.0,
        })
        self._started_at = time.time()

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        user_agent: str | None = None,
    ) -> None:
        """Record a completed request.

        Parameters
        ----------
        endpoint:
            Route path (e.g. ``/api/ask``).
        method:
            HTTP method (GET, POST, etc.).
        status_code:
            HTTP response status code.
        latency_ms:
            Request latency in milliseconds.
        user_agent:
            Optional User-Agent header value.
        """
        now = time.time()
        is_error = status_code >= 400

        self._total += 1
        self._latency_sum += latency_ms
        if is_error:
            self._errors += 1

        ep = self._by_endpoint[endpoint]
        ep["count"] += 1
        ep["latency_sum"] += latency_ms
        if is_error:
            ep["errors"] += 1

        record = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "user_agent": user_agent,
            "timestamp": now,
        }
        self._records.append(record)
        # Trim to keep memory bounded.
        if len(self._records) > self._MAX_RECORDS:
            self._records = self._records[-self._MAX_RECORDS:]

    def get_metrics(self) -> dict:
        """Return aggregated request metrics.

        Returns
        -------
        dict with keys:
            ``total_requests`` (int), ``avg_latency`` (float ms),
            ``error_rate`` (float 0-1), ``requests_per_minute`` (float),
            ``endpoint_breakdown`` (dict[str, dict]).
        """
        avg_latency = (
            self._latency_sum / self._total if self._total > 0 else 0.0
        )
        error_rate = self._errors / self._total if self._total > 0 else 0.0

        # Requests per minute over the last 60 seconds.
        cutoff = time.time() - 60
        recent_count = sum(
            1 for r in self._records if r["timestamp"] >= cutoff
        )

        breakdown: dict[str, dict] = {}
        for endpoint, data in self._by_endpoint.items():
            count = data["count"]
            breakdown[endpoint] = {
                "count": count,
                "errors": data["errors"],
                "avg_latency_ms": (
                    round(data["latency_sum"] / count, 2) if count > 0 else 0.0
                ),
                "error_rate": round(
                    data["errors"] / count, 4
                ) if count > 0 else 0.0,
            }

        return {
            "total_requests": self._total,
            "avg_latency": round(avg_latency, 2),
            "error_rate": round(error_rate, 4),
            "requests_per_minute": round(recent_count, 1),
            "endpoint_breakdown": dict(breakdown),
        }

    def get_slow_requests(self, threshold_ms: float = 5000.0) -> list[dict]:
        """Return recent requests whose latency exceeds *threshold_ms*.

        Parameters
        ----------
        threshold_ms:
            Latency threshold in milliseconds.

        Returns
        -------
        list[dict]
            Up to 50 of the most recent slow request records.
        """
        slow = [r for r in self._records if r["latency_ms"] >= threshold_ms]
        slow.sort(key=lambda r: r["timestamp"], reverse=True)
        return slow[:50]


# Module-level singletons
response_cache = ResponseCache()
request_monitor = RequestMonitor()

"""Simple conversation history and analytics tracking with disk persistence."""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from threading import RLock
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)
PERSIST_FILE = Path(
    os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent)
) / "memory_store.json"

MAX_SESSIONS = 1_000
MAX_TURNS_PER_SESSION = 100
MAX_QUERY_RECORDS = 5_000
MAX_LATENCY_RECORDS = 5_000


class MemoryStore:
    def __init__(self):
        self.sessions: dict[str, list] = {}
        self.queries: list = []
        self.feedback: dict[str, int] = {"up": 0, "down": 0}
        self.latencies: list[float] = []
        self._lock = RLock()
        self._load()

    # --- persistence ---------------------------------------------------
    def _load(self):
        if PERSIST_FILE.exists():
            try:
                data = json.loads(PERSIST_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Memory store must contain a JSON object")
                sessions = data.get("sessions", {})
                queries = data.get("queries", [])
                feedback = data.get("feedback", {"up": 0, "down": 0})
                latencies = data.get("latencies", [])
                if not isinstance(sessions, dict) or not isinstance(queries, list):
                    raise ValueError("Memory store sessions or queries have an invalid shape")
                if not isinstance(feedback, dict) or not isinstance(latencies, list):
                    raise ValueError("Memory store feedback or latencies have an invalid shape")
                self.sessions = {
                    str(session_id): turns[-MAX_TURNS_PER_SESSION:]
                    for session_id, turns in list(sessions.items())[-MAX_SESSIONS:]
                    if isinstance(turns, list)
                }
                self.queries = queries[-MAX_QUERY_RECORDS:]
                self.feedback = {
                    "up": int(feedback.get("up", 0)),
                    "down": int(feedback.get("down", 0)),
                }
                self.latencies = latencies[-MAX_LATENCY_RECORDS:]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                logger.exception("Could not load conversation memory from %s", PERSIST_FILE)

    def _save(self):
        temporary_path: str | None = None
        try:
            PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({
                    "sessions": self.sessions,
                    "queries": self.queries,
                    "feedback": self.feedback,
                    "latencies": self.latencies,
                }, default=str)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=PERSIST_FILE.parent,
                prefix=f"{PERSIST_FILE.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temporary_path = temp_file.name
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temporary_path, PERSIST_FILE)
        except OSError:
            logger.exception("Could not save conversation memory to %s", PERSIST_FILE)
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    logger.exception("Could not remove temporary memory file %s", temporary_path)

    # --- public API ----------------------------------------------------
    def add_turn(self, session_id: str, user: str, assistant: str, latency_ms: float = 0.0):
        with self._lock:
            turns = self.sessions.setdefault(session_id, [])
            turns.append({
                "user": user,
                "assistant": assistant,
                "ts": time.time(),
            })
            del turns[:-MAX_TURNS_PER_SESSION]
            self.sessions[session_id] = self.sessions.pop(session_id)
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.pop(next(iter(self.sessions)))

            self.queries.append({"q": user, "ts": time.time(), "len": len(user)})
            del self.queries[:-MAX_QUERY_RECORDS]
            if latency_ms > 0:
                self.latencies.append(latency_ms)
                del self.latencies[:-MAX_LATENCY_RECORDS]
            self._save()

    def record_feedback(self, rating: str, question: str):
        with self._lock:
            if rating in self.feedback:
                self.feedback[rating] += 1
                self._save()

    def reset_analytics(self):
        with self._lock:
            self.queries.clear()
            self.feedback = {"up": 0, "down": 0}
            self.latencies.clear()
            self._save()

    def avg_latency(self) -> float:
        with self._lock:
            if not self.latencies:
                return 0.0
            return round(sum(self.latencies) / len(self.latencies), 1)

    def analytics_snapshot(self) -> dict:
        with self._lock:
            recent_queries = [
                {"question": query["q"], "ts": query["ts"]}
                for query in sorted(
                    self.queries,
                    key=lambda item: item["ts"],
                    reverse=True,
                )[:10]
            ]
            average_latency = (
                round(sum(self.latencies) / len(self.latencies), 1)
                if self.latencies
                else 0.0
            )
            return {
                "total_queries": len(self.queries),
                "thumbs_up": self.feedback["up"],
                "thumbs_down": self.feedback["down"],
                "avg_latency_ms": average_latency,
                "recent_queries": recent_queries,
            }

    def get_session(self, session_id: str) -> list:
        with self._lock:
            return list(self.sessions.get(session_id, []))


store = MemoryStore()

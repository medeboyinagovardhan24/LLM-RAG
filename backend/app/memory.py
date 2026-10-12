"""Simple conversation history and analytics tracking with disk persistence."""
from __future__ import annotations
import json
import time
from pathlib import Path

PERSIST_FILE = Path(__file__).resolve().parent.parent / "memory_store.json"


class MemoryStore:
    def __init__(self):
        self.sessions: dict[str, list] = {}
        self.queries: list = []
        self.feedback: dict[str, int] = {"up": 0, "down": 0}
        self.latencies: list[float] = []
        self._load()

    # --- persistence ---------------------------------------------------
    def _load(self):
        if PERSIST_FILE.exists():
            try:
                data = json.loads(PERSIST_FILE.read_text(encoding="utf-8"))
                self.sessions = data.get("sessions", {})
                self.queries = data.get("queries", [])
                self.feedback = data.get("feedback", {"up": 0, "down": 0})
                self.latencies = data.get("latencies", [])
            except Exception:
                pass  # start fresh on corrupt file

    def _save(self):
        try:
            PERSIST_FILE.write_text(
                json.dumps({
                    "sessions": self.sessions,
                    "queries": self.queries,
                    "feedback": self.feedback,
                    "latencies": self.latencies,
                }, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    # --- public API ----------------------------------------------------
    def add_turn(self, session_id: str, user: str, assistant: str, latency_ms: float = 0.0):
        self.sessions.setdefault(session_id, []).append({
            "user": user,
            "assistant": assistant,
            "ts": time.time(),
        })
        self.queries.append({"q": user, "ts": time.time(), "len": len(user)})
        if latency_ms > 0:
            self.latencies.append(latency_ms)
        self._save()

    def record_feedback(self, rating: str, question: str):
        if rating in self.feedback:
            self.feedback[rating] += 1
            self._save()

    def reset_analytics(self):
        self.queries.clear()
        self.feedback = {"up": 0, "down": 0}
        self.latencies.clear()
        self._save()

    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return round(sum(self.latencies) / len(self.latencies), 1)

    def get_session(self, session_id: str) -> list:
        return self.sessions.get(session_id, [])


store = MemoryStore()

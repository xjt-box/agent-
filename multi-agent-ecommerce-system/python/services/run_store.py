"""SQLite persistence for resumable recommendation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from models.schemas import RecommendationRequest, RecommendationResponse


class RunBase(DeclarativeBase):
    """Base class for recommendation run persistence models."""


class RecommendationRunRecord(RunBase):
    """Durable state for one recommendation request."""

    __tablename__ = "recommendation_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    stage: Mapped[str] = mapped_column(String(32), default="started")
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


@dataclass(frozen=True)
class RecommendationCheckpoint:
    """A decoded checkpoint ready for orchestration recovery."""

    run_id: str
    user_id: str
    status: str
    stage: str
    state: dict[str, Any]


class RecommendationRunStore:
    """Stores request checkpoints in the configured SQLite database."""

    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        RunBase.metadata.create_all(self.engine)

    def close(self) -> None:
        """Release database connections held by the store engine."""
        self.engine.dispose()

    def start(self, run_id: str, request: RecommendationRequest) -> None:
        """Create a new run record before executing its first stage."""
        state = {"request": request.model_dump(mode="json")}
        with self.session_factory() as session:
            if session.get(RecommendationRunRecord, run_id) is not None:
                raise ValueError(f"Recommendation run already exists: {run_id}")
            session.add(
                RecommendationRunRecord(
                    run_id=run_id,
                    user_id=request.user_id,
                    state_json=self._encode(state),
                )
            )
            session.commit()

    def save_checkpoint(self, run_id: str, stage: str, state: dict[str, Any]) -> None:
        """Persist an orchestration stage and its serializable state."""
        with self.session_factory() as session:
            record = session.get(RecommendationRunRecord, run_id)
            if record is None:
                raise ValueError(f"Recommendation run was not found: {run_id}")
            record.stage = stage
            record.status = "running"
            record.state_json = self._encode(state)
            session.commit()

    def load_checkpoint(self, run_id: str) -> RecommendationCheckpoint | None:
        """Load the latest checkpoint for a run, if it exists."""
        with self.session_factory() as session:
            record = session.get(RecommendationRunRecord, run_id)
            if record is None:
                return None
            return RecommendationCheckpoint(
                run_id=record.run_id,
                user_id=record.user_id,
                status=record.status,
                stage=record.stage,
                state=self._decode(record.state_json),
            )

    def complete(self, run_id: str, response: RecommendationResponse) -> None:
        """Mark a run complete and retain its final response."""
        self.save_checkpoint(run_id, "completed", {"response": response.model_dump(mode="json")})
        with self.session_factory() as session:
            record = session.get(RecommendationRunRecord, run_id)
            if record is None:
                raise ValueError(f"Recommendation run was not found: {run_id}")
            record.status = "completed"
            session.commit()

    def fail(self, run_id: str, stage: str, error: str) -> None:
        """Record a failed stage without hiding the original exception."""
        with self.session_factory() as session:
            record = session.get(RecommendationRunRecord, run_id)
            if record is None:
                return
            state = self._decode(record.state_json)
            state["error"] = error
            record.stage = stage
            record.status = "failed"
            record.state_json = self._encode(state)
            session.commit()

    @staticmethod
    def _encode(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _decode(value: str) -> dict[str, Any]:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("Recommendation run state must be a JSON object.")
        return decoded
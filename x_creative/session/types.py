"""Session and stage data types."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageInfo(BaseModel):
    """Information about a pipeline stage."""

    status: StageStatus = Field(default=StageStatus.PENDING)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    error: str | None = Field(default=None)


StageName = Literal["problem", "biso", "search", "verify"]

# Stage dependencies: each stage requires the previous one to be completed
STAGE_DEPENDENCIES: dict[StageName, StageName | None] = {
    "problem": None,
    "biso": "problem",
    "search": "biso",
    "verify": "search",
}

STAGE_ORDER: list[StageName] = ["problem", "biso", "search", "verify"]


class Session(BaseModel):
    """A creative research session."""

    id: str = Field(..., description="Unique session ID")
    topic: str = Field(..., description="Session topic/description")
    created_at: datetime = Field(default_factory=datetime.now)
    current_stage: StageName = Field(default="problem")
    stages: dict[StageName, StageInfo] = Field(
        default_factory=lambda: {
            "problem": StageInfo(),
            "biso": StageInfo(),
            "search": StageInfo(),
            "verify": StageInfo(),
        }
    )

    def get_stage_status(self, stage: StageName) -> StageStatus:
        """Get the status of a specific stage."""
        return self.stages[stage].status

    def is_stage_completed(self, stage: StageName) -> bool:
        """Check if a stage is completed."""
        return self.stages[stage].status == StageStatus.COMPLETED

    def can_run_stage(self, stage: StageName) -> bool:
        """Check if a stage can be run (dependencies satisfied)."""
        dependency = STAGE_DEPENDENCIES[stage]
        if dependency is None:
            return True
        return self.is_stage_completed(dependency)

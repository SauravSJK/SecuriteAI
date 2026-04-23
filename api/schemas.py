"""
SecuriteAI Data Schemas
-----------------------
Description: Pydantic models for request validation and type safety.

These schemas enforce strict data governance at the ingestion layer,
ensuring the neural engine only processes well-formed system logs.
"""

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """
    Schema for an individual Linux system log entry.

    Attributes:
        Year, Month, Date, Time: Temporal markers for cyclical engineering.
        Component: The originating service (e.g., 'auth', 'systemd').
        EventId: The categorical event type used for Isolation Normalization.
        Content: Raw log message used for semantic embedding.
    """

    Year: int = Field(..., ge=2000, le=2100, examples=[2024])
    Month: str = Field(
        ...,
        pattern="^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
        examples=["Jan"],
    )
    Date: int = Field(..., ge=1, le=31, examples=[15])
    Time: str = Field(..., examples=["14:23:45"])
    Component: str = Field(...)
    EventId: str = Field(..., min_length=1, max_length=255, examples=["E02"])
    Content: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        examples=["Started User Manager for UID 1000"],
    )


class FeedbackRequest(BaseModel):
    """
    Schema for a 20-log window submitted for auditor feedback.

    This model supports the Human-in-the-Loop GRC requirement, allowing auditors
    to flag specific sequences for the retraining pipeline.
    """

    logs: list[LogEntry] = Field(..., min_length=20, max_length=20)

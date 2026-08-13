"""Pydantic models for structured transcript output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TranscriptionMethod(str, Enum):
    """How the transcript was obtained."""
    SUBTITLE = "subtitle"        # Platform subtitles (zero cost)
    WHISPER = "whisper"          # Local Whisper transcription
    UNSUPPORTED = "unsupported"  # Platform not supported (guidance returned)


class TranscriptSegment(BaseModel):
    """A single timestamped text segment."""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcript text for this segment")


class VideoInfo(BaseModel):
    """Metadata about the source video/audio."""
    title: str = Field(default="Unknown", description="Video title")
    uploader: str = Field(default="Unknown", description="Channel/uploader name")
    duration: int = Field(default=0, description="Duration in seconds")
    platform: str = Field(default="unknown", description="Source platform")
    url: str = Field(default="", description="Source URL")


class TranscriptResult(BaseModel):
    """Complete transcription result."""
    video_info: VideoInfo = Field(description="Source video metadata")
    method: TranscriptionMethod = Field(description="Transcription method used")
    segments: list[TranscriptSegment] = Field(default_factory=list, description="Timestamped segments")
    full_text: str = Field(default="", description="Full transcript as plain text")
    subtitle_language: str | None = Field(default=None, description="Subtitle language code if from subtitles")
    whisper_model: str | None = Field(default=None, description="Whisper model used if applicable")
    warning: str | None = Field(default=None, description="Warning message if any issues occurred")
    guidance: str | None = Field(default=None, description="Guidance for unsupported platforms")


class TaskStatus(str, Enum):
    """Status of an async transcription task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """Status info for an async transcription task."""
    task_id: str = Field(description="Unique task identifier")
    status: TaskStatus = Field(description="Current task status")
    progress: float = Field(default=0.0, description="Progress 0.0-1.0")
    message: str = Field(default="", description="Human-readable status message")
    result: TranscriptResult | None = Field(default=None, description="Result if completed")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: float = Field(description="Task creation timestamp")
    completed_at: float | None = Field(default=None, description="Task completion timestamp")


class TranscriptListItem(BaseModel):
    """A completed transcript in the listing."""
    task_id: str = Field(description="Task identifier")
    title: str = Field(description="Video title")
    platform: str = Field(description="Source platform")
    method: TranscriptionMethod = Field(description="Transcription method")
    duration: int = Field(default=0, description="Video duration in seconds")
    segment_count: int = Field(default=0, description="Number of transcript segments")
    created_at: float = Field(description="Creation timestamp")

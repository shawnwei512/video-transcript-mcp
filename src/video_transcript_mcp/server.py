"""MCP Server for video/audio transcription.

Provides 4 tools:
  - transcribe_url: Transcribe video from URL (sync or async mode)
  - transcribe_file: Transcribe local audio/video file
  - get_transcript_status: Poll status of async transcription task
  - list_transcripts: List all completed transcripts

Three-tier transcription strategy:
  Tier 1: Platform subtitle extraction (zero cost)
  Tier 2: Whisper local transcription (offline, free)
  Tier 3: Mini-program guidance for closed platforms
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Annotated

from pydantic import Field

from .models import (
    TaskInfo,
    TaskStatus,
    TranscriptResult,
)
from .transcriber import DEFAULT_LANGUAGE, DEFAULT_MODEL, SEGMENT_MINUTES, Transcriber

# MCP SDK import (compatible with v1 and v2)
try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
except ImportError:
    try:
        from mcp.server.fastmcp import Context
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install 'mcp[cli]'"
        )

logger = logging.getLogger(__name__)

# ============================================================
# Task Store (in-memory, suitable for stdio single-client mode)
# ============================================================
_task_store: dict[str, TaskInfo] = {}
_completed_results: dict[str, TranscriptResult] = {}
_transcribers: dict[str, Transcriber] = {}

# Default output directory for transcripts
DEFAULT_OUTPUT_DIR = os.environ.get(
    "TRANSCRIPT_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), ".video-transcript-mcp", "output"),
)


# ============================================================
# MCP Server
# ============================================================
mcp = MCPServer(
    "video-transcript-mcp",
    instructions=(
        "A video/audio transcription server supporting YouTube, Bilibili, "
        "Douyin, Kuaishou, TikTok, and more. Uses a three-tier strategy: "
        "subtitle extraction first, Whisper transcription as fallback, "
        "and mini-program guidance for closed platforms. "
        "For long videos, use async_mode=True to get a task_id, "
        "then poll with get_transcript_status."
    ),
)


# ----------------------------------------------------------
# Tool: transcribe_url
# ----------------------------------------------------------
@mcp.tool()
async def transcribe_url(
    url: Annotated[str, Field(description="Video URL (YouTube, Bilibili, Douyin, etc.)")],
    model: Annotated[
        str,
        Field(description="Whisper model: tiny, base, small, medium, large-v3-turbo"),
    ] = DEFAULT_MODEL,
    language: Annotated[str, Field(description="Language code: zh, en, ja, etc.")] = DEFAULT_LANGUAGE,
    cookies_browser: Annotated[
        str | None,
        Field(description="Browser for cookies: chrome, firefox, safari, edge"),
    ] = None,
    skip_subtitles: Annotated[
        bool, Field(description="Skip subtitle extraction, use Whisper directly")
    ] = False,
    segment_minutes: Annotated[
        int,
        Field(description="Audio segment length in minutes for splitting long videos. Default 30. Increase for 1h+ videos."),
    ] = SEGMENT_MINUTES,
    async_mode: Annotated[
        bool,
        Field(description="If true, return task_id immediately for long videos"),
    ] = False,
    ctx: Context = None,
) -> dict:
    """Transcribe a video from URL using three-tier strategy.

    Tier 1: Extract platform subtitles (zero cost, fastest)
    Tier 2: Download audio + Whisper local transcription (offline, free)
    Tier 3: Return guidance for closed platforms (Xiaohongshu/WeChat)

    For videos under 30 minutes, use sync mode (default) for direct results.
    For longer videos, set async_mode=true to get a task_id, then poll
    with get_transcript_status. Adjust segment_minutes for 1h+ videos.
    """
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"task_{int(time.time())}")
    transcriber = Transcriber(
        output_dir=output_dir,
        model=model,
        language=language,
        cookies_browser=cookies_browser,
        segment_minutes=segment_minutes,
    )

    if async_mode:
        return await _start_async_task(
            transcriber, url, skip_subtitles, ctx, is_file=False
        )

    # Sync mode: run in thread pool with progress reporting
    return await _run_sync_transcription(
        transcriber, url, skip_subtitles, ctx, is_file=False
    )


# ----------------------------------------------------------
# Tool: transcribe_file
# ----------------------------------------------------------
@mcp.tool()
async def transcribe_file(
    file_path: Annotated[str, Field(description="Path to local audio/video file")],
    model: Annotated[
        str,
        Field(description="Whisper model: tiny, base, small, medium, large-v3-turbo"),
    ] = DEFAULT_MODEL,
    language: Annotated[str, Field(description="Language code: zh, en, ja, etc.")] = DEFAULT_LANGUAGE,
    segment_minutes: Annotated[
        int,
        Field(description="Audio segment length in minutes for splitting long files. Default 30. Increase for 1h+ files."),
    ] = SEGMENT_MINUTES,
    async_mode: Annotated[
        bool, Field(description="If true, return task_id for long files")
    ] = False,
    ctx: Context = None,
) -> dict:
    """Transcribe a local audio/video file using Whisper.

    Supports mp3, m4a, mp4, wav, flac, and other FFmpeg-compatible formats.
    For files over 30 minutes, use async_mode=true. Adjust segment_minutes
    for 1h+ files to reduce segment count.
    """
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"task_{int(time.time())}")
    transcriber = Transcriber(
        output_dir=output_dir,
        model=model,
        language=language,
        segment_minutes=segment_minutes,
    )

    if async_mode:
        return await _start_async_task(
            transcriber, file_path, False, ctx, is_file=True
        )

    return await _run_sync_transcription(
        transcriber, file_path, False, ctx, is_file=True
    )


# ----------------------------------------------------------
# Tool: get_transcript_status
# ----------------------------------------------------------
@mcp.tool()
async def get_transcript_status(
    task_id: Annotated[str, Field(description="Task ID from transcribe_url or transcribe_file")],
) -> dict:
    """Check the status of an async transcription task.

    Returns the task status, progress, and result (if completed).
    Poll this periodically until status is 'completed' or 'failed'.
    """
    task = _task_store.get(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}

    response: dict = {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
    }

    if task.status == TaskStatus.COMPLETED and task.result:
        response["result"] = task.result.model_dump()
        # Include a readable summary
        response["summary"] = _build_summary(task.result)
    elif task.status == TaskStatus.FAILED:
        response["error"] = task.error or "Unknown error"

    return response


# ----------------------------------------------------------
# Tool: list_transcripts
# ----------------------------------------------------------
@mcp.tool()
async def list_transcripts() -> list[dict]:
    """List all completed transcripts.

    Returns a list of completed transcription tasks with metadata.
    Use get_transcript_status with a task_id to get full transcript text.
    """
    items: list[dict] = []
    for task_id, task in _task_store.items():
        if task.status != TaskStatus.COMPLETED:
            continue
        result = task.result
        if not result:
            continue
        items.append({
            "task_id": task_id,
            "title": result.video_info.title,
            "platform": result.video_info.platform,
            "method": result.method.value,
            "duration": result.video_info.duration,
            "segment_count": len(result.segments),
            "created_at": task.created_at,
        })

    # Sort by creation time, newest first
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


# ============================================================
# Internal helpers
# ============================================================
async def _run_sync_transcription(
    transcriber: Transcriber,
    source: str,
    skip_subtitles: bool,
    ctx: Context,
    is_file: bool,
) -> dict:
    """Run transcription synchronously with progress reporting."""
    loop = asyncio.get_event_loop()

    # Bridge sync progress callback to async ctx.report_progress
    def progress_cb(progress: float, total: float | None, message: str) -> None:
        if ctx:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, total, message),
                loop,
            )

    def do_transcribe() -> TranscriptResult:
        if is_file:
            return transcriber.transcribe_file(source, progress_cb=progress_cb)
        else:
            return transcriber.transcribe_url(source, skip_subtitles, progress_cb)

    try:
        result = await loop.run_in_executor(None, do_transcribe)
    except Exception as e:
        logger.exception("Transcription failed")
        return {"error": f"Transcription failed: {e}"}

    # Store result
    task_id = str(uuid.uuid4())[:8]
    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        progress=1.0,
        message="Completed",
        result=result,
        created_at=time.time(),
        completed_at=time.time(),
    )
    _task_store[task_id] = task_info
    _completed_results[task_id] = result
    _transcribers[task_id] = transcriber

    response = result.model_dump()
    response["task_id"] = task_id
    response["summary"] = _build_summary(result)
    return response


async def _start_async_task(
    transcriber: Transcriber,
    source: str,
    skip_subtitles: bool,
    ctx: Context,
    is_file: bool,
) -> dict:
    """Start an async transcription task and return task_id immediately."""
    task_id = str(uuid.uuid4())[:8]

    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        message="Task queued",
        created_at=time.time(),
    )
    _task_store[task_id] = task_info

    # Start background task
    asyncio.create_task(
        _run_background_task(task_id, transcriber, source, skip_subtitles, is_file)
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Transcription started. Use get_transcript_status to poll progress.",
    }


async def _run_background_task(
    task_id: str,
    transcriber: Transcriber,
    source: str,
    skip_subtitles: bool,
    is_file: bool,
) -> None:
    """Run transcription in background, updating task store with progress."""
    task = _task_store[task_id]
    task.status = TaskStatus.PROCESSING
    task.message = "Processing..."

    loop = asyncio.get_event_loop()

    def progress_cb(progress: float, total: float | None, message: str) -> None:
        task.progress = progress / total if total else progress
        task.message = message

    def do_transcribe() -> TranscriptResult:
        if is_file:
            return transcriber.transcribe_file(source, progress_cb=progress_cb)
        else:
            return transcriber.transcribe_url(source, skip_subtitles, progress_cb)

    try:
        result = await loop.run_in_executor(None, do_transcribe)
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.message = "Completed"
        task.result = result
        task.completed_at = time.time()
        _completed_results[task_id] = result
        _transcribers[task_id] = transcriber
        logger.info("Task %s completed: %s", task_id, result.video_info.title)
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"Failed: {e}"
        logger.exception("Task %s failed", task_id)


def _build_summary(result: TranscriptResult) -> str:
    """Build a human-readable summary of the transcript."""
    lines: list[str] = []
    lines.append(f"Title: {result.video_info.title}")
    lines.append(f"Author: {result.video_info.uploader}")
    if result.video_info.duration:
        mins = result.video_info.duration // 60
        secs = result.video_info.duration % 60
        lines.append(f"Duration: {mins}:{secs:02d}")
    lines.append(f"Method: {result.method.value}")
    if result.subtitle_language:
        lines.append(f"Subtitle language: {result.subtitle_language}")
    if result.whisper_model:
        lines.append(f"Whisper model: {result.whisper_model}")
    lines.append(f"Segments: {len(result.segments)}")
    lines.append(f"Text length: {len(result.full_text)} characters")
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    if result.guidance:
        lines.append(f"Guidance: {result.guidance[:100]}...")
    # Preview first 200 chars
    if result.full_text:
        preview = result.full_text[:200].replace("\n", " ")
        lines.append(f"Preview: {preview}...")
    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================
def main():
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

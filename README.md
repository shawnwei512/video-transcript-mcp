# Video Transcript MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server for video/audio transcription with multi-platform support.

## Features

- **Three-tier transcription strategy**: Subtitle extraction first (zero cost) → Whisper local transcription (offline free) → Mini-program guidance for closed platforms
- **1000+ platform support** via yt-dlp: YouTube, Bilibili, Douyin, Kuaishou, TikTok, and more
- **Long video handling**: Auto-split by 30-minute segments (configurable) with checkpoint resume
- **Chinese ASR optimization**: Bilibili AI subtitles, HuggingFace mirror, SenseVoice/Paraformer ready
- **Sync & Async modes**: Direct results for short videos, task polling for long videos
- **Structured output**: Pydantic-validated results with timestamps, segments, and metadata

## Quick Start

### Install

```bash
pip install video-transcript-mcp

# With Whisper support
pip install 'video-transcript-mcp[whisper]'

# With dev tools (MCP Inspector, testing)
pip install 'video-transcript-mcp[dev]'
```

### Run

```bash
# Direct run
video-transcript-mcp

# Or with uvx (no install needed)
uvx video-transcript-mcp

# Debug with MCP Inspector
mcp dev video_transcript_mcp.server:mcp
```

### Prerequisites

The server relies on external tools for audio processing:

```bash
# Install yt-dlp (video download + subtitle extraction)
pip install yt-dlp

# Install FFmpeg (audio splitting + format conversion)
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian

# Install faster-whisper (local transcription)
pip install faster-whisper
```

## MCP Client Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "video-transcript": {
      "command": "uvx",
      "args": ["video-transcript-mcp"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "video-transcript": {
      "command": "uvx",
      "args": ["video-transcript-mcp"]
    }
  }
}
```

### Trae

Add to Trae MCP settings:

```json
{
  "mcpServers": {
    "video-transcript": {
      "command": "python3",
      "args": ["-m", "video_transcript_mcp.server"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add video-transcript -- uvx video-transcript-mcp
```

## Tools

### `transcribe_url`

Transcribe a video from URL using the three-tier strategy.

```python
# Short video (sync mode - direct result)
transcribe_url(url="https://www.youtube.com/watch?v=xxxxx")

# Long video (async mode - returns task_id)
transcribe_url(
    url="https://www.bilibili.com/video/BVxxxxx",
    async_mode=True
)
# Then poll:
get_transcript_status(task_id="abc12345")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | Video URL |
| `model` | str | `large-v3-turbo` | Whisper model |
| `language` | str | `zh` | Language code |
| `cookies_browser` | str? | null | Browser for cookies |
| `skip_subtitles` | bool | false | Skip to Whisper directly |
| `segment_minutes` | int | 30 | Segment length for long video splitting. Increase for 1h+ videos |
| `async_mode` | bool | false | Return task_id for polling |

### `transcribe_file`

Transcribe a local audio/video file.

```python
# Short file (sync mode)
transcribe_file(file_path="/path/to/audio.mp3")

# Long file (1h+) with larger segments
transcribe_file(
    file_path="/path/to/lecture.mp4",
    segment_minutes=60,
    async_mode=True
)
```

### `get_transcript_status`

Poll the status of an async transcription task.

```python
get_transcript_status(task_id="abc12345")
# Returns: {status: "completed", progress: 1.0, result: {...}}
```

### `list_transcripts`

List all completed transcripts.

```python
list_transcripts()
# Returns: [{task_id, title, platform, method, duration, ...}]
```

## Three-Tier Transcription Strategy

```
URL Input
    │
    ├─ Tier 1: Subtitle Extraction (zero cost, fastest)
    │   ├─ YouTube: zh-Hans, zh-CN, zh, en
    │   ├─ Bilibili: ai-zh (AI subtitles)
    │   └─ Others: zh-CN, zh, en
    │
    ├─ Tier 2: Whisper Transcription (offline, free)
    │   ├─ Download audio via yt-dlp
    │   ├─ Split by 30-min segments (configurable, long video)
    │   ├─ Transcribe each segment with faster-whisper
    │   ├─ Global timestamp concatenation
    │   └─ Checkpoint resume support
    │
    └─ Tier 3: Mini-Program Guidance (closed platforms)
        ├─ Xiaohongshu (小红书)
        └─ WeChat Video (视频号)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_ENDPOINT` | _(unset)_ | Set to `https://hf-mirror.com` for China network optimization |
| `HF_HUB_DISABLE_XET` | `1` | Disable Xet storage (avoids download errors) |
| `TRANSCRIPT_OUTPUT_DIR` | `~/.video-transcript-mcp/output` | Output directory |

## Supported Platforms

| Platform | Subtitle Extraction | Whisper Fallback | Notes |
|----------|:---:|:---:|-------|
| YouTube | ✅ | ✅ | Auto-subs + manual subs |
| Bilibili | ✅ | ✅ | AI subtitle (ai-zh), requires cookies for subtitle access |
| Douyin | ✅ | ✅ | |
| Kuaishou | ✅ | ✅ | |
| TikTok | ✅ | ✅ | |
| Weibo | ✅ | ✅ | |
| Xiaohongshu | ❌ | ❌ | Mini-program guidance |
| WeChat Video | ❌ | ❌ | Mini-program guidance |
| Local files | N/A | ✅ | mp3, mp4, wav, m4a, flac |
| Podcast URLs | ✅ | ✅ | Direct audio download |

## License

MIT
"""Tests for video-transcript-mcp server."""

import pytest

import video_transcript_mcp.server as srv
from video_transcript_mcp.models import (
    TaskInfo,
    TaskStatus,
    TranscriptionMethod,
    TranscriptResult,
    TranscriptSegment,
    VideoInfo,
)
from video_transcript_mcp.transcriber import (
    Transcriber,
    detect_platform,
)


# ============================================================
# Platform detection tests
# ============================================================
class TestPlatformDetection:
    def test_youtube(self):
        assert detect_platform("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtube_short(self):
        assert detect_platform("https://youtu.be/abc") == "youtube"

    def test_bilibili(self):
        assert detect_platform("https://www.bilibili.com/video/BV1234") == "bilibili"

    def test_bilibili_short(self):
        assert detect_platform("https://b23.tv/abc") == "bilibili"

    def test_douyin(self):
        assert detect_platform("https://www.douyin.com/video/123") == "douyin"

    def test_tiktok(self):
        assert detect_platform("https://www.tiktok.com/@user/video/123") == "tiktok"

    def test_kuaishou(self):
        assert detect_platform("https://www.kuaishou.com/short-video/123") == "kuaishou"

    def test_xiaohongshu(self):
        assert detect_platform("https://www.xiaohongshu.com/explore/123") == "xiaohongshu"

    def test_xiaohongshu_short(self):
        assert detect_platform("https://xhslink.com/abc") == "xiaohongshu"

    def test_weixin_video(self):
        assert detect_platform("https://channels.weixin.qq.com/xxx") == "weixin_video"

    def test_weibo(self):
        assert detect_platform("https://weibo.com/123/456") == "weibo"

    def test_podcast_mp3(self):
        assert detect_platform("https://example.com/episode.mp3") == "podcast"

    def test_local_file_unix(self):
        assert detect_platform("/path/to/file.mp4") == "local_file"

    def test_local_file_relative(self):
        assert detect_platform("./audio.mp3") == "local_file"

    def test_unknown(self):
        assert detect_platform("https://example.com/page") == "unknown"


# ============================================================
# Pydantic model tests
# ============================================================
class TestModels:
    def test_video_info_defaults(self):
        info = VideoInfo()
        assert info.title == "Unknown"
        assert info.uploader == "Unknown"
        assert info.duration == 0
        assert info.platform == "unknown"

    def test_transcript_result_subtitle(self):
        result = TranscriptResult(
            video_info=VideoInfo(title="Test", platform="youtube"),
            method=TranscriptionMethod.SUBTITLE,
            full_text="Hello world",
            subtitle_language="en",
        )
        assert result.method == TranscriptionMethod.SUBTITLE
        assert result.full_text == "Hello world"
        assert result.subtitle_language == "en"
        assert len(result.segments) == 0

    def test_transcript_result_whisper(self):
        segments = [
            TranscriptSegment(start=0.0, end=2.5, text="Hello"),
            TranscriptSegment(start=2.5, end=5.0, text="world"),
        ]
        result = TranscriptResult(
            video_info=VideoInfo(title="Test"),
            method=TranscriptionMethod.WHISPER,
            segments=segments,
            full_text="Hello\nworld",
            whisper_model="tiny",
        )
        assert result.method == TranscriptionMethod.WHISPER
        assert len(result.segments) == 2
        assert result.whisper_model == "tiny"

    def test_transcript_result_unsupported(self):
        result = TranscriptResult(
            video_info=VideoInfo(title="Xiaohongshu"),
            method=TranscriptionMethod.UNSUPPORTED,
            guidance="Use mini-program",
        )
        assert result.method == TranscriptionMethod.UNSUPPORTED
        assert result.guidance == "Use mini-program"

    def test_task_info_pending(self):
        task = TaskInfo(
            task_id="abc123",
            status=TaskStatus.PENDING,
            created_at=1000.0,
        )
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        assert task.result is None

    def test_json_serialization(self):
        result = TranscriptResult(
            video_info=VideoInfo(title="Test", platform="youtube"),
            method=TranscriptionMethod.SUBTITLE,
            full_text="Hello",
        )
        json_str = result.model_dump_json()
        assert '"method":"subtitle"' in json_str
        assert '"title":"Test"' in json_str


# ============================================================
# MCP tool registration tests
# ============================================================
class TestMCPTools:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        tools = await srv.mcp.list_tools()
        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "transcribe_url",
            "transcribe_file",
            "get_transcript_status",
            "list_transcripts",
        }

    @pytest.mark.asyncio
    async def test_transcribe_url_schema(self):
        tools = await srv.mcp.list_tools()
        tool = next(t for t in tools if t.name == "transcribe_url")
        props = tool.input_schema.get("properties", {})
        assert "url" in props
        assert "model" in props
        assert "language" in props
        assert "cookies_browser" in props
        assert "skip_subtitles" in props
        assert "async_mode" in props
        required = tool.input_schema.get("required", [])
        assert "url" in required

    @pytest.mark.asyncio
    async def test_transcribe_file_schema(self):
        tools = await srv.mcp.list_tools()
        tool = next(t for t in tools if t.name == "transcribe_file")
        props = tool.input_schema.get("properties", {})
        assert "file_path" in props
        assert "model" in props
        assert "language" in props
        assert "async_mode" in props
        required = tool.input_schema.get("required", [])
        assert "file_path" in required


# ============================================================
# MCP tool behavior tests (no network required)
# ============================================================
class TestToolBehavior:
    @pytest.mark.asyncio
    async def test_list_transcripts_empty(self):
        """list_transcripts should return empty list when no tasks."""
        result = await srv.list_transcripts()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_transcript_status_not_found(self):
        """get_transcript_status should return error for unknown task."""
        result = await srv.get_transcript_status(task_id="nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    @pytest.mark.asyncio
    async def test_transcribe_url_closed_platform(self):
        """transcribe_url should return guidance for closed platforms."""
        result = await srv.transcribe_url(
            url="https://www.xiaohongshu.com/explore/123456789",
            model="tiny",
            language="zh",
        )
        assert result["method"] == "unsupported"
        assert result["guidance"] is not None
        assert "mini-program" in result["guidance"].lower() or "\u5c0f\u7a0b\u5e8f" in result["guidance"]

    @pytest.mark.asyncio
    async def test_transcribe_url_weixin_video(self):
        """transcribe_url should return guidance for WeChat Video."""
        result = await srv.transcribe_url(
            url="https://channels.weixin.qq.com/xxx",
            model="tiny",
        )
        assert result["method"] == "unsupported"
        assert result["guidance"] is not None

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        """transcribe_file should return warning for non-existent file."""
        result = await srv.transcribe_file(
            file_path="/tmp/nonexistent_audio_file.mp3",
            model="tiny",
        )
        assert result["warning"] is not None
        assert "not found" in result["warning"].lower()


# ============================================================
# Transcriber unit tests
# ============================================================
class TestTranscriber:
    def test_init_creates_output_dir(self, tmp_path):
        output_dir = str(tmp_path / "test_output")
        Transcriber(output_dir=output_dir)
        import os
        assert os.path.exists(output_dir)

    def test_segment_minutes_default(self, tmp_path):
        """Transcriber should default to 30-minute segments."""
        transcriber = Transcriber(output_dir=str(tmp_path / "test"))
        assert transcriber.segment_minutes == 30

    def test_segment_minutes_custom(self, tmp_path):
        """Transcriber should accept custom segment_minutes."""
        transcriber = Transcriber(output_dir=str(tmp_path / "test"), segment_minutes=60)
        assert transcriber.segment_minutes == 60

    def test_segment_minutes_from_transcriber_module(self):
        """SEGMENT_MINUTES constant should be 30."""
        from video_transcript_mcp.transcriber import SEGMENT_MINUTES
        assert SEGMENT_MINUTES == 30

    def test_parse_srt(self, tmp_path):
        """Test SRT parsing with duplicate removal."""
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:04,000
Hello world

3
00:00:04,000 --> 00:00:06,000
This is a test
"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        text = Transcriber._parse_srt(str(srt_path))
        # Duplicates should be removed
        assert "Hello world" in text
        assert "This is a test" in text
        # "Hello world" should appear only once after dedup
        assert text.count("Hello world") == 1

    def test_parse_vtt_with_attributes(self, tmp_path):
        """Test VTT parsing with align/position attributes and inline tags."""
        vtt_content = """WEBVTT
Kind: captions
Language: zh-Hans

00:00:00.320 --> 00:00:18.790 align:start position:0%

[\u97f3\u4e50]

00:00:18.800 --> 00:00:21.790 align:start position:0%

\u6211\u4eec<00:00:19.008><c>\u5bf9</c><00:00:19.216><c>\u7231\u60c5</c><00:00:19.424><c>\u5e76</c><00:00:19.632><c>\u4e0d</c><00:00:19.840><c>\u964c\u751f</c>

00:00:21.790 --> 00:00:21.800 align:start position:0%
\u6211\u4eec\u5bf9\u7231\u60c5\u5e76\u4e0d\u964c\u751f

"""
        vtt_path = tmp_path / "test.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        text = Transcriber._parse_srt(str(vtt_path))

        # VTT headers should not appear
        assert "WEBVTT" not in text
        assert "Kind:" not in text
        assert "Language:" not in text

        # Timestamp lines with attributes should not appear
        assert "align:start" not in text
        assert "position:0%" not in text
        assert "-->" not in text

        # Inline VTT tags should be stripped
        assert "<c>" not in text
        assert "<00:00:" not in text
        assert "</c>" not in text

        # Clean text should be present
        assert "[\u97f3\u4e50]" in text
        assert "\u6211\u4eec\u5bf9\u7231\u60c5\u5e76\u4e0d\u964c\u751f" in text

        # The tagged version should be cleaned to match the clean version
        # After tag removal, "\u6211\u4eec\u5bf9\u7231\u60c5\u5e76\u4e0d\u964c\u751f" from inline tags should dedup with the plain line
        assert text.count("\u6211\u4eec\u5bf9\u7231\u60c5\u5e76\u4e0d\u964c\u751f") == 1

    def test_parse_vtt_english(self, tmp_path):
        """Test simple English VTT parsing."""
        vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:01.360 --> 00:00:03.040
[\u266a\u266a\u266a]

00:00:18.640 --> 00:00:21.880
We're no strangers to love

00:00:22.640 --> 00:00:26.960
You know the rules
and so do I
"""
        vtt_path = tmp_path / "test_en.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        text = Transcriber._parse_srt(str(vtt_path))

        assert "WEBVTT" not in text
        assert "Kind:" not in text
        assert "-->" not in text
        assert "[\u266a\u266a\u266a]" in text
        assert "We're no strangers to love" in text
        assert "You know the rules" in text
        assert "and so do I" in text

    def test_build_guidance_result(self):
        transcriber = Transcriber(output_dir="/tmp/test")
        result = transcriber._build_guidance_result(
            "https://xiaohongshu.com/123", "xiaohongshu"
        )
        assert result.method == TranscriptionMethod.UNSUPPORTED
        assert result.guidance is not None
        assert "Xiaohongshu" in result.guidance

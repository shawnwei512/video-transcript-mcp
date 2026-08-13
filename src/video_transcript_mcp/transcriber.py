"""Core transcription logic - refactored from video_transcript.py for MCP server use.

Supports three-tier transcription strategy:
  1. Platform subtitle extraction (zero cost)
  2. Whisper local transcription (offline, free)
  3. Mini-program guidance for closed platforms (Xiaohongshu/WeChat Video)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable

from .models import (
    TranscriptionMethod,
    TranscriptResult,
    TranscriptSegment,
    VideoInfo,
)

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_LANGUAGE = "zh"
SEGMENT_MINUTES = 30
COMPUTE_TYPE = "int8"
CPU_THREADS = 8
BEAM_SIZE = 5
VAD_MIN_SILENCE_MS = 500

# HuggingFace configuration
# Set HF_ENDPOINT=https://hf-mirror.com for China network optimization if needed
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Platform detection patterns (order matters: local_file before podcast)
PLATFORM_PATTERNS: dict[str, list[str]] = {
    "youtube": [r"youtube\.com", r"youtu\.be"],
    "bilibili": [r"bilibili\.com", r"b23\.tv"],
    "douyin": [r"douyin\.com", r"iesdouyin\.com"],
    "tiktok": [r"tiktok\.com"],
    "kuaishou": [r"kuaishou\.com", r"chenzhongtech\.com"],
    "xiaohongshu": [r"xiaohongshu\.com", r"xhslink\.com"],
    "weixin_video": [r"channels\.weixin\.qq\.com"],
    "weibo": [r"weibo\.com", r"weibo\.cn"],
    "local_file": [r"^/", r"^\./", r"^[A-Z]:/"],
    "podcast": [r"\.mp3$", r"\.m4a$", r"\.wav$", r"\.flac$"],
}

# Platforms that require mini-program assistance
CLOSED_PLATFORMS = {"xiaohongshu", "weixin_video"}

# Progress callback type: (progress: float, total: float | None, message: str) -> None
ProgressCallback = Callable[[float, float | None, str], None]


def detect_platform(url: str) -> str:
    """Detect the platform from a URL or file path."""
    url_lower = url.lower()
    for platform, patterns in PLATFORM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, url_lower):
                return platform
    return "unknown"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Transcriber:
    """Multi-platform video/audio transcriber with three-tier strategy.

    Args:
        output_dir: Directory for temporary files (audio downloads, segments).
        model: Whisper model name (tiny, base, small, medium, large-v3-turbo, etc.)
        language: Language code for Whisper (zh, en, ja, etc.)
        cookies_browser: Browser to extract cookies from (chrome/firefox/safari/edge).
        segment_minutes: Audio segment length in minutes for long video splitting.
            Default 30 minutes. Increase for longer videos to reduce segment count.
    """

    def __init__(
        self,
        output_dir: str = "./transcripts",
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        cookies_browser: str | None = None,
        segment_minutes: int = SEGMENT_MINUTES,
    ):
        self.output_dir = output_dir
        self.model = model
        self.language = language
        self.cookies_browser = cookies_browser
        self.segment_minutes = segment_minutes
        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Tier 1: Video metadata
    # ----------------------------------------------------------
    def get_video_info(self, url: str) -> VideoInfo:
        """Fetch video metadata using yt-dlp."""
        cmd = ["yt-dlp", "--dump-json", "--no-warnings"]
        if self.cookies_browser:
            cmd.extend(["--cookies-from-browser", self.cookies_browser])
        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode != 0:
                logger.warning("yt-dlp metadata failed: %s", result.stderr[:200])
                return VideoInfo(url=url)
            info = json.loads(result.stdout.strip())
            return VideoInfo(
                title=info.get("title", "Unknown"),
                uploader=info.get("uploader", info.get("channel", "Unknown")),
                duration=int(info.get("duration", 0) or 0),
                platform=info.get("extractor_key", "unknown").lower(),
                url=url,
            )
        except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Metadata error: %s", e)
            return VideoInfo(url=url)

    # ----------------------------------------------------------
    # Tier 2: Subtitle extraction
    # ----------------------------------------------------------
    def try_extract_subtitles(
        self, url: str, platform: str
    ) -> tuple[str | None, str | None]:
        """Try to extract platform subtitles. Returns (text, language_code) or (None, None)."""
        subtitle_dir = os.path.join(self.output_dir, "subtitles")
        os.makedirs(subtitle_dir, exist_ok=True)

        sub_langs = {
            "bilibili": "ai-zh,zh-CN,zh",
            "youtube": "zh-Hans,zh-CN,zh,en",
            "default": "zh-CN,zh,en",
        }
        target_langs = sub_langs.get(platform, sub_langs["default"])

        # Primary attempt: try all target languages with auto-subs
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--sub-langs", target_langs,
            "--skip-download",
            "--ignore-errors",
            "-o", os.path.join(subtitle_dir, "%(title)s.%(ext)s"),
        ]
        if self.cookies_browser:
            cmd.extend(["--cookies-from-browser", self.cookies_browser])
        cmd.append(url)

        subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, check=False)

        # Also try manual subs
        cmd_manual = [
            "yt-dlp",
            "--write-subs",
            "--sub-langs", target_langs,
            "--skip-download",
            "--ignore-errors",
            "-o", os.path.join(subtitle_dir, "%(title)s.%(ext)s"),
        ]
        if self.cookies_browser:
            cmd_manual.extend(["--cookies-from-browser", self.cookies_browser])
        cmd_manual.append(url)

        subprocess.run(cmd_manual, capture_output=True, text=True, cwd=self.output_dir, check=False)

        srt_files = glob.glob(os.path.join(subtitle_dir, "*.srt")) + \
                    glob.glob(os.path.join(subtitle_dir, "*.vtt"))

        if srt_files:
            text = self._parse_srt(srt_files[0])
            # Detect language from filename
            fname = os.path.basename(srt_files[0])
            if "zh" in fname or "ai-zh" in fname:
                lang = "zh"
            elif "en" in fname:
                lang = "en"
            else:
                lang = "zh" if "zh" in target_langs else "en"
            return text, lang

        # Bilibili AI subtitle special handling
        if platform == "bilibili":
            cmd_bili = [
                "yt-dlp", "--write-subs",
                "--sub-langs", "ai-zh",
                "--skip-download",
                "-o", os.path.join(subtitle_dir, "%(title)s.%(ext)s"),
            ]
            if self.cookies_browser:
                cmd_bili.extend(["--cookies-from-browser", self.cookies_browser])
            cmd_bili.append(url)
            subprocess.run(cmd_bili, capture_output=True, text=True, cwd=self.output_dir, check=False)

            srt_files = glob.glob(os.path.join(subtitle_dir, "*.srt"))
            if srt_files:
                text = self._parse_srt(srt_files[0])
                return text, "ai-zh"

        return None, None

    @staticmethod
    def _parse_srt(srt_path: str) -> str:
        """Parse SRT/VTT file into plain text, removing duplicates.

        Handles:
        - SRT sequence numbers and timestamp lines
        - VTT headers (WEBVTT, NOTE, Kind:, Language:)
        - VTT timestamp lines with attributes (e.g., align:start position:0%)
        - Inline VTT tags (e.g., <00:00:19.008>, <c>, </c>)
        - Duplicate lines (common in auto-generated subtitles)
        """
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Pattern for VTT timestamp lines (with optional attributes like "align:start position:0%")
        vtt_timestamp_re = re.compile(
            r"^\d{2}:\d{2}:\d{2}[\.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[\.,]\d{3}"
        )

        lines = content.split("\n")
        text_lines = []
        for line in lines:
            line = line.strip()
            # Skip SRT sequence numbers
            if re.match(r"^\d+$", line):
                continue
            # Skip simple SRT/VTT timestamp lines
            if re.match(r"^[\d:,\.\->\s]+$", line):
                continue
            # Skip VTT timestamp lines with attributes (e.g., "00:00:00.320 --> 00:00:18.790 align:start position:0%")
            if vtt_timestamp_re.match(line):
                continue
            # Skip VTT headers
            if line.startswith(("WEBVTT", "NOTE")):
                continue
            # Skip VTT metadata headers (Kind:, Language:, etc.)
            if re.match(r"^[A-Z][a-z]+:", line):
                continue
            if not line:
                continue
            # Strip inline VTT tags (e.g., <00:00:19.008>, <c>, </c>, <i>, </i>)
            line = re.sub(r"<[^>]+>", "", line).strip()
            if not line:
                continue
            text_lines.append(line)

        # Remove duplicates while preserving order
        unique_lines = []
        seen: set[str] = set()
        for line in text_lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

        return "\n".join(unique_lines)

    # ----------------------------------------------------------
    # Tier 3: Whisper transcription
    # ----------------------------------------------------------
    def download_audio(self, url: str) -> str | None:
        """Download audio from URL using yt-dlp."""
        audio_base = os.path.join(self.output_dir, "audio")

        cmd = [
            "yt-dlp", "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", audio_base + ".%(ext)s",
            "--no-warnings",
        ]
        if self.cookies_browser:
            cmd.extend(["--cookies-from-browser", self.cookies_browser])
        cmd.append(url)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            # Try m4a fallback (change audio format from mp3 to m4a)
            cmd[3] = "m4a"
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.error("Audio download failed: %s", result.stderr[:300])
                return None

        for ext in [".mp3", ".m4a", ".webm", ".opus"]:
            path = audio_base + ext
            if os.path.exists(path):
                return path

        audio_files = glob.glob(os.path.join(self.output_dir, "audio.*"))
        return audio_files[0] if audio_files else None

    def split_audio(self, audio_path: str, segment_minutes: int = SEGMENT_MINUTES) -> list[str] | None:
        """Split audio file into segments using FFmpeg."""
        seg_dir = os.path.join(self.output_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        pattern = os.path.join(seg_dir, "seg_%04d.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-f", "segment",
            "-segment_time", str(segment_minutes * 60),
            "-reset_timestamps", "1",
            pattern,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("FFmpeg split failed: %s", result.stderr[:300])
            return None

        segments = sorted(glob.glob(os.path.join(seg_dir, "seg_*.wav")))
        return segments if segments else None

    def transcribe_with_whisper(
        self,
        audio_path: str,
        progress_cb: ProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio file using faster-whisper.

        Automatically splits long audio into segments and handles checkpoint resume.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install with: pip install 'video-transcript-mcp[whisper]'"
            )

        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        logger.info("Audio duration: %.1f minutes", duration / 60)

        need_split = duration > self.segment_minutes * 60

        if progress_cb:
            progress_cb(0, None, f"Loading Whisper model ({self.model})...")

        t0 = time.time()
        model = WhisperModel(
            self.model,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
        )
        logger.info("Model loaded in %.1fs", time.time() - t0)

        all_segments: list[TranscriptSegment] = []

        if need_split:
            seg_files = self.split_audio(audio_path, self.segment_minutes)
            if not seg_files:
                raise RuntimeError("Audio splitting failed")

            total_segs = len(seg_files)
            for idx, seg_file in enumerate(seg_files):
                if progress_cb:
                    msg = f"Transcribing segment {idx + 1}/{total_segs}..."
                    progress_cb(idx, total_segs, msg)

                global_offset = idx * self.segment_minutes * 60
                seg_results = self._transcribe_segment(model, seg_file)
                for r in seg_results:
                    r.start += global_offset
                    r.end += global_offset
                    all_segments.append(r)

                # Save checkpoint for resume support
                self._save_checkpoint(all_segments, idx + 1)
        else:
            if progress_cb:
                progress_cb(0, None, "Transcribing audio...")
            all_segments = self._transcribe_segment(model, audio_path)

        if progress_cb:
            progress_cb(1.0, 1.0, f"Transcription complete: {len(all_segments)} segments")

        return all_segments

    def _transcribe_segment(self, model, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe a single audio segment."""
        segments, _info = model.transcribe(
            audio_path,
            language=self.language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
            condition_on_previous_text=True,
            temperature=0.0,
        )

        results: list[TranscriptSegment] = []
        for seg in segments:
            results.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))
        return results

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            return float(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
            return 0.0

    def _save_checkpoint(self, segments: list[TranscriptSegment], seg_num: int) -> None:
        """Save checkpoint for resume support."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{seg_num:03d}.json")
        data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def transcribe_url(
        self,
        url: str,
        skip_subtitles: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> TranscriptResult:
        """Transcribe a video from URL using the three-tier strategy.

        Tier 1: Extract platform subtitles (zero cost)
        Tier 2: Download audio + Whisper transcription (offline free)
        Tier 3: Return guidance for closed platforms (mini-program)

        Args:
            url: Video URL (YouTube, Bilibili, Douyin, etc.)
            skip_subtitles: Skip subtitle extraction, go straight to Whisper.
            progress_cb: Optional callback for progress updates.

        Returns:
            TranscriptResult with segments, text, and metadata.
        """
        platform = detect_platform(url)
        logger.info("Detected platform: %s", platform)

        if progress_cb:
            progress_cb(0, None, f"Detected platform: {platform}")

        # Check for closed platforms that need mini-program assistance
        if platform in CLOSED_PLATFORMS:
            return self._build_guidance_result(url, platform)

        # Get video metadata
        if progress_cb:
            progress_cb(0, None, "Fetching video metadata...")
        video_info = self.get_video_info(url)
        logger.info("Title: %s, Duration: %ds", video_info.title, video_info.duration)

        # Tier 1: Try subtitle extraction
        if not skip_subtitles:
            if progress_cb:
                progress_cb(0, None, "Extracting subtitles...")
            sub_text, sub_lang = self.try_extract_subtitles(url, platform)
            if sub_text:
                if progress_cb:
                    progress_cb(1.0, 1.0, "Subtitle extraction complete")
                return TranscriptResult(
                    video_info=video_info,
                    method=TranscriptionMethod.SUBTITLE,
                    full_text=sub_text,
                    subtitle_language=sub_lang,
                )

        # Tier 2: Whisper transcription
        if progress_cb:
            progress_cb(0, None, "Downloading audio...")
        audio_path = self.download_audio(url)
        if not audio_path:
            return TranscriptResult(
                video_info=video_info,
                method=TranscriptionMethod.WHISPER,
                warning="Audio download failed. Check URL or use cookies_browser option.",
            )

        segments = self.transcribe_with_whisper(audio_path, progress_cb)

        full_text = "\n".join(seg.text for seg in segments)
        return TranscriptResult(
            video_info=video_info,
            method=TranscriptionMethod.WHISPER,
            segments=segments,
            full_text=full_text,
            whisper_model=self.model,
        )

    def transcribe_file(
        self,
        file_path: str,
        progress_cb: ProgressCallback | None = None,
    ) -> TranscriptResult:
        """Transcribe a local audio/video file using Whisper.

        Args:
            file_path: Path to local audio/video file (mp3, mp4, wav, etc.)
            progress_cb: Optional callback for progress updates.

        Returns:
            TranscriptResult with segments, text, and metadata.
        """
        if not os.path.exists(file_path):
            return TranscriptResult(
                video_info=VideoInfo(title="File not found", url=file_path),
                method=TranscriptionMethod.WHISPER,
                warning=f"File not found: {file_path}",
            )

        # Extract filename as title
        title = os.path.splitext(os.path.basename(file_path))[0]
        video_info = VideoInfo(
            title=title,
            platform="local_file",
            url=file_path,
            duration=int(self._get_audio_duration(file_path)),
        )

        segments = self.transcribe_with_whisper(file_path, progress_cb)
        full_text = "\n".join(seg.text for seg in segments)

        return TranscriptResult(
            video_info=video_info,
            method=TranscriptionMethod.WHISPER,
            segments=segments,
            full_text=full_text,
            whisper_model=self.model,
        )

    def _build_guidance_result(self, url: str, platform: str) -> TranscriptResult:
        """Build a guidance result for closed platforms."""
        platform_names = {
            "xiaohongshu": "Xiaohongshu (\u5c0f\u7ea2\u4e66)",
            "weixin_video": "WeChat Video Channel (\u5fae\u4fe1\u89c6\u9891\u53f7)",
        }
        name = platform_names.get(platform, platform)

        guidance = (
            f"This platform ({name}) restricts direct video download.\n\n"
            "To transcribe videos from this platform:\n"
            "1. Open WeChat and search for a mini-program like:\n"
            "   - '\u6587\u6848\u63d0\u53d6\u5b9d' (Transcript Extractor)\n"
            "   - '\u5a92\u5c0f\u4e09AI\u521b\u4f5c' (Media AI Assistant)\n"
            "   - 'Get\u7b14\u8bb0' (Get Notes)\n"
            "2. Copy the video share link and paste it into the mini-program\n"
            "3. Copy the extracted transcript and use the transcribe_file tool\n"
            "   with the transcript text, or download the video manually\n"
            "   and use transcribe_file with the local file path."
        )

        return TranscriptResult(
            video_info=VideoInfo(title=f"{name} video", platform=platform, url=url),
            method=TranscriptionMethod.UNSUPPORTED,
            guidance=guidance,
        )

    def save_transcript_files(
        self,
        result: TranscriptResult,
        output_dir: str | None = None,
    ) -> dict[str, str]:
        """Save transcript result to text files.

        Returns dict with paths to saved files:
        - 'plain_text': Plain text transcript
        - 'timestamped': Timestamped transcript
        - 'json': Full JSON result
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        safe_title = re.sub(r'[\\/:*?"<>|]', "_", result.video_info.title)[:50]

        # Plain text
        plain_path = os.path.join(out_dir, f"{safe_title}_transcript.txt")
        with open(plain_path, "w", encoding="utf-8") as f:
            f.write(f"# {result.video_info.title}\n")
            f.write(f"Author: {result.video_info.uploader}\n")
            f.write(f"Source: {result.video_info.url}\n\n")
            f.write(result.full_text)

        # Timestamped (only if segments exist)
        ts_path = os.path.join(out_dir, f"{safe_title}_timestamped.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {result.video_info.title}\n")
            f.write(f"Author: {result.video_info.uploader}\n")
            f.write(f"Source: {result.video_info.url}\n")
            f.write("=" * 60 + "\n\n")
            f.writelines(f"[{_format_timestamp(seg.start)}] {seg.text}\n" for seg in result.segments)

        # JSON
        json_path = os.path.join(out_dir, f"{safe_title}_transcript.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        return {
            "plain_text": plain_path,
            "timestamped": ts_path,
            "json": json_path,
        }

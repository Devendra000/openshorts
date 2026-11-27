"""
ElevenLabs Video Translation/Dubbing Module

Uses ElevenLabs Dubbing API to translate video audio to different languages.
"""

import os
import re
import subprocess
import time
import httpx
from typing import Optional

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Supported target languages for dubbing
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "id": "Indonesian",
    "fil": "Filipino",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "ta": "Tamil",
}

# HTTP status codes worth retrying (server-side / rate-limit hiccups).
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class _TransientHTTPError(Exception):
    """Raised for retryable HTTP responses (5xx / 429)."""


def _with_retry(operation, description, max_attempts=3, base_delay=2.0):
    """
    Run `operation` (a zero-arg callable) with exponential backoff on transient
    failures: HTTP 5xx/429, timeouts, and connection errors. Backoff is ~2s/4s/8s.
    Non-transient errors (e.g. 4xx) propagate immediately and fail fast.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except (_TransientHTTPError, httpx.TransportError) as e:
            if attempt >= max_attempts:
                raise Exception(f"{description} failed after {max_attempts} attempts: {e}")
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"[ElevenLabs] {description} transient error "
                f"(attempt {attempt}/{max_attempts}): {e}. Retrying in {int(delay)}s..."
            )
            time.sleep(delay)


def create_dubbing_project(
    video_path: str,
    target_language: str,
    api_key: str,
    source_language: Optional[str] = None,
) -> dict:
    """
    Create a new dubbing project with ElevenLabs.

    Args:
        video_path: Path to the video file
        target_language: Target language code (e.g., 'es', 'fr', 'de')
        api_key: ElevenLabs API key
        source_language: Source language code (auto-detected if None)

    Returns:
        dict with dubbing_id and expected_duration_sec
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing"

    headers = {
        "xi-api-key": api_key,
    }

    # Prepare form data
    data = {
        "target_lang": target_language,
        "mode": "automatic",
        "num_speakers": "0",
        "watermark": "false",
    }

    if source_language:
        data["source_lang"] = source_language

    def _submit():
        # Re-open the file per attempt so a retry uploads from the start.
        with open(video_path, "rb") as video_file:
            files = {
                "file": (os.path.basename(video_path), video_file, "video/mp4")
            }
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(url, headers=headers, data=data, files=files)
        if resp.status_code in _TRANSIENT_STATUS:
            raise _TransientHTTPError(f"HTTP {resp.status_code}")
        return resp

    print(f"[ElevenLabs] Creating dubbing project for {target_language}...")
    response = _with_retry(_submit, "create dubbing project")

    if response.status_code not in [200, 201]:
        error_msg = response.text
        try:
            error_data = response.json()
            error_msg = error_data.get("detail", {}).get("message", response.text)
        except (ValueError, AttributeError, KeyError, TypeError):
            pass
        raise Exception(f"ElevenLabs API error ({response.status_code}): {error_msg}")

    result = response.json()
    print(f"[ElevenLabs] Dubbing project created: {result.get('dubbing_id')}")
    return result


def get_dubbing_status(dubbing_id: str, api_key: str) -> dict:
    """
    Check the status of a dubbing project.

    Returns:
        dict with status ('dubbing', 'dubbed', 'failed') and other metadata
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}"

    headers = {
        "xi-api-key": api_key,
    }

    def _poll():
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code in _TRANSIENT_STATUS:
            raise _TransientHTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    response = _with_retry(_poll, "dubbing status check")

    if response.status_code != 200:
        raise Exception(f"Failed to get dubbing status: {response.text}")

    return response.json()


def download_dubbed_video(
    dubbing_id: str,
    target_language: str,
    output_path: str,
    api_key: str
) -> str:
    """
    Download the dubbed video file.

    Args:
        dubbing_id: The dubbing project ID
        target_language: Target language code
        output_path: Where to save the dubbed video
        api_key: ElevenLabs API key

    Returns:
        Path to the downloaded file
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}/audio/{target_language}"

    headers = {
        "xi-api-key": api_key,
    }

    def _download():
        with httpx.Client(timeout=120.0) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code in _TRANSIENT_STATUS:
                    raise _TransientHTTPError(f"HTTP {response.status_code}")
                if response.status_code != 200:
                    # Body must be read before .text is available on a stream.
                    response.read()
                    raise Exception(f"Failed to download dubbed video: {response.text}")

                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
        return output_path

    print(f"[ElevenLabs] Downloading dubbed video...")
    _with_retry(_download, "dubbed video download")

    print(f"[ElevenLabs] Dubbed video saved to: {output_path}")
    return output_path


def translate_video(
    video_path: str,
    output_path: str,
    target_language: str,
    api_key: str,
    source_language: Optional[str] = None,
    max_wait_seconds: int = 600,
    poll_interval: int = 5,
) -> str:
    """
    Translate a video to a target language using ElevenLabs dubbing.

    This is a blocking call that waits for the dubbing to complete.

    Args:
        video_path: Path to input video
        output_path: Path to save translated video
        target_language: Target language code
        api_key: ElevenLabs API key
        source_language: Source language code (auto-detected if None)
        max_wait_seconds: Maximum time to wait for dubbing (default 10 min)
        poll_interval: Seconds between status checks

    Returns:
        Path to the translated video
    """
    # Create dubbing project
    project = create_dubbing_project(
        video_path=video_path,
        target_language=target_language,
        api_key=api_key,
        source_language=source_language,
    )

    dubbing_id = project["dubbing_id"]
    expected_duration = project.get("expected_duration_sec", 60)

    print(f"[ElevenLabs] Dubbing ID: {dubbing_id}, Expected duration: {expected_duration}s")

    # Poll for completion
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise Exception(f"Dubbing timed out after {max_wait_seconds} seconds")

        status = get_dubbing_status(dubbing_id, api_key)
        current_status = status.get("status", "unknown")

        print(f"[ElevenLabs] Status: {current_status} (elapsed: {int(elapsed)}s)")

        if current_status == "dubbed":
            # Download the result
            return download_dubbed_video(
                dubbing_id=dubbing_id,
                target_language=target_language,
                output_path=output_path,
                api_key=api_key,
            )

        elif current_status == "failed":
            error = status.get("error", "Unknown error")
            raise Exception(f"Dubbing failed: {error}")

        # Still processing, wait and poll again
        time.sleep(poll_interval)


def get_supported_languages() -> dict:
    """Return dict of supported language codes and names."""
    return SUPPORTED_LANGUAGES.copy()


# ---------------------------------------------------------------------------
# Script voiceover: your own words, spoken by an ElevenLabs voice, laid over
# the clip. Text-to-speech, not dubbing: the clip needs no speech of its own.
# ---------------------------------------------------------------------------

# The voice id goes into the request path, so only the shape ElevenLabs uses.
_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9]{8,64}$")
MAX_SCRIPT_CHARS = 5000
# A voice a little longer than the clip is sped up to fit; past this it would
# sound rushed, so the last frame is held instead for what is left.
MAX_SPEEDUP = 1.15
# The clip's own audio under the voice when it is kept.
DUCK_VOLUME = 0.15
VOICEOVER_PREFIX = "translated_vo_"
# eleven_v3 performs inline audio tags ("[excited] Now watch!"); v2 would read
# them aloud, so they are removed for it.
TTS_MODELS = ("eleven_v3", "eleven_multilingual_v2")
DEFAULT_TTS_MODEL = "eleven_v3"
# Silence kept between two timed lines when one runs into the next.
MIN_LINE_GAP = 0.15

_TIME = r"(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?"
_TIMED_LINE_RE = re.compile(
    rf"^\s*[\[(]?\s*({_TIME})(?:\s*(?:-|\u2013|\u2014|to)\s*({_TIME}))?\s*[\])]?\s*[:\-\u2013\u2014]?\s*(.*)$")
_TAG_RE = re.compile(r"\[[^\[\]\n]{1,40}\]")


class TTSError(Exception):
    def __init__(self, status, message):
        super().__init__(f"ElevenLabs TTS error ({status}): {message}")
        self.status = status


def strip_audio_tags(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _TAG_RE.sub("", text)).strip()


def _seconds(stamp: str) -> float:
    total = 0.0
    for part in stamp.split(":"):
        total = total * 60 + float(part)
    return total


def parse_timed_script(script: str):
    """Lines that start with a clip time ("0:04", "0:04-0:10", "0:04–0:10")
    as [{"start", "end", "text"}], or None when no line has a time (a plain
    script, spoken in one go). An untimed line continues the line above it."""
    lines, timed = [], False
    for raw in (script or "").splitlines():
        if not raw.strip():
            continue
        m = _TIMED_LINE_RE.match(raw)
        if m:
            timed = True
            end = _seconds(m.group(2)) if m.group(2) else None
            lines.append({"start": _seconds(m.group(1)), "end": end,
                          "text": m.group(3).strip()})
        elif lines:
            lines[-1]["text"] = f"{lines[-1]['text']} {raw.strip()}".strip()
        else:
            lines.append({"start": 0.0, "end": None, "text": raw.strip()})
    if not timed:
        return None
    lines = [ln for ln in lines if strip_audio_tags(ln["text"])]
    return sorted(lines, key=lambda ln: ln["start"])


def place_timed_lines(lines, durations, max_speedup: float = MAX_SPEEDUP):
    """[(start, tempo)] per line: each starts at its own time, sped up (to
    max_speedup) when it is longer than its slot, and pushed later rather
    than spoken over the line before it."""
    placed, free_at = [], 0.0
    for i, (line, dur) in enumerate(zip(lines, durations)):
        start = max(line["start"], free_at)
        slot_end = line.get("end")
        if i + 1 < len(lines):
            nxt = lines[i + 1]["start"]
            slot_end = nxt if slot_end is None else min(slot_end, nxt)
        tempo = 1.0
        if slot_end is not None and slot_end > start and dur > slot_end - start:
            tempo = min(dur / (slot_end - start), max_speedup)
        placed.append((start, tempo))
        free_at = start + dur / tempo + MIN_LINE_GAP
    return placed


def valid_voice_id(voice_id) -> bool:
    return isinstance(voice_id, str) and bool(_VOICE_ID_RE.match(voice_id))


def synthesize_speech(script: str, voice_id: str, api_key: str, output_path: str,
                      model_id: str = DEFAULT_TTS_MODEL) -> str:
    """ElevenLabs text-to-speech of ``script`` into an mp3 at ``output_path``."""
    if not valid_voice_id(voice_id):
        raise ValueError("invalid voice id")
    if model_id not in TTS_MODELS:
        raise ValueError("unknown TTS model")
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    if model_id == "eleven_v3":
        # v3 takes only a stability preset (0 creative, 0.5 natural, 1 robust).
        settings = {"stability": 0.5}
    else:
        script = strip_audio_tags(script)
        settings = {"stability": 0.5, "similarity_boost": 0.75,
                    "style": 0.3, "use_speaker_boost": True}
    body = {"text": script, "model_id": model_id, "voice_settings": settings}

    def _submit():
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, headers=headers, json=body,
                               params={"output_format": "mp3_44100_128"})
        if resp.status_code in _TRANSIENT_STATUS:
            raise _TransientHTTPError(f"HTTP {resp.status_code}")
        return resp

    print(f"[ElevenLabs] Synthesising voiceover ({len(script)} chars)...")
    resp = _with_retry(_submit, "text-to-speech")
    if resp.status_code != 200:
        msg = resp.text
        try:
            detail = resp.json().get("detail")
            msg = detail.get("message", msg) if isinstance(detail, dict) else (detail or msg)
        except (ValueError, AttributeError):
            pass
        raise TTSError(resp.status_code, msg)
    with open(output_path, "wb") as f:
        f.write(resp.content)
    return output_path


def _probe(path: str, args: list) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of",
                          "default=noprint_wrappers=1:nokey=1", path],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def media_duration(path: str) -> float:
    try:
        return float(_probe(path, ["-show_entries", "format=duration"]).splitlines()[0])
    except (ValueError, IndexError):
        raise Exception(f"could not read the duration of {os.path.basename(path)}")


def has_audio(path: str) -> bool:
    return bool(_probe(path, ["-select_streams", "a", "-show_entries", "stream=index"]))


def voiceover_plan(video_seconds: float, voice_seconds: float,
                   max_speedup: float = MAX_SPEEDUP):
    """(tempo, hold_seconds): how to fit the voice to the clip.

    Shorter voice: as it is (silence after it). Up to MAX_SPEEDUP longer:
    sped up to end with the clip. Longer still: MAX_SPEEDUP, and the last
    frame is held for what is left so no word is cut."""
    if video_seconds <= 0 or voice_seconds <= video_seconds:
        return 1.0, 0.0
    ratio = voice_seconds / video_seconds
    if ratio <= max_speedup:
        return ratio, 0.0
    return max_speedup, voice_seconds / max_speedup - video_seconds


ORIGINAL_AUDIO_MODES = ("duck", "mix", "replace")
MAX_UPLOAD_AUDIO_BYTES = 50 * 1024 * 1024
UPLOAD_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg",
                                     ".oga", ".opus", ".flac", ".webm"})


def build_voiceover_command(video_path: str, voice_path: str, output_path: str,
                            video_seconds: float, voice_seconds: float,
                            original_audio: str, source_has_audio: bool,
                            fit_voice: bool = True, max_speedup: float = MAX_SPEEDUP,
                            reencode_video: bool = False) -> list:
    """The ffmpeg command that lays ``voice_path`` over the clip.

    original_audio: ``duck`` (clip audio quietly underneath), ``mix`` (both
    at full level, e.g. a music bed) or ``replace``. ``fit_voice`` speeds up /
    holds the last frame so a long voice is not cut (generated speech); off,
    the added audio is simply cut at the clip's end (uploaded tracks)."""
    from ffmpeg_utils import LOUDNORM_FILTER, video_encode_args

    if original_audio not in ORIGINAL_AUDIO_MODES:
        raise ValueError(f"original_audio must be one of {ORIGINAL_AUDIO_MODES}")
    tempo, hold = (voiceover_plan(video_seconds, voice_seconds, max_speedup)
                   if fit_voice else (1.0, 0.0))
    total = video_seconds + hold
    graph = [f"[1:a]aresample=48000,atempo={tempo:.4f},apad[vo]"]
    if original_audio != "replace" and source_has_audio:
        level = DUCK_VOLUME if original_audio == "duck" else 1.0
        graph.append(f"[0:a]aresample=48000,volume={level},apad[bg]")
        graph.append("[bg][vo]amix=inputs=2:duration=longest:normalize=0[mix]")
        last = "mix"
    else:
        last = "vo"
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        graph.append(f"[{last}]{LOUDNORM_FILTER}[aout]")
    else:
        graph.append(f"[{last}]anull[aout]")
    if hold > 0:
        graph.append(f"[0:v]tpad=stop_mode=clone:stop_duration={hold:.3f}[vout]")
        video_map, video_codec = "[vout]", video_encode_args()
    elif reencode_video:
        video_map, video_codec = "0:v:0", video_encode_args()
    else:
        video_map, video_codec = "0:v:0", ["-c:v", "copy"]

    return ["ffmpeg", "-y", "-i", video_path, "-i", voice_path,
            "-filter_complex", ";".join(graph),
            "-map", video_map, "-map", "[aout]",
            *video_codec, "-c:a", "aac", "-b:a", "192k",
            "-t", f"{total:.3f}", "-movflags", "+faststart", output_path]


def mix_audio_into_video(video_path: str, audio_path: str, output_path: str,
                         original_audio: str = "duck", fit_voice: bool = False,
                         max_speedup: float = MAX_SPEEDUP) -> str:
    """Lay ``audio_path`` over the clip (see build_voiceover_command)."""
    if not has_audio(audio_path):
        raise ValueError("the uploaded file has no audio track")
    tmp_out = output_path + ".tmp.mp4"
    try:
        args = (video_path, audio_path, tmp_out,
                media_duration(video_path), media_duration(audio_path),
                original_audio, has_audio(video_path))
        cmd = build_voiceover_command(*args, fit_voice=fit_voice, max_speedup=max_speedup)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0 and "copy" in cmd:
            # A video codec the mp4 container cannot carry as is (an upload
            # from elsewhere): encode it instead of copying.
            cmd = build_voiceover_command(*args, fit_voice=fit_voice,
                                          max_speedup=max_speedup, reencode_video=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0 or not os.path.exists(tmp_out):
            raise Exception(f"ffmpeg audio mix failed: {result.stderr[-500:]}")
        os.replace(tmp_out, output_path)
    finally:
        try:
            os.remove(tmp_out)
        except OSError:
            pass
    return output_path


MAX_AUDIO_PIECES = 50
MAX_UPLOAD_VIDEO_BYTES = 1024 * 1024 * 1024
UPLOAD_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv"})
_TIME_TOKEN_RE = re.compile(_TIME)


def parse_audio_placements(text: str):
    """Pieces of an uploaded track placed in the clip, one per line:
    ``0:00-0:05 -> 0:05-0:10`` (audio 0:00-0:05 plays at clip 0:05-0:10) or
    ``0:00-0:05 -> 0:05`` (clip end optional). Only the times are read, so
    "audio 0:00 - 0:05 in video 0:05" works too. A video range shorter than
    the audio range cuts the piece; nothing is stretched. Lines without a
    time (headings of a pasted table) are skipped.

    Returns [{"a_start", "a_end", "v_start"}] or None when ``text`` is blank.
    Raises ValueError naming the line it cannot read."""
    pieces = []
    for n, raw in enumerate((text or "").splitlines(), start=1):
        if not raw.strip():
            continue
        times = [_seconds(t) for t in _TIME_TOKEN_RE.findall(raw)]
        if not times:
            continue  # a heading or a note ("Here is the placement map:")
        if len(times) not in (3, 4):
            raise ValueError(f"placement line {n}: write it as 0:00-0:05 -> 0:05-0:10")
        a_start, a_end, v_start = times[:3]
        if a_end <= a_start:
            raise ValueError(f"placement line {n}: the audio range ends before it starts")
        if len(times) == 4:
            if times[3] <= v_start:
                raise ValueError(f"placement line {n}: the video range ends before it starts")
            a_end = min(a_end, a_start + times[3] - v_start)
        pieces.append({"a_start": a_start, "a_end": a_end, "v_start": v_start})
    if len(pieces) > MAX_AUDIO_PIECES:
        raise ValueError(f"at most {MAX_AUDIO_PIECES} placement lines")
    return pieces or None


def build_pieces_track_command(audio_path: str, pieces, output_path: str) -> list:
    """ffmpeg command cutting ``pieces`` out of one track and placing each
    at its clip time (silence everywhere else)."""
    n = len(pieces)
    graph = [f"[0:a]aresample=48000,asplit={n}" + "".join(f"[s{k}]" for k in range(n))
             if n > 1 else "[0:a]aresample=48000,anull[s0]"]
    labels = []
    for k, p in enumerate(pieces):
        ms = int(round(p["v_start"] * 1000))
        graph.append(f"[s{k}]atrim=start={p['a_start']:.3f}:end={p['a_end']:.3f},"
                     f"asetpts=PTS-STARTPTS,adelay={ms}:all=1[p{k}]")
        labels.append(f"[p{k}]")
    graph.append(f"{''.join(labels)}amix=inputs={n}:duration=longest:normalize=0[out]")
    return ["ffmpeg", "-y", "-i", audio_path, "-filter_complex", ";".join(graph),
            "-map", "[out]", "-c:a", "pcm_s16le", output_path]


def mix_audio_pieces(video_path: str, audio_path: str, output_path: str, pieces,
                     original_audio: str = "duck") -> str:
    """Place parts of an uploaded track at chosen clip times, then mix."""
    if not has_audio(audio_path):
        raise ValueError("the uploaded file has no audio track")
    track = output_path + ".pieces.wav"
    try:
        result = subprocess.run(build_pieces_track_command(audio_path, pieces, track),
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0 or not os.path.exists(track):
            raise Exception(f"ffmpeg audio pieces failed: {result.stderr[-500:]}")
        return mix_audio_into_video(video_path, track, output_path,
                                    original_audio=original_audio, fit_voice=False)
    finally:
        try:
            os.remove(track)
        except OSError:
            pass


def _speak(text, voice_id, api_key, path, model_id):
    """synthesize_speech, falling back from v3 to v2 when the account or the
    voice cannot use v3 (v2 then drops the audio tags)."""
    try:
        return synthesize_speech(text, voice_id, api_key, path, model_id)
    except TTSError as e:
        if model_id != "eleven_v3" or e.status in (401, 402):
            raise
        print(f"[ElevenLabs] v3 refused ({e.status}), retrying with multilingual v2")
        return synthesize_speech(text, voice_id, api_key, path, "eleven_multilingual_v2")


def build_timed_track_command(line_paths, placed, output_path) -> list:
    """ffmpeg command putting each line's audio at its start (and tempo)."""
    cmd = ["ffmpeg", "-y"]
    for path in line_paths:
        cmd += ["-i", path]
    graph, labels = [], []
    for k, (start, tempo) in enumerate(placed):
        ms = int(round(start * 1000))
        graph.append(f"[{k}:a]aresample=48000,atempo={tempo:.4f},adelay={ms}:all=1[l{k}]")
        labels.append(f"[l{k}]")
    graph.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[out]")
    return cmd + ["-filter_complex", ";".join(graph), "-map", "[out]",
                  "-c:a", "pcm_s16le", output_path]


def voiceover_video(video_path: str, output_path: str, script: str, voice_id: str,
                    api_key: str, original_audio: str = "duck",
                    model_id: str = DEFAULT_TTS_MODEL) -> str:
    """Speak ``script`` with an ElevenLabs voice over the clip at ``video_path``.

    A plain script is spoken in one go and fitted to the clip. A timed script
    ("0:04–0:10 [excited] line") is spoken line by line, each line placed at
    its own time with silence between; a line longer than its slot is sped
    up a little, and the last frame is held if the voice outlasts the clip."""
    script = (script or "").strip()
    if not script:
        raise ValueError("the script is empty")
    if len(script) > MAX_SCRIPT_CHARS:
        raise ValueError(f"the script is longer than {MAX_SCRIPT_CHARS} characters")
    if model_id not in TTS_MODELS:
        raise ValueError("unknown TTS model")

    lines = parse_timed_script(script)
    if lines is not None and not lines:
        raise ValueError("the timed script has no words to speak")
    scratch = []
    try:
        if lines is None:
            voice_path = output_path + ".voice.mp3"
            scratch.append(voice_path)
            _speak(script, voice_id, api_key, voice_path, model_id)
            mix_audio_into_video(video_path, voice_path, output_path,
                                 original_audio=original_audio, fit_voice=True)
        else:
            from concurrent.futures import ThreadPoolExecutor
            line_paths = [f"{output_path}.line{k}.mp3" for k in range(len(lines))]
            scratch += line_paths
            print(f"[ElevenLabs] Timed script: {len(lines)} lines")
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda kv: _speak(kv[1]["text"], voice_id, api_key,
                                                line_paths[kv[0]], model_id),
                              enumerate(lines)))
            placed = place_timed_lines(lines, [media_duration(p) for p in line_paths])
            track = output_path + ".track.wav"
            scratch.append(track)
            result = subprocess.run(build_timed_track_command(line_paths, placed, track),
                                    capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(f"ffmpeg timed track failed: {result.stderr[-500:]}")
            # Lines are already timed: no global speed-up, only the hold.
            mix_audio_into_video(video_path, track, output_path,
                                 original_audio=original_audio, fit_voice=True,
                                 max_speedup=1.0)
    finally:
        for p in scratch:
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"[ElevenLabs] Voiceover ready: {output_path}")
    return output_path

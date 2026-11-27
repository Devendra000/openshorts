"""Local WhisperX transcription provider (``TRANSCRIBE_BACKEND=whisperx``).

WhisperX lives in its own virtual environment (its torch / pyannote stack does
not fit next to this app's), so it is never imported here: it runs as a
subprocess through its own CLI, ``WHISPERX_BIN``. Its JSON is turned into the
transcript contract of ``transcribe_backends`` (absolute file timestamps,
leading-space words, native floats), so clip cutting and karaoke captions
consume it exactly like a faster-whisper transcript.

Timeline: the audio is extracted with its leading gap padded with silence
(``aresample=async=1:first_pts=0``), so second 0 of the wav is second 0 of the
video as ffmpeg's ``-ss`` sees it, even when the audio track starts late.

Configuration (env):
  WHISPERX_BIN            executable (path, or a name on PATH). Default "whisperx".
  WHISPERX_MODEL          whisper model name. Default "small".
  WHISPERX_DEVICE         cpu | cuda. Default "cpu".
  WHISPERX_COMPUTE_TYPE   int8 | float16 | float32 | default. Default "int8".
  WHISPERX_LANGUAGE       ISO code; empty detects the language.
  WHISPERX_BATCH_SIZE     default 8 (lower it if a GPU runs out of memory).
  WHISPERX_THREADS        CPU threads; default as WHISPER_CPU_THREADS.
  WHISPERX_ALIGN_MODEL    alignment model override; empty = WhisperX default.
  WHISPERX_VAD_METHOD     pyannote | silero; empty = WhisperX default.
  WHISPERX_CACHE_DIR      where models are downloaded (HF_HOME, TORCH_HOME,
                          NLTK_DATA under it); empty = the user's ~/.cache.
  WHISPERX_MODEL_CACHE_ONLY  1 = never download, fail if a model is missing.
  WHISPERX_OUTPUT_DIR     where WhisperX's raw JSON is kept; empty = a
                          ``.whisperx/`` folder next to the media.
  WHISPERX_TIMEOUT_SECONDS   hard limit; empty = 600 s + 4x (CPU) or 1x (GPU)
                          the media length.
  WHISPERX_FALLBACK       whisper | parakeet | none: provider used when
                          WhisperX fails (read by transcribe_backends).
"""
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import transcribe_backends as tb
from subtitles import merge_continuation_words


class WhisperXError(RuntimeError):
    """WhisperX ran and failed, or produced output that cannot be used."""


class WhisperXUnavailable(WhisperXError):
    """The WhisperX executable is missing or not runnable."""


class WhisperXTimeout(WhisperXError):
    """WhisperX ran past its time limit and was stopped."""


def _env(name, default=""):
    return (os.environ.get(name) or "").strip() or default


def _int_env(name, default):
    try:
        value = int(_env(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def config():
    """The provider's settings, read from the environment on every call."""
    device = _env("WHISPERX_DEVICE", "cpu").lower()
    return {
        "bin": _env("WHISPERX_BIN", "whisperx"),
        "model": _env("WHISPERX_MODEL", "small"),
        "device": device,
        "compute_type": _env("WHISPERX_COMPUTE_TYPE", "int8").lower(),
        "language": _env("WHISPERX_LANGUAGE").lower(),
        "batch_size": _int_env("WHISPERX_BATCH_SIZE", 8),
        "threads": _int_env("WHISPERX_THREADS", tb.whisper_cpu_threads()),
        "align_model": _env("WHISPERX_ALIGN_MODEL"),
        "vad_method": _env("WHISPERX_VAD_METHOD").lower(),
        "cache_dir": _env("WHISPERX_CACHE_DIR"),
        "cache_only": _env("WHISPERX_MODEL_CACHE_ONLY").lower() in ("1", "true", "yes"),
        "output_dir": _env("WHISPERX_OUTPUT_DIR"),
        "timeout": _env("WHISPERX_TIMEOUT_SECONDS"),
    }


def fallback_backend():
    """Provider to use when WhisperX fails: whisper (default), parakeet, none."""
    value = _env("WHISPERX_FALLBACK", "whisper").lower()
    return value if value in ("whisper", "parakeet", "none") else "whisper"


def resolve_executable(name):
    """Absolute path of the WhisperX executable, or WhisperXUnavailable."""
    path = os.path.expanduser(name)
    if os.sep not in path:
        found = shutil.which(path)
        if not found:
            raise WhisperXUnavailable(
                f"WhisperX executable '{name}' is not on PATH; set WHISPERX_BIN "
                f"to the whisperx inside its virtualenv (.venv/bin/whisperx)")
        return found
    if not os.path.isfile(path):
        raise WhisperXUnavailable(f"WhisperX executable not found at {path} (WHISPERX_BIN)")
    if not os.access(path, os.X_OK):
        raise WhisperXUnavailable(f"WhisperX executable at {path} is not executable")
    return path


def timeout_seconds(cfg, media_duration):
    """Hard limit for one run. Model downloads on a first run count too, hence
    the generous base."""
    try:
        explicit = float(cfg.get("timeout") or 0)
    except ValueError:
        explicit = 0
    if explicit > 0:
        return explicit
    factor = 1.0 if cfg["device"].startswith("cuda") else 4.0
    return 600.0 + factor * float(media_duration or 0)


def build_command(cfg, executable, audio_path, out_dir):
    cmd = [
        executable, audio_path,
        "--model", cfg["model"],
        "--device", cfg["device"],
        "--compute_type", cfg["compute_type"],
        "--batch_size", str(cfg["batch_size"]),
        "--output_dir", out_dir,
        "--output_format", "json",
        "--print_progress", "True",
        "--log-level", "warning",
    ]
    if cfg["device"] == "cpu":
        cmd += ["--threads", str(cfg["threads"])]
    if cfg["language"]:
        cmd += ["--language", cfg["language"]]
    if cfg["align_model"]:
        cmd += ["--align_model", cfg["align_model"]]
    if cfg["vad_method"]:
        cmd += ["--vad_method", cfg["vad_method"]]
    if cfg["cache_only"]:
        cmd += ["--model_cache_only", "True"]
    return cmd


# Only what WhisperX needs from this process's environment: never the app's
# API keys, and never this app's own Python (PYTHONPATH / VIRTUAL_ENV would
# point a Python 3.12 venv at 3.11 packages).
_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ",
             "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN", "HF_HUB_OFFLINE", "TORCH_HOME",
             "XDG_CACHE_HOME", "NLTK_DATA", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")


def child_env(cfg):
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env.update({k: v for k, v in os.environ.items() if k.startswith("NVIDIA_")})
    env["PYTHONUNBUFFERED"] = "1"
    # Job processes run with OMP_NUM_THREADS=1; torch inside WhisperX would
    # inherit that and decode on one core.
    env["OMP_NUM_THREADS"] = str(cfg["threads"])
    if cfg["cache_dir"]:
        root = os.path.abspath(os.path.expanduser(cfg["cache_dir"]))
        os.makedirs(root, exist_ok=True)
        env["HF_HOME"] = os.path.join(root, "huggingface")
        env["TORCH_HOME"] = os.path.join(root, "torch")
        env["NLTK_DATA"] = os.path.join(root, "nltk_data")
        env["XDG_CACHE_HOME"] = root
        env.pop("HF_HUB_CACHE", None)
    if cfg["cache_only"]:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def extract_audio(media_path, wav_path, timeout=1800):
    """16 kHz mono wav on the video's timeline (leading gap padded)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", media_path, "-map", "0:a:0",
           "-vn", "-af", "aresample=async=1:first_pts=0",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace").strip()[-300:]
        raise WhisperXError(f"audio extraction failed: {tail}") from e


def output_dir_for(cfg, media_path):
    """Directory that keeps this run's raw WhisperX JSON (created)."""
    stem = os.path.splitext(os.path.basename(media_path))[0] or "media"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
    if cfg["output_dir"]:
        base = os.path.abspath(os.path.expanduser(cfg["output_dir"]))
        target = os.path.join(base, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}")
    else:
        target = os.path.join(os.path.dirname(os.path.abspath(media_path)), ".whisperx")
    try:
        os.makedirs(target, exist_ok=True)
        return target
    except OSError:
        return tempfile.mkdtemp(prefix="whisperx_")


_PROGRESS_RX = re.compile(r"Progress:\s*([\d.]+)%")
# Lines of WhisperX's output that are noise in a job log: dependency warnings
# it prints on every run and download bars.
_NOISE_RX = re.compile(r"torchcodec|libavutil|libtorchcodec|FFmpeg version|warnings\.warn|"
                       r"UserWarning|Lightning automatically|^\s*$|\d+%\|")


class _Run:
    """Reads the child's output: progress lines, a tail for errors."""

    def __init__(self, proc):
        self.proc = proc
        self.tail = []
        self.phase = "transcribe"
        self.last_pct = 0.0
        self.progress = tb._TranscribeProgress(100)
        self.downloading = False

    def pump(self):
        for raw in self.proc.stdout:
            line = raw.rstrip("\n").split("\r")[-1]
            m = _PROGRESS_RX.search(line)
            if m:
                self._on_progress(float(m.group(1)))
                continue
            if "Downloading" in line and not self.downloading:
                self.downloading = True
                print("⬇️ [ASR] whisperx is downloading a model (first run only)…", flush=True)
            if _NOISE_RX.search(line):
                continue
            self.tail.append(line)
            del self.tail[:-40]

    def _on_progress(self, pct):
        # WhisperX prints transcription progress, then alignment progress,
        # each from 0 to 100.
        if self.phase == "transcribe":
            if pct + 1 < self.last_pct and self.last_pct >= 99:
                self.phase = "align"
                print("🎙️ [ASR] whisperx aligning words…", flush=True)
            else:
                self.progress.update(pct)
        self.last_pct = pct


def _stop(proc):
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_whisperx(cmd, env, timeout):
    """Run WhisperX, streaming its progress; raise on failure or timeout."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, text=True,
                                encoding="utf-8", errors="replace", bufsize=1,
                                start_new_session=True)
    except OSError as e:
        raise WhisperXUnavailable(f"could not start WhisperX ({e})") from e
    run = _Run(proc)
    reader = threading.Thread(target=run.pump, daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop(proc)
        raise WhisperXTimeout(f"WhisperX ran past {timeout:.0f}s and was stopped "
                              f"(WHISPERX_TIMEOUT_SECONDS)")
    finally:
        reader.join(timeout=5)
    run.progress.update(100)
    if code != 0:
        detail = " | ".join(run.tail[-6:]) or "no output"
        raise WhisperXError(f"WhisperX exited with code {code}: {detail[-600:]}")
    return run


# --- WhisperX JSON -> transcript contract ------------------------------------

def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _fill_word_times(words, seg_start, seg_end):
    """Give every word a start and end. WhisperX leaves out the times of words
    its alignment model has no characters for (numbers, symbols, some names);
    they keep their place in the captions between their neighbours."""
    n = len(words)
    for i, w in enumerate(words):
        if w["start"] is not None and w["end"] is not None:
            continue
        prev_end = next((words[j]["end"] for j in range(i - 1, -1, -1)
                         if words[j]["end"] is not None), seg_start)
        nxt = next(((j, words[j]["start"]) for j in range(i + 1, n)
                    if words[j]["start"] is not None), (n, seg_end))
        # Share the gap between the run of untimed words it holds.
        run_len = nxt[0] - i
        gap_end = nxt[1] if nxt[1] is not None else prev_end
        span = max(0.0, gap_end - prev_end) / max(run_len, 1)
        if w["start"] is None:
            w["start"] = prev_end
        if w["end"] is None:
            w["end"] = w["start"] + span
    return words


def to_transcript(data, duration=None, language_hint=""):
    """WhisperX's JSON (``{"segments": [...], "language": ...}``) as the
    pipeline's transcript: absolute seconds, " word" texts, floats only,
    chronological, clamped to the media."""
    limit = _num(duration)
    out_segments = []
    for seg in (data or {}).get("segments") or []:
        text = str(seg.get("text") or "").strip()
        seg_start = _num(seg.get("start"))
        seg_end = _num(seg.get("end"))
        words = []
        for w in seg.get("words") or []:
            token = str(w.get("word") or "").strip()
            if not token:
                continue
            words.append({"word": " " + token, "start": _num(w.get("start")),
                          "end": _num(w.get("end"))})
        if not text and not words:
            continue
        if not text:
            text = "".join(w["word"] for w in words).strip()
        timed = [w for w in words if w["start"] is not None]
        if seg_start is None:
            seg_start = timed[0]["start"] if timed else 0.0
        if seg_end is None:
            seg_end = max((w["end"] for w in words if w["end"] is not None), default=seg_start)
        _fill_word_times(words, seg_start, seg_end)
        out_segments.append({"start": seg_start, "end": seg_end, "text": text,
                             "words": words})

    out_segments.sort(key=lambda s: s["start"])
    prev = 0.0
    for seg in out_segments:
        for w in seg["words"]:
            start = max(w["start"], prev)
            end = max(w["end"], start)
            if limit:
                start, end = min(start, limit), min(end, limit)
            w["start"], w["end"] = float(start), float(end)
            prev = start
        seg["words"] = merge_continuation_words(seg["words"])
        if seg["words"]:
            seg["start"] = min(seg["start"], seg["words"][0]["start"])
            seg["end"] = max(seg["end"], seg["words"][-1]["end"])
        if limit:
            seg["start"], seg["end"] = min(seg["start"], limit), min(seg["end"], limit)
        seg["start"], seg["end"] = float(seg["start"]), float(max(seg["end"], seg["start"]))

    text = " ".join(s["text"] for s in out_segments if s["text"])
    language = str((data or {}).get("language") or language_hint or "").strip().lower()
    return {"text": text, "language": language or tb._detect_language(text),
            "segments": out_segments}


def validate_transcript(transcript, duration=None, tolerance=0.5):
    """Problems that would desync captions (empty list = valid)."""
    problems = []
    prev = -1.0
    limit = _num(duration)
    for si, seg in enumerate(transcript.get("segments") or []):
        for key in ("start", "end"):
            if type(seg.get(key)) is not float:
                problems.append(f"segment {si} {key} is not a float")
        for wi, w in enumerate(seg.get("words") or []):
            where = f"segment {si} word {wi}"
            if type(w.get("start")) is not float or type(w.get("end")) is not float:
                problems.append(f"{where}: times are not floats")
                continue
            if not str(w.get("word", "")).startswith(" "):
                problems.append(f"{where}: no leading space")
            if w["start"] < 0 or w["end"] < w["start"]:
                problems.append(f"{where}: bad span {w['start']}..{w['end']}")
            if w["start"] < prev:
                problems.append(f"{where}: starts before the previous word")
            if limit and w["end"] > limit + tolerance:
                problems.append(f"{where}: ends after the media ({w['end']} > {limit})")
            prev = w["start"]
    return problems


# --- entry point --------------------------------------------------------------

def transcribe(media_path):
    """Transcribe ``media_path`` with the local WhisperX install."""
    cfg = config()
    gpu = cfg["device"].startswith("cuda")
    if tb._gpu_only and not gpu:
        raise tb.GpuUnavailable("whisperx is configured for the CPU")
    executable = resolve_executable(cfg["bin"])
    duration = tb.media_seconds(media_path)
    out_dir = output_dir_for(cfg, media_path)
    stem = os.path.splitext(os.path.basename(media_path))[0] or "media"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
    work = tempfile.mkdtemp(prefix="whisperx_audio_")
    wav_path = os.path.join(work, f"{stem}.wav")
    started = time.monotonic()
    try:
        extract_audio(media_path, wav_path)
        cmd = build_command(cfg, executable, wav_path, out_dir)
        limit = timeout_seconds(cfg, duration)
        print(f"🎙️ [ASR] provider=whisperx model={cfg['model']} device={cfg['device']} "
              f"compute={cfg['compute_type']} language={cfg['language'] or 'auto'} "
              f"bin={executable} timeout={limit:.0f}s", flush=True)
        # One GPU transcription at a time in this process, as for whisper.
        gate = tb._ASR_GATE if gpu else tb._NULL_GATE
        with gate:
            run_whisperx(cmd, child_env(cfg), limit)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    raw_path = os.path.join(out_dir, f"{stem}.json")
    try:
        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise WhisperXError(f"WhisperX produced no readable JSON at {raw_path} ({e})") from e

    transcript = to_transcript(data, duration, cfg["language"])
    problems = validate_transcript(transcript, duration)
    if problems:
        raise WhisperXError(f"invalid word timestamps: {problems[0]} "
                            f"(+{len(problems) - 1} more)")
    n_words = sum(len(s["words"]) for s in transcript["segments"])
    print(f"🎙️ [ASR] whisperx ok: lang={transcript['language']} "
          f"segments={len(transcript['segments'])} words={n_words} "
          f"in {time.monotonic() - started:.0f}s; raw output saved to {raw_path}",
          flush=True)
    return transcript

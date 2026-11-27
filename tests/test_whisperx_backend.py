"""Local WhisperX provider (whisperx_backend): config, JSON conversion, word
timestamp validity, fallback, timeouts, and (when a WhisperX install is
configured) a real run on a sample video whose audio starts late."""
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap

import pytest

import transcribe_backends as tb
import whisperx_backend as wx

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "speech_sample.ogg")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("WHISPERX_") and not key.startswith("WHISPERX_TEST_"):
            monkeypatch.delenv(key, raising=False)


# --- config and command ------------------------------------------------------

def test_defaults_are_cpu_int8():
    cfg = wx.config()
    assert (cfg["device"], cfg["compute_type"], cfg["model"]) == ("cpu", "int8", "small")
    assert cfg["language"] == ""


def test_command_carries_settings(monkeypatch):
    monkeypatch.setenv("WHISPERX_MODEL", "large-v3")
    monkeypatch.setenv("WHISPERX_DEVICE", "cuda")
    monkeypatch.setenv("WHISPERX_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("WHISPERX_LANGUAGE", "EN")
    monkeypatch.setenv("WHISPERX_MODEL_CACHE_ONLY", "1")
    cmd = wx.build_command(wx.config(), "/x/whisperx", "/tmp/a.wav", "/tmp/out")
    joined = " ".join(cmd)
    assert cmd[:2] == ["/x/whisperx", "/tmp/a.wav"]
    for part in ("--model large-v3", "--device cuda", "--compute_type float16",
                 "--language en", "--output_format json", "--output_dir /tmp/out",
                 "--model_cache_only True"):
        assert part in joined
    assert "--threads" not in cmd  # GPU run


def test_timeout_scales_with_media(monkeypatch):
    assert wx.timeout_seconds(wx.config(), 100) == 1000
    monkeypatch.setenv("WHISPERX_TIMEOUT_SECONDS", "42")
    assert wx.timeout_seconds(wx.config(), 100) == 42


def test_child_env_keeps_no_secrets_nor_app_python(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/opt/venv/lib")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("WHISPERX_CACHE_DIR", str(tmp_path / "cache"))
    env = wx.child_env(wx.config())
    assert "GEMINI_API_KEY" not in env
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["TORCH_HOME"].startswith(str(tmp_path))


def test_missing_executable_is_reported(monkeypatch, tmp_path):
    with pytest.raises(wx.WhisperXUnavailable, match="not found"):
        wx.resolve_executable(str(tmp_path / "nope" / "whisperx"))
    with pytest.raises(wx.WhisperXUnavailable, match="PATH"):
        wx.resolve_executable("definitely-not-a-whisperx-binary")


# --- JSON conversion -----------------------------------------------------------

RAW = {
    "language": "en",
    "segments": [
        {"start": 0.42, "end": 3.0, "text": " And so my fellow Americans",
         "words": [{"word": "And", "start": 0.42, "end": 0.54, "score": 0.6},
                   {"word": "so", "start": 0.64, "end": 0.94},
                   {"word": "my", "start": 1.02, "end": 1.24},
                   {"word": "fellow", "start": 1.32, "end": 1.64},
                   {"word": "Americans", "start": 1.70, "end": 2.22}]},
        {"start": 3.5, "end": 6.0, "text": " in 1961 we",
         "words": [{"word": "in", "start": 3.5, "end": 3.7},
                   {"word": "1961"},                # alignment has no digits
                   {"word": "we", "start": 5.0, "end": 5.3}]},
    ],
}


def test_to_transcript_matches_the_pipeline_contract():
    t = wx.to_transcript(RAW, duration=11.0)
    assert t["language"] == "en"
    assert t["text"] == "And so my fellow Americans in 1961 we"
    words = [w for s in t["segments"] for w in s["words"]]
    assert [w["word"] for w in words][:2] == [" And", " so"]
    assert all(set(w) == {"word", "start", "end"} for w in words)
    assert wx.validate_transcript(t, 11.0) == []


def test_unaligned_word_is_kept_between_its_neighbours():
    seg = wx.to_transcript(RAW)["segments"][1]
    num = seg["words"][1]
    assert num["word"] == " 1961"
    assert 3.7 <= num["start"] <= num["end"] <= 5.0


def test_times_are_clamped_to_the_media():
    raw = {"segments": [{"start": 9.0, "end": 12.5, "text": "end",
                         "words": [{"word": "end", "start": 9.0, "end": 12.5}]}]}
    t = wx.to_transcript(raw, duration=10.0, language_hint="en")
    assert t["segments"][0]["words"][0]["end"] == 10.0
    assert t["segments"][0]["end"] == 10.0


def test_validate_flags_desync():
    bad = {"segments": [{"start": 0.0, "end": 1.0, "text": "a b", "words": [
        {"word": " a", "start": 0.5, "end": 0.4},
        {"word": "b", "start": 0.1, "end": 0.2}]}]}
    problems = wx.validate_transcript(bad)
    assert any("bad span" in p for p in problems)
    assert any("leading space" in p for p in problems)
    assert any("before the previous" in p for p in problems)


# --- the subprocess, with a fake whisperx ---------------------------------------

def _fake_whisperx(tmp_path, body):
    script = tmp_path / "fake-whisperx"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Skip real audio extraction and probing (CI has no media stack)."""
    monkeypatch.setattr(wx, "extract_audio", lambda src, dst, timeout=0: open(dst, "wb").close())
    monkeypatch.setattr(tb, "media_seconds", lambda path: 11.0)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)


def test_transcribe_runs_the_executable_and_keeps_its_json(monkeypatch, tmp_path, no_ffmpeg):
    fake = _fake_whisperx(tmp_path, f"""
        import json, os, sys
        args = sys.argv[1:]
        out = args[args.index("--output_dir") + 1]
        stem = os.path.splitext(os.path.basename(args[0]))[0]
        print("Progress: 50.00%...", flush=True)
        print("Progress: 100.00%...", flush=True)
        json.dump({json.dumps(RAW)}, open(os.path.join(out, stem + ".json"), "w"))
    """)
    monkeypatch.setenv("WHISPERX_BIN", fake)
    monkeypatch.setenv("WHISPERX_OUTPUT_DIR", str(tmp_path / "raw"))
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    t = wx.transcribe(str(media))
    assert wx.validate_transcript(t, 11.0) == []
    saved = list((tmp_path / "raw").glob("clip-*/clip.json"))
    assert len(saved) == 1


def test_nonzero_exit_raises_with_its_output(monkeypatch, tmp_path, no_ffmpeg):
    fake = _fake_whisperx(tmp_path, """
        import sys
        print("ValueError: model 'nope' not found")
        sys.exit(3)
    """)
    monkeypatch.setenv("WHISPERX_BIN", fake)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    with pytest.raises(wx.WhisperXError, match="code 3.*nope"):
        wx.transcribe(str(media))


def test_timeout_stops_the_run(tmp_path):
    fake = _fake_whisperx(tmp_path, """
        import time
        time.sleep(30)
    """)
    with pytest.raises(wx.WhisperXTimeout):
        wx.run_whisperx([fake], {"PATH": os.environ.get("PATH", "")}, timeout=1)


# --- provider selection and fallback ----------------------------------------------

def test_transcribe_media_uses_whisperx(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisperx")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(wx, "transcribe", lambda path: sentinel)
    monkeypatch.setattr(tb, "_transcribe_with_whisper",
                        lambda path: (_ for _ in ()).throw(AssertionError("no second run")))
    assert tb.transcribe_media("video.mp4") is sentinel


def test_whisperx_failure_falls_back_to_whisper(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisperx")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(wx, "transcribe",
                        lambda path: (_ for _ in ()).throw(wx.WhisperXUnavailable("missing")))
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_whisperx_without_fallback_raises(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisperx")
    monkeypatch.setenv("WHISPERX_FALLBACK", "none")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(wx, "transcribe",
                        lambda path: (_ for _ in ()).throw(wx.WhisperXTimeout("slow")))
    with pytest.raises(wx.WhisperXTimeout):
        tb.transcribe_media("video.mp4")


# --- a real WhisperX run --------------------------------------------------------------

def _real_whisperx():
    try:
        return wx.resolve_executable(os.environ.get("WHISPERX_TEST_BIN")
                                     or os.environ.get("WHISPERX_BIN") or "whisperx")
    except wx.WhisperXUnavailable:
        return None


@pytest.mark.skipif(not _real_whisperx() or not shutil.which("ffmpeg"),
                    reason="needs a WhisperX install (WHISPERX_TEST_BIN) and ffmpeg")
def test_real_whisperx_word_timestamps_follow_the_video(monkeypatch, tmp_path):
    """The sample speech is muxed into a video with its audio starting 1.5 s
    late: every word must come back on the video's timeline, inside it."""
    delay = 1.5
    video = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=320x568:r=25:d=13",
        "-itsoffset", str(delay), "-i", FIXTURE,
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-t", "13", str(video)], check=True)
    monkeypatch.setenv("WHISPERX_BIN", _real_whisperx())
    monkeypatch.setenv("WHISPERX_MODEL", os.environ.get("WHISPERX_TEST_MODEL", "tiny"))
    monkeypatch.setenv("WHISPERX_LANGUAGE", "en")
    monkeypatch.setenv("WHISPERX_OUTPUT_DIR", str(tmp_path / "raw"))
    for key in ("WHISPERX_CACHE_DIR", "WHISPERX_DEVICE", "WHISPERX_COMPUTE_TYPE"):
        if os.environ.get(key.replace("WHISPERX_", "WHISPERX_TEST_")):
            monkeypatch.setenv(key, os.environ[key.replace("WHISPERX_", "WHISPERX_TEST_")])

    t = wx.transcribe(str(video))
    duration = tb.media_seconds(str(video))
    assert wx.validate_transcript(t, duration) == []
    words = [w for s in t["segments"] for w in s["words"]]
    assert len(words) >= 15
    assert "country" in t["text"].lower()
    # "And" is spoken ~0.4 s into the sample, so ~1.9 s into the video.
    assert delay < words[0]["start"] < delay + 1.0
    assert words[-1]["end"] <= duration + 0.5
    assert list((tmp_path / "raw").glob("sample-*/sample.json"))

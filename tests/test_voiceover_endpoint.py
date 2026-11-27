"""HTTP tests for /api/voiceover (TTS from a script) and /api/voiceover/upload.

Same conventions as test_rerender_endpoint.py: a real ASGI round-trip with
BILLING_ENABLED=0; the audio work is stubbed at translate's seams, so these own
the endpoint contract (validation, input walk-back, metadata update)."""

import asyncio
import json

import httpx
import pytest

app_module = pytest.importorskip("app")
import translate  # noqa: E402

JOB_ID = "voiceover-endpoint-test-job"
VOICE = "21m00Tcm4TlvDq8ikWAM"


def _request(method, path, **kwargs):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "_archive_clip_edit_bg", lambda *a, **k: None)
    clip = {"start": 0.0, "end": 30.0, "video_url": f"/videos/{JOB_ID}/t_clip_1.mp4"}
    (job_dir / "t_metadata.json").write_text(json.dumps({"shorts": [clip]}))
    (job_dir / "t_clip_1.mp4").write_bytes(b"clip")
    app_module.jobs[JOB_ID] = {"status": "completed", "logs": [], "user_id": None,
                               "result": {"clips": [dict(clip)]}}
    try:
        yield job_dir
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def calls(monkeypatch):
    seen = []

    def fake_tts(**kw):
        seen.append(("tts", kw))
        open(kw["output_path"], "wb").write(b"vo")

    def fake_mix(video, audio, out, original_audio="duck", fit_voice=False):
        seen.append(("mix", {"video": video, "audio": open(audio, "rb").read(),
                             "original_audio": original_audio, "fit_voice": fit_voice}))
        open(out, "wb").write(b"mixed")

    monkeypatch.setattr(translate, "voiceover_video", fake_tts)
    monkeypatch.setattr(translate, "mix_audio_into_video", fake_mix)
    monkeypatch.setattr("ffmpeg_utils.mark_ai_generated", lambda *a, **k: True)
    return seen


def _tts(body, key="k"):
    headers = {"X-ElevenLabs-Key": key} if key else {}
    return _request("POST", "/api/voiceover", json=body, headers=headers)


def test_script_voiceover_replaces_the_clip(job, calls):
    resp = _tts({"job_id": JOB_ID, "clip_index": 0, "script": " Hello there ",
                 "voice_id": VOICE, "original_audio": "replace"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["new_video_url"] == f"/videos/{JOB_ID}/translated_vo_t_clip_1.mp4"
    kind, kw = calls[0]
    assert kind == "tts" and kw["script"] == "Hello there"
    assert kw["original_audio"] == "replace" and kw["voice_id"] == VOICE
    assert kw["model_id"] == "eleven_v3"
    meta = json.loads((job / "t_metadata.json").read_text())
    assert meta["shorts"][0]["video_url"].endswith("translated_vo_t_clip_1.mp4")


def test_a_second_voiceover_starts_from_the_clip_under_the_first(job, calls):
    (job / "translated_vo_t_clip_1.mp4").write_bytes(b"old vo")
    resp = _tts({"job_id": JOB_ID, "clip_index": 0, "script": "again", "voice_id": VOICE,
                 "input_filename": "translated_vo_t_clip_1.mp4"})
    assert resp.status_code == 200, resp.text
    assert calls[0][1]["video_path"].endswith("/t_clip_1.mp4")


@pytest.mark.parametrize("body,key", [
    ({"script": "hi", "voice_id": VOICE}, None),
    ({"script": "   ", "voice_id": VOICE}, "k"),
    ({"script": "hi", "voice_id": "../../user"}, "k"),
    ({"script": "hi", "voice_id": VOICE, "original_audio": "loud"}, "k"),
    ({"script": "hi", "voice_id": VOICE, "model": "eleven_monolingual_v1"}, "k"),
])
def test_bad_requests_are_refused_before_any_work(job, calls, body, key):
    resp = _tts({"job_id": JOB_ID, "clip_index": 0, **body}, key=key)
    assert resp.status_code == 400
    assert calls == []


def _upload(name, content, **fields):
    data = {"job_id": JOB_ID, "clip_index": "0", **fields}
    return _request("POST", "/api/voiceover/upload", data=data,
                    files={"file": (name, content, "audio/mpeg")})


def test_uploaded_audio_is_mixed_without_a_key(job, calls):
    resp = _upload("music.mp3", b"ID3audio", original_audio="mix")
    assert resp.status_code == 200, resp.text
    kind, kw = calls[0]
    assert kind == "mix" and kw["audio"] == b"ID3audio"
    assert kw["original_audio"] == "mix" and kw["fit_voice"] is False
    assert (job / "translated_vo_t_clip_1.mp4").read_bytes() == b"mixed"


def test_upload_refuses_non_audio_types(job, calls):
    assert _upload("evil.sh", b"#!/bin/sh").status_code == 400
    assert calls == []


def test_upload_refuses_an_unknown_clip(job, calls):
    resp = _request("POST", "/api/voiceover/upload",
                    data={"job_id": JOB_ID, "clip_index": "5"},
                    files={"file": ("a.mp3", b"x", "audio/mpeg")})
    assert resp.status_code == 404


def test_upload_with_placements_places_pieces(job, monkeypatch, calls):
    got = {}

    def fake_pieces(video, audio, out, pieces, original_audio="duck"):
        got["pieces"] = pieces
        open(out, "wb").write(b"pieces")

    monkeypatch.setattr(translate, "mix_audio_pieces", fake_pieces)
    resp = _upload("voice.mp3", b"ID3", placements="0:00-0:05 -> 0:05-0:10")
    assert resp.status_code == 200, resp.text
    assert got["pieces"] == [{"a_start": 0.0, "a_end": 5.0, "v_start": 5.0}]
    assert calls == []  # not the whole-track mix


def test_upload_with_unreadable_placements_is_refused(job, calls):
    resp = _upload("voice.mp3", b"ID3", placements="audio 0:00 to 0:05, somewhere")
    assert resp.status_code == 400 and "line 1" in resp.json()["detail"]


# --- standalone mixer ------------------------------------------------------

import shutil  # noqa: E402
import subprocess  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _lavfi(path, *args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, str(path)], check=True)


@needs_ffmpeg
def test_audio_mix_returns_the_mixed_video_and_keeps_nothing(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    video, audio = tmp_path / "My clip!.mp4", tmp_path / "voice.wav"
    _lavfi(video, "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=6",
           "-c:v", "libx264", "-pix_fmt", "yuv420p")
    _lavfi(audio, "-f", "lavfi", "-i", "sine=frequency=500:duration=4")
    table = ("Here is the placement map:\nElevenLabs audio timestamp    Place it at video timestamp\n"
             "00:00 – 00:01.5    00:00 – 00:02\n00:01.9 – 00:03.2    00:04 – 00:06\n")
    resp = _request("POST", "/api/audio-mix",
                    data={"original_audio": "replace", "placements": table},
                    files={"video": (video.name, video.read_bytes(), "video/mp4"),
                           "audio": (audio.name, audio.read_bytes(), "audio/wav")})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "video/mp4"
    assert 'filename="My_clip_mixed.mp4"' in resp.headers["content-disposition"]
    out = tmp_path / "out.mp4"
    out.write_bytes(resp.content)
    assert translate.has_audio(str(out))
    assert translate.media_duration(str(out)) == pytest.approx(6.0, abs=0.25)
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize("files,data", [
    ({"video": ("a.exe", b"x"), "audio": ("a.mp3", b"x")}, {}),
    ({"video": ("a.mp4", b"x"), "audio": ("a.txt", b"x")}, {}),
    ({"video": ("a.mp4", b"x"), "audio": ("a.mp3", b"x")}, {"placements": "0:00-0:05"}),
    ({"video": ("a.mp4", b"x"), "audio": ("a.mp3", b"x")}, {"original_audio": "loud"}),
])
def test_audio_mix_refuses_bad_input(files, data):
    assert _request("POST", "/api/audio-mix", data=data, files=files).status_code == 400

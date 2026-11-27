"""Script voiceover (ElevenLabs TTS) and uploaded-audio mixing (translate.py)."""
import shutil
import subprocess

import pytest

import translate
from translate import (MAX_SPEEDUP, build_voiceover_command, mix_audio_into_video,
                       valid_voice_id, voiceover_plan)


def test_voice_shorter_than_clip_is_left_alone():
    assert voiceover_plan(30.0, 12.0) == (1.0, 0.0)


def test_slightly_long_voice_is_sped_up_to_fit():
    tempo, hold = voiceover_plan(30.0, 33.0)
    assert tempo == pytest.approx(1.1)
    assert hold == 0.0


def test_much_longer_voice_holds_the_last_frame():
    tempo, hold = voiceover_plan(20.0, 46.0)
    assert tempo == MAX_SPEEDUP
    assert 20.0 + hold == pytest.approx(46.0 / MAX_SPEEDUP)


@pytest.mark.parametrize("voice_id,ok", [
    ("21m00Tcm4TlvDq8ikWAM", True),
    ("../../v1/user", False),
    ("abc", False),
    ("21m00Tcm4Tlv?x=1", False),
    (None, False),
])
def test_voice_id_shape(voice_id, ok):
    assert valid_voice_id(voice_id) is ok


def _graph(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_duck_keeps_clip_audio_quietly_and_copies_video():
    cmd = build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 30.0, 10.0, "duck", True)
    graph = _graph(cmd)
    assert f"volume={translate.DUCK_VOLUME}" in graph and "amix" in graph
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-t") + 1] == "30.000"


def test_replace_and_silent_source_drop_clip_audio():
    for mode, src_audio in (("replace", True), ("mix", False)):
        assert "[0:a]" not in _graph(
            build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 30.0, 10.0, mode, src_audio))


def test_mix_keeps_clip_audio_at_full_level():
    assert "volume=1.0" in _graph(
        build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 30.0, 10.0, "mix", True))


def test_long_voice_extends_the_video():
    cmd = build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 20.0, 46.0, "duck", True)
    assert "tpad=stop_mode=clone" in _graph(cmd)
    assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(46.0 / MAX_SPEEDUP, abs=0.01)


def test_uploaded_track_is_cut_at_the_clip_end():
    cmd = build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 20.0, 180.0, "mix", True,
                                  fit_voice=False)
    assert "tpad" not in _graph(cmd) and "atempo=1.0000" in _graph(cmd)
    assert cmd[cmd.index("-t") + 1] == "20.000"


def test_unknown_original_audio_mode_is_refused():
    with pytest.raises(ValueError):
        build_voiceover_command("v.mp4", "a.mp3", "o.mp4", 20.0, 5.0, "loud", True)


def test_script_limits_are_checked_before_any_api_call(monkeypatch):
    monkeypatch.setattr(translate, "synthesize_speech",
                        lambda *a, **k: pytest.fail("no TTS call expected"))
    with pytest.raises(ValueError):
        translate.voiceover_video("v.mp4", "o.mp4", "   ", "21m00Tcm4TlvDq8ikWAM", "k")
    with pytest.raises(ValueError):
        translate.voiceover_video("v.mp4", "o.mp4", "x" * (translate.MAX_SCRIPT_CHARS + 1),
                                  "21m00Tcm4TlvDq8ikWAM", "k")


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg not installed")


def _lavfi(out, *args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, out], check=True)


@needs_ffmpeg
def test_mixes_an_audio_file_into_a_real_clip(tmp_path):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "track.wav")
    out = str(tmp_path / "translated_vo_clip.mp4")
    _lavfi(video, "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=3",
           "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest")
    _lavfi(audio, "-f", "lavfi", "-i", "sine=frequency=880:duration=8")

    mix_audio_into_video(video, audio, out, original_audio="duck")

    assert translate.has_audio(out)
    assert translate.media_duration(out) == pytest.approx(3.0, abs=0.25)


@needs_ffmpeg
def test_a_file_without_audio_is_refused(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _lavfi(video, "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p")
    with pytest.raises(ValueError, match="no audio"):
        mix_audio_into_video(video, video, str(tmp_path / "o.mp4"))


# --- timed scripts ---------------------------------------------------------

TIMED = """0:00–0:04    [curious] What kind of games are hiding inside this pack?
0:04-0:10    [explaining] See every available item and its odds before opening.
0:10 — 0:17  [excited] Now... watch what came out of this pack!
  and keep watching
0:23         [energetic] Discover new games!"""


def test_parse_timed_script():
    lines = translate.parse_timed_script(TIMED)
    assert [(ln["start"], ln["end"]) for ln in lines] == [
        (0.0, 4.0), (4.0, 10.0), (10.0, 17.0), (23.0, None)]
    assert lines[0]["text"].startswith("[curious] What kind")
    assert lines[2]["text"].endswith("pack! and keep watching")


def test_plain_script_is_not_timed():
    assert translate.parse_timed_script("Hello there.\nSecond sentence at 10:30 pm") is None


def test_lines_start_at_their_time_with_silence_between():
    lines = translate.parse_timed_script(TIMED)
    placed = translate.place_timed_lines(lines, [2.5, 3.0, 2.0, 2.0])
    assert placed == [(0.0, 1.0), (4.0, 1.0), (10.0, 1.0), (23.0, 1.0)]


def test_a_line_too_long_for_its_slot_is_sped_up_then_pushed():
    lines = translate.parse_timed_script(TIMED)
    start0, tempo0 = translate.place_timed_lines(lines, [6.0, 3.0, 2.0, 2.0])[0]
    assert tempo0 == MAX_SPEEDUP
    start1 = translate.place_timed_lines(lines, [6.0, 3.0, 2.0, 2.0])[1][0]
    assert start1 == pytest.approx(6.0 / MAX_SPEEDUP + translate.MIN_LINE_GAP)


def test_timed_track_delays_each_line():
    cmd = translate.build_timed_track_command(["a.mp3", "b.mp3"], [(0.0, 1.0), (4.25, 1.1)], "t.wav")
    graph = _graph(cmd)
    assert "adelay=0:all=1" in graph and "adelay=4250:all=1" in graph
    assert "atempo=1.1000" in graph and "amix=inputs=2" in graph


def test_audio_tags_are_removed_for_v2():
    assert translate.strip_audio_tags("[excited] Now...  [laughs] watch!") == "Now... watch!"


def _fake_tts_status(monkeypatch, refuse_v3_with):
    sent = []

    class Resp:
        def __init__(self, status, content=b"mp3"):
            self.status_code, self.content, self.text = status, content, "nope"

        def json(self):
            return {"detail": {"message": "nope"}}

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None, params=None):
            sent.append(json)
            if json["model_id"] == "eleven_v3" and refuse_v3_with:
                return Resp(refuse_v3_with)
            return Resp(200)

    monkeypatch.setattr(translate.httpx, "Client", Client)
    return sent


def test_v3_falls_back_to_v2_without_tags(monkeypatch, tmp_path):
    sent = _fake_tts_status(monkeypatch, refuse_v3_with=400)
    translate._speak("[excited] Go!", "21m00Tcm4TlvDq8ikWAM", "k", str(tmp_path / "o.mp3"),
                     "eleven_v3")
    assert [b["model_id"] for b in sent] == ["eleven_v3", "eleven_multilingual_v2"]
    assert sent[0]["text"] == "[excited] Go!" and sent[1]["text"] == "Go!"


def test_a_bad_key_does_not_fall_back(monkeypatch, tmp_path):
    sent = _fake_tts_status(monkeypatch, refuse_v3_with=401)
    with pytest.raises(translate.TTSError):
        translate._speak("Go", "21m00Tcm4TlvDq8ikWAM", "k", str(tmp_path / "o.mp3"), "eleven_v3")
    assert len(sent) == 1


@needs_ffmpeg
def test_timed_voiceover_end_to_end(monkeypatch, tmp_path):
    """Fake TTS (a 1 s tone per line) through the real ffmpeg placement."""
    video = str(tmp_path / "clip.mp4")
    out = str(tmp_path / "translated_vo_clip.mp4")
    _lavfi(video, "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p")
    texts = []

    def fake_speech(text, voice_id, api_key, path, model_id):
        texts.append(text)
        _lavfi(path, "-f", "lavfi", "-i", "sine=frequency=440:duration=1")
        return path

    monkeypatch.setattr(translate, "synthesize_speech", fake_speech)
    translate.voiceover_video(video, out, "0:00 [curious] one\n0:02 two\n0:04 three",
                              "21m00Tcm4TlvDq8ikWAM", "k", original_audio="replace")

    assert sorted(texts) == ["[curious] one", "three", "two"]
    assert translate.has_audio(out)
    # The third line starts at 4 s on a 4 s clip: the last frame is held for it.
    assert translate.media_duration(out) == pytest.approx(5.0, abs=0.3)
    assert not list(tmp_path.glob("*.line*")) and not list(tmp_path.glob("*.track*"))


# --- uploaded audio, placed in parts --------------------------------------

def test_parse_audio_placements():
    pieces = translate.parse_audio_placements(
        "0:00-0:05 -> 0:05-0:10\naudio 0:12 - 0:20 in video 0:15\n\n0:30-0:40 -> 0:20-0:23")
    assert pieces == [
        {"a_start": 0.0, "a_end": 5.0, "v_start": 5.0},
        {"a_start": 12.0, "a_end": 20.0, "v_start": 15.0},
        # A shorter video range cuts the piece; nothing is stretched.
        {"a_start": 30.0, "a_end": 33.0, "v_start": 20.0},
    ]
    assert translate.parse_audio_placements("  \n") is None


@pytest.mark.parametrize("text,msg", [
    ("0:00-0:05", "line 1"),
    ("0:00-0:05 -> 0:05\n0:09-0:02 -> 0:10", "line 2: the audio range"),
    ("0:00-0:05 -> 0:10-0:08", "video range"),
])
def test_bad_placements_name_the_line(text, msg):
    with pytest.raises(ValueError, match=msg):
        translate.parse_audio_placements(text)


def test_pieces_track_trims_and_delays_each_piece():
    pieces = translate.parse_audio_placements("0:00-0:05 -> 0:05\n0:12-0:20 -> 0:15")
    graph = _graph(translate.build_pieces_track_command("a.mp3", pieces, "t.wav"))
    assert "asplit=2[s0][s1]" in graph
    assert "atrim=start=0.000:end=5.000,asetpts=PTS-STARTPTS,adelay=5000:all=1" in graph
    assert "atrim=start=12.000:end=20.000,asetpts=PTS-STARTPTS,adelay=15000:all=1" in graph


@needs_ffmpeg
def test_audio_pieces_end_to_end(tmp_path):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "track.wav")
    out = str(tmp_path / "translated_vo_clip.mp4")
    _lavfi(video, "-f", "lavfi", "-i", "testsrc=size=160x284:rate=25:duration=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p")
    _lavfi(audio, "-f", "lavfi", "-i", "sine=frequency=500:duration=6")
    pieces = translate.parse_audio_placements("0:00-0:01 -> 0:02-0:03\n0:04-0:06 -> 0:00")

    translate.mix_audio_pieces(video, audio, out, pieces, original_audio="replace")

    assert translate.has_audio(out)
    assert translate.media_duration(out) == pytest.approx(4.0, abs=0.25)
    assert not list(tmp_path.glob("*.pieces.wav"))


PASTED_TABLE = """Here is the placement map:
ElevenLabs audio timestamp    Place it at video timestamp
00:00 – 00:02.5    00:00 – 00:05
00:02.9 – 00:06.2    00:10 – 00:18
00:06.5 – 00:08.8    00:18 – 00:24
00:09.0 – 00:10.3    00:24 – 00:29
00:10.4 – 00:12.2    00:29 – 00:36
00:12.4 – 00:15.15    00:42 – 00:48
"""


def test_a_pasted_table_with_headings_is_read():
    pieces = translate.parse_audio_placements(PASTED_TABLE)
    assert [(p["a_start"], p["a_end"], p["v_start"]) for p in pieces] == [
        (0.0, 2.5, 0.0), (2.9, 6.2, 10.0), (6.5, 8.8, 18.0),
        (9.0, 10.3, 24.0), (10.4, 12.2, 29.0), (12.4, 15.15, 42.0)]

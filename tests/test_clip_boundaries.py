"""Smart clip boundaries (clip_boundaries.py): sentence-aware edges, padding,
duration band, LLM review (mocked), audio pauses, timestamp offsets and
caption sync after the edges move."""
import os
import shutil

import pytest

import clip_boundaries as cb
from subtitles import _collect_word_blocks

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "speech_sample.ogg")

WORD = 0.35       # seconds per word
WORD_GAP = 0.05   # between words of one sentence
PAUSE = 0.6       # between sentences

SCRIPT = [
    "Welcome back to the show everyone.",                                  # 0
    "Today we are talking about the strangest job I ever had.",            # 1
    "I worked nights at a bakery downtown for two whole years.",           # 2
    "Every morning the owner counted every single cake on the shelves.",  # 3
    "One morning a cake was missing and he blamed me.",                    # 4
    "And that is when I found out the dog had eaten the cake!",           # 5
    "He never apologised to me about it.",                                 # 6
    "Anyway that is the story of my first job.",                           # 7
    "Next week we will talk about my second job.",                         # 8
]


def make_transcript(script=SCRIPT, offset=0.0, pauses=None):
    """One segment per sentence, evenly timed words. ``pauses`` maps a
    sentence index to the silence before it."""
    t = offset
    segments = []
    for i, text in enumerate(script):
        if i:
            t += (pauses or {}).get(i, PAUSE)
        words = []
        for token in text.split():
            words.append({"word": " " + token, "start": round(t, 3), "end": round(t + WORD, 3)})
            t += WORD + WORD_GAP
        t -= WORD_GAP
        segments.append({"start": words[0]["start"], "end": words[-1]["end"],
                         "text": text, "words": words})
    return {"language": "en", "segments": segments}


def sentence_span(transcript, i):
    seg = transcript["segments"][i]
    return seg["words"][0]["start"], seg["words"][-1]["end"]


def word_at(transcript, i, j):
    return transcript["segments"][i]["words"][j]


def cfg(**kw):
    base = dict(min_duration=10.0, max_duration=60.0)
    base.update(kw)
    return cb.Config(**base)


@pytest.fixture
def tr():
    return make_transcript()


# --- sentences -----------------------------------------------------------------

def test_sentence_ends_come_from_punctuation_and_pauses():
    t = make_transcript(["so this has no full stop", "and this one does."], pauses={1: 1.5})
    words = cb.words_from_transcript(t)
    ends = [w["w"].strip() for w in words if w["eos"]]
    assert ends == ["stop", "does."]  # the long pause ends the first one


def test_segment_end_without_a_pause_is_not_a_sentence_end():
    t = make_transcript(["we were walking down", "the road together."], pauses={1: 0.1})
    words = cb.words_from_transcript(t)
    assert [w["w"].strip() for w in words if w["eos"]] == ["together."]


# --- edges ---------------------------------------------------------------------

def test_mid_sentence_start_moves_back_to_the_sentence(tr):
    words = cb.words_from_transcript(tr)
    s2, _ = sentence_span(tr, 2)
    ai_start = word_at(tr, 2, 4)["start"] + 0.1     # inside "at"
    _, ai_end = sentence_span(tr, 5)
    b = cb.refine(ai_start, ai_end, words, 120, cfg())
    assert b["speech"][0] == pytest.approx(s2)
    assert b["start"] < s2
    assert "incomplete_sentence_start" not in b["warnings"]
    assert any("start moved back" in r for r in b["reasons"])


def test_mid_sentence_end_finishes_the_sentence(tr):
    words = cb.words_from_transcript(tr)
    s2, _ = sentence_span(tr, 2)
    _, e4 = sentence_span(tr, 4)
    ai_end = word_at(tr, 4, 3)["end"]               # "...a cake was" | missing
    b = cb.refine(s2, ai_end, words, 120, cfg())
    assert b["speech"][1] == pytest.approx(e4)
    assert "incomplete_sentence_end" not in b["warnings"]
    assert any("end extended" in r for r in b["reasons"])


def test_incomplete_punchline_is_completed(tr):
    words = cb.words_from_transcript(tr)
    s3, _ = sentence_span(tr, 3)
    ai_end = word_at(tr, 5, 9)["start"] - 0.02      # right before "eaten"
    b = cb.refine(s3, ai_end, words, 120, cfg())
    _, e5 = sentence_span(tr, 5)
    assert b["speech"][1] == pytest.approx(e5)
    assert b["end"] >= e5
    kept = [w["w"].strip() for w in words if b["start"] <= w["s"] and w["e"] <= b["end"]]
    assert "cake!" in kept and "eaten" in kept


def test_padding_stops_before_neighbouring_speech(tr):
    words = cb.words_from_transcript(tr)
    s2, _ = sentence_span(tr, 2)
    _, e4 = sentence_span(tr, 4)
    b = cb.refine(s2, e4, words, 120, cfg(pad_start=2.0, pad_end=2.0))
    prev_end = sentence_span(tr, 1)[1]
    next_start = sentence_span(tr, 5)[0]
    assert b["start"] >= prev_end + 0.08 - 1e-6
    assert b["end"] <= next_start - 0.08 + 1e-6
    # The natural pause is kept, not removed.
    assert b["pad"][0] == pytest.approx(PAUSE - 0.08, abs=0.01)


def test_long_silence_gets_the_configured_padding_only():
    t = make_transcript(pauses={2: 10.0, 5: 10.0})
    words = cb.words_from_transcript(t)
    s2, _ = sentence_span(t, 2)
    _, e4 = sentence_span(t, 4)
    b = cb.refine(s2, e4, words, 200, cfg(pad_start=2.0, pad_end=2.0))
    assert b["start"] == pytest.approx(s2 - 2.0)
    assert b["end"] == pytest.approx(e4 + 2.0)
    assert "excessive_silence" not in b["warnings"]


def test_long_silence_inside_the_clip_is_flagged_not_removed():
    t = make_transcript(pauses={3: 6.0})
    words = cb.words_from_transcript(t)
    s2, _ = sentence_span(t, 2)
    _, e4 = sentence_span(t, 4)
    b = cb.refine(s2, e4, words, 200, cfg())
    assert "excessive_silence" in b["warnings"]
    assert b["speech"] == [pytest.approx(s2), pytest.approx(e4)]


def test_clip_that_opens_on_a_connective_is_flagged(tr):
    words = cb.words_from_transcript(tr)
    s5, _ = sentence_span(tr, 5)
    _, e7 = sentence_span(tr, 7)
    b = cb.refine(s5, e7, words, 120, cfg())
    assert "missing_context" in b["warnings"]


# --- duration ------------------------------------------------------------------

def test_short_clip_grows_by_whole_sentences(tr):
    words = cb.words_from_transcript(tr)
    s4, e4 = sentence_span(tr, 4)
    b = cb.refine(s4, e4, words, 120, cfg(min_duration=20.0))
    assert b["end"] - b["start"] >= 20.0
    assert "too_short" not in b["warnings"]
    # Grown on sentence ends: the speech ends on a sentence's last word.
    assert b["speech"][1] in [pytest.approx(sentence_span(tr, i)[1]) for i in range(9)]


def test_clip_that_cannot_reach_the_minimum_is_kept_and_flagged():
    t = make_transcript(["Hello there.", "Short one."])
    words = cb.words_from_transcript(t)
    b = cb.refine(0.0, 2.0, words, 10.0, cfg(min_duration=20.0))
    assert "too_short" in b["warnings"]
    assert b["end"] > b["start"]


def test_never_exceeds_the_maximum(tr):
    words = cb.words_from_transcript(tr)
    s0, _ = sentence_span(tr, 0)
    _, e8 = sentence_span(tr, 8)
    b = cb.refine(s0, e8, words, 120, cfg(max_duration=20.0))
    assert b["end"] - b["start"] <= 20.0 + 1e-6
    assert "incomplete_sentence_end" not in b["warnings"]  # cut on a sentence end


def test_one_sentence_longer_than_the_maximum_is_flagged():
    long = " ".join(["word"] * 80) + "."
    t = make_transcript([long])
    words = cb.words_from_transcript(t)
    b = cb.refine(0.0, 25.0, words, 40.0, cfg(max_duration=20.0))
    assert b["end"] - b["start"] <= 20.0 + 1e-6
    assert "incomplete_sentence_end" in b["warnings"]


# --- offsets and reproducibility ----------------------------------------------------

def test_timestamp_offsets_are_preserved():
    base, late = make_transcript(), make_transcript(offset=3600.0)
    w0, w1 = cb.words_from_transcript(base), cb.words_from_transcript(late)
    ai_start = word_at(base, 2, 4)["start"]
    ai_end = word_at(base, 5, 3)["end"]
    a = cb.refine(ai_start, ai_end, w0, 200, cfg())
    b = cb.refine(ai_start + 3600.0, ai_end + 3600.0, w1, 3800, cfg())
    assert b["start"] - a["start"] == pytest.approx(3600.0, abs=1e-3)
    assert b["end"] - a["end"] == pytest.approx(3600.0, abs=1e-3)
    assert b["words"] == a["words"]


def test_refinement_is_reproducible(tr):
    words = cb.words_from_transcript(tr)
    one = cb.refine(5.1, 30.2, words, 120, cfg())
    two = cb.refine(5.1, 30.2, cb.words_from_transcript(tr), 120, cfg())
    assert one == two
    assert one["ai"] == [5.1, 30.2]


def test_no_transcript_keeps_the_ai_edges():
    b = cb.refine(3.0, 40.0, [], 100, cfg())
    assert (b["start"], b["end"]) == (3.0, 40.0)
    assert b["warnings"] == ["no_transcript"]


def test_env_switches(monkeypatch):
    monkeypatch.delenv("CLIP_MIN_SECONDS", raising=False)
    monkeypatch.delenv("CLIP_MAX_SECONDS", raising=False)
    monkeypatch.delenv("CLIP_REFINE", raising=False)
    c = cb.config_from_env()
    assert c.enabled and c.llm and c.audio
    assert (c.min_duration, c.max_duration, c.pad_start, c.pad_end) == (20.0, 60.0, 2.0, 2.0)
    monkeypatch.setenv("CLIP_REFINE", "0")
    monkeypatch.setenv("CLIP_REFINE_LLM", "off")
    monkeypatch.setenv("CLIP_MIN_SECONDS", "30")
    c = cb.config_from_env()
    assert not c.enabled and not c.llm and c.min_duration == 30.0


# --- captions after the edges move ----------------------------------------------------

def test_captions_stay_in_sync_after_refinement(tr):
    words = cb.words_from_transcript(tr)
    ai_start = word_at(tr, 2, 4)["start"] + 0.1
    b = cb.refine(ai_start, sentence_span(tr, 5)[1], words, 120, cfg())
    blocks = _collect_word_blocks(tr, b["start"], b["end"])
    flat = [w for blk in blocks for w in blk]
    first = word_at(tr, 2, 0)
    assert flat[0]["word"].lower().startswith("i")
    assert flat[0]["start"] == pytest.approx(first["start"] - b["start"])
    last = tr["segments"][5]["words"][-1]
    assert flat[-1]["end"] == pytest.approx(last["end"] - b["start"])
    assert all(w["start"] >= 0 and w["end"] <= b["end"] - b["start"] + 1e-6 for w in flat)


# --- LLM review (mocked) --------------------------------------------------------------

def _flagged_clip(tr, words):
    s5, _ = sentence_span(tr, 5)
    _, e7 = sentence_span(tr, 7)
    b = cb.refine(s5, e7, words, 120, cfg())
    assert "missing_context" in b["warnings"]
    return {"start": b["start"], "end": b["end"], "boundaries": b,
            "video_title_for_youtube_short": "The dog ate the cake"}


def test_llm_review_moves_edges_to_transcript_sentences(tr):
    words = cb.words_from_transcript(tr)
    clip = _flagged_clip(tr, words)
    prompts = []

    def call(prompt, schema):
        prompts.append(prompt)
        return {"clips": [{"clip": 0, "verdict": "adjust", "first_sentence": 3,
                           "last_sentence": 6, "reason": "needs the missing cake"}]}, {"total_cost": 0.0}

    costs = cb.review_flagged([clip], words, 120, cfg(), call)
    assert len(prompts) == 1 and "[3]" in prompts[0]
    s3, _ = sentence_span(tr, 3)
    _, e6 = sentence_span(tr, 6)
    b = clip["boundaries"]
    assert b["speech"] == [pytest.approx(s3), pytest.approx(e6)]
    assert (clip["start"], clip["end"]) == (b["start"], b["end"])
    assert b["llm_reviewed"] and "missing_context" not in b["warnings"]
    assert costs == [{"total_cost": 0.0}]


def test_llm_review_ignores_ids_it_was_not_given(tr):
    words = cb.words_from_transcript(tr)
    clip = _flagged_clip(tr, words)
    before = dict(clip["boundaries"])

    def call(prompt, schema):
        return {"clips": [{"clip": 0, "verdict": "adjust", "first_sentence": 99,
                           "last_sentence": 120, "reason": "made up"}]}, None

    cb.review_flagged([clip], words, 120, cfg(), call)
    assert clip["boundaries"]["start"] == before["start"]
    assert clip["boundaries"]["end"] == before["end"]


def test_llm_review_can_flag_and_never_fails_the_job(tr):
    words = cb.words_from_transcript(tr)
    clip = _flagged_clip(tr, words)
    cb.review_flagged([clip], words, 120, cfg(), lambda p, s: (
        {"clips": [{"clip": 0, "verdict": "incoherent", "first_sentence": 5,
                    "last_sentence": 7, "reason": "no setup"}]}, None))
    assert "llm_incoherent" in clip["boundaries"]["warnings"]

    other = _flagged_clip(tr, words)

    def boom(prompt, schema):
        raise RuntimeError("server down")

    assert cb.review_flagged([other], words, 120, cfg(), boom) == []


def test_clean_clips_cost_no_llm_call(tr):
    words = cb.words_from_transcript(tr)
    s2, _ = sentence_span(tr, 2)
    _, e4 = sentence_span(tr, 4)
    b = cb.refine(s2, e4, words, 120, cfg())
    assert b["warnings"] == []
    calls = []
    cb.review_flagged([{"start": b["start"], "end": b["end"], "boundaries": b}], words, 120,
                      cfg(), lambda p, s: calls.append(p))
    assert calls == []


# --- audio ---------------------------------------------------------------------------

def _boundaries(speech=(10.0, 40.0), pad=(2.0, 2.0)):
    return {"refined": True, "speech": list(speech), "pad": list(pad),
            "start": speech[0] - pad[0], "end": speech[1] + pad[1],
            "reasons": [], "warnings": [], "audio_checked": False}


def test_audio_keeps_edges_that_are_already_quiet():
    b = cb.apply_audio(_boundaries(), [(7.0, 10.0), (40.0, 43.0)])
    assert (b["start"], b["end"]) == (8.0, 42.0)
    assert b["warnings"] == [] and b["audio_checked"]


def test_audio_moves_edges_into_a_pause():
    # Sound (a laugh, music) right after the last word, then a pause; the
    # tail window ends while the sound is going again.
    b = cb.apply_audio(_boundaries(), [(9.0, 10.0), (40.2, 41.0)])
    assert 9.0 <= b["start"] <= 10.0
    assert 40.2 <= b["end"] <= 41.0
    assert "speech_cutoff_end" not in b["warnings"]


def test_audio_flags_a_cut_where_sound_goes_on():
    b = cb.apply_audio(_boundaries(), [(7.0, 10.0)])
    assert "speech_cutoff_end" in b["warnings"]
    assert b["end"] == 42.0


def test_no_pauses_at_all_changes_nothing():
    b = cb.apply_audio(_boundaries(), [])
    assert (b["start"], b["end"], b["warnings"]) == (8.0, 42.0, [])


def test_audio_flags_long_silence_inside_the_speech():
    b = cb.apply_audio(_boundaries(), [(7.0, 10.0), (20.0, 25.0), (40.0, 43.0)])
    assert "excessive_silence" in b["warnings"]


def test_audio_pass_runs_once():
    b = cb.apply_audio(_boundaries(), [(9.0, 10.0), (40.2, 41.0)])
    again = cb.apply_audio(dict(b, reasons=list(b["reasons"])), [(7.0, 10.0)])
    assert (again["start"], again["end"]) == (b["start"], b["end"])


def test_parse_silences():
    stderr = ("[silencedetect @ 0x1] silence_start: 0.5\n"
              "[silencedetect @ 0x1] silence_end: 1.25 | silence_duration: 0.75\n"
              "[silencedetect @ 0x1] silence_start: 3.0\n")
    assert cb.parse_silences(stderr, offset=100.0, region_end=105.0) == [
        (100.5, 101.25), (103.0, 105.0)]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_detect_silences_on_real_speech():
    silences = cb.detect_silences(FIXTURE, 0.0, 12.0)
    assert silences is not None
    assert all(0.0 <= a < b <= 12.0 + 1e-3 for a, b in silences)
    # Speech has pauses, but is not one long silence.
    assert silences and sum(b - a for a, b in silences) < 9.0


def test_silence_threshold_follows_the_loudness():
    assert cb.silence_threshold(-14.0) == -30.0
    assert cb.silence_threshold(-60.0) == -50.0
    assert cb.silence_threshold(0.0) == -20.0


def test_a_pause_folded_into_a_word_ends_the_sentence():
    # Plain whisper: no gaps, the pause lives inside "okay".
    t = {"segments": [{"start": 0.0, "end": 6.0, "text": "", "words": [
        {"word": " thank", "start": 0.0, "end": 0.3},
        {"word": " you", "start": 0.3, "end": 0.5},
        {"word": " okay", "start": 0.5, "end": 3.9},
        {"word": " so", "start": 3.9, "end": 4.2},
        {"word": " next", "start": 4.2, "end": 4.5},
        {"word": " question.", "start": 4.5, "end": 5.0}]}]}
    words = cb.words_from_transcript(t)
    assert [w["w"].strip() for w in words if w["eos"]] == ["okay", "question."]


def test_capitalised_word_after_a_pause_starts_a_sentence():
    t = make_transcript(["a fight between two sisters What does it look like?"])
    seg = t["segments"][0]["words"]
    for w in seg[5:]:                       # a 0.4 s pause before "What"
        w["start"] += 0.4
        w["end"] += 0.4
    words = cb.words_from_transcript(t)
    assert [w["w"].strip() for w in words if w["eos"]] == ["sisters", "like?"]

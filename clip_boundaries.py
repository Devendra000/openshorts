"""Smart clip boundaries: turn the model's proposed [start, end] into a clip
that starts at the start of a sentence, ends when its last sentence (and the
punchline) is finished, and keeps a little breathing room on both sides.

Three passes, cheapest first, each one optional:

1. ``refine`` (deterministic, transcript only): sentence boundaries come from
   the words' punctuation, the transcript's own segments (WhisperX splits them
   by sentence) and long pauses. The start moves back to the start of the
   sentence it lands in (or forward to the next one when going back would add
   too much), the end moves forward to the end of its sentence, then up to
   ``pad_start`` / ``pad_end`` seconds of the surrounding silence are added,
   never reaching into a neighbouring word. Speech is never cut to hit a
   length: a clip that cannot be made whole inside the band is kept and
   flagged.
2. ``review_flagged`` (LLM, only for clips still flagged): the model sees the
   numbered sentences around the clip and picks the first and last sentence
   by id. It never gives a timestamp: every time comes from the transcript.
3. ``apply_audio`` (ffmpeg ``silencedetect`` around each edge): moves a padded
   edge into a real pause and flags edges where the audio is still going.

Everything a pass decides is written to the clip's ``boundaries`` dict (AI
range, refined range, speech span, padding, reasons, warnings), and every
input is in the job (transcript, source, env), so a re-run gives the same
result. ``CLIP_REFINE=0`` turns the whole thing off and the clip goes through
the classic ``clip_selection.snap_clip_to_words`` instead.

Configuration (env, read per job):
  CLIP_REFINE                 1 (default) | 0
  CLIP_REFINE_PAD_START       seconds of lead-in, default 2.0
  CLIP_REFINE_PAD_END         seconds of tail, default 2.0
  CLIP_REFINE_MIN_SECONDS     default: CLIP_MIN_SECONDS when set, else 20
  CLIP_REFINE_MAX_SECONDS     default: the job's max clip length (60)
  CLIP_REFINE_LLM             auto (default: review flagged clips) | off
  CLIP_REFINE_AUDIO           1 (default) | 0
"""
import os
import re
import subprocess
from dataclasses import asdict, dataclass

VERSION = 1

# Words that make a clip opening with them depend on what came before.
_CONTEXT_OPENERS = {"and", "but", "or", "because", "which", "also", "plus",
                    "then", "though", "although", "whereas"}
_TERMINAL_RX = re.compile(r"[.?!…。？！]+[\"'”’)\]]*$")
_PHRASE_RX = re.compile(r"[,;:—–-]+[\"'”’)\]]*$")

# Warnings a later pass (the LLM review) can still fix.
REVIEWABLE = ("incomplete_sentence_start", "incomplete_sentence_end", "missing_context")


@dataclass
class Config:
    enabled: bool = True
    pad_start: float = 2.0
    pad_end: float = 2.0
    min_duration: float = 20.0
    max_duration: float = 60.0
    max_start_extend: float = 10.0   # how far back the start may move for context
    max_end_extend: float = 12.0     # how far forward the end may move to finish
    max_trim: float = 5.0            # how much speech an edge may drop instead
    guard: float = 0.08              # never pad closer than this to another word
    sentence_gap: float = 1.2        # a pause this long ends a sentence
    segment_gap: float = 0.5         # a segment end followed by this ends one
    unpunctuated_gap: float = 0.6    # ...or this, where no punctuation is near
    capital_gap: float = 0.25        # pause before a capitalised word that ends one
    edge_tolerance: float = 0.3      # word times are this rough at the edges
    excessive_silence: float = 3.0
    llm: bool = True
    audio: bool = True


def _float_env(name, default):
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def _flag_env(name, default=True):
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def config_from_env():
    from clip_selection import clip_duration_bounds
    lo, hi = clip_duration_bounds()
    default_min = lo if os.environ.get("CLIP_MIN_SECONDS") else 20.0
    min_d = _float_env("CLIP_REFINE_MIN_SECONDS", default_min)
    max_d = _float_env("CLIP_REFINE_MAX_SECONDS", hi)
    max_d = min(max(max_d, 10.0), 180.0)
    min_d = min(max(min_d, 5.0), max_d)
    return Config(
        enabled=_flag_env("CLIP_REFINE"),
        pad_start=min(max(_float_env("CLIP_REFINE_PAD_START", 2.0), 0.0), 5.0),
        pad_end=min(max(_float_env("CLIP_REFINE_PAD_END", 2.0), 0.0), 5.0),
        min_duration=min_d,
        max_duration=max_d,
        llm=(os.environ.get("CLIP_REFINE_LLM") or "auto").strip().lower() != "off",
        audio=_flag_env("CLIP_REFINE_AUDIO"),
    )


# --- words and sentences -------------------------------------------------------

def words_from_transcript(transcript, cfg=None):
    """Flat word list ``{'w','s','e','eos','phrase'}`` in time order: ``eos``
    marks the last word of a sentence, ``phrase`` a softer break (comma, a
    half-second pause)."""
    cfg = cfg or Config()
    flat = []
    for seg in (transcript or {}).get("segments") or []:
        seg_words = [w for w in seg.get("words") or []
                     if str(w.get("word") or "").strip()]
        for i, w in enumerate(seg_words):
            try:
                s, e = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            flat.append({"w": str(w["word"]), "s": s, "e": max(e, s),
                         "seg_last": i == len(seg_words) - 1})
    flat.sort(key=lambda w: w["s"])
    punct = [bool(_TERMINAL_RX.search(w["w"].strip())) for w in flat]
    # Running count of terminal punctuation, to ask "is there any near here?"
    seen, total = [], 0
    for p in punct:
        total += p
        seen.append(total)

    def _punctuated_near(i, reach=40):
        lo, hi = max(0, i - reach), min(len(flat) - 1, i + reach)
        return seen[hi] - (seen[lo - 1] if lo else 0) > 0

    for i, w in enumerate(flat):
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        gap = (nxt["s"] - w["e"]) if nxt else float("inf")
        # Plain whisper glues words end to start and folds a pause into the
        # word before it ("okay" lasting 3 s): count that stretch as pause.
        gap += _stretch(w)
        seg_last = w.pop("seg_last")
        eos = punct[i] or gap >= cfg.sentence_gap or (seg_last and gap >= cfg.segment_gap)
        # Whisper drops punctuation in stretches (fast crosstalk, shouting):
        # there a shorter pause has to do.
        if not eos and gap >= cfg.unpunctuated_gap and not _punctuated_near(i):
            eos = True
        # "...my sisters What does it look like?": a capitalised word after a
        # pause starts a sentence the transcript did not close.
        if not eos and nxt and gap >= cfg.capital_gap and _starts_sentence(nxt["w"]):
            eos = True
        w["eos"] = eos
        w["phrase"] = eos or bool(_PHRASE_RX.search(w["w"].strip())) or gap >= 0.5
    return flat


def _stretch(word):
    """Seconds a word lasts beyond a slow speaking rate."""
    letters = len(re.sub(r"\W+", "", word["w"]))
    return max(0.0, (word["e"] - word["s"]) - (0.4 + 0.1 * letters))


_NOT_SENTENCE_STARTS = {"I", "I'm", "I've", "I'll", "I'd", "OK", "Okay"}


def _starts_sentence(word):
    token = word.strip().strip("\"'“‘(")
    return bool(token) and token[0].isupper() and token not in _NOT_SENTENCE_STARTS \
        and not token.isupper()


def sentences(words):
    """``[(first_index, last_index), ...]`` covering every word."""
    out, first = [], 0
    for i, w in enumerate(words):
        if w["eos"] or i == len(words) - 1:
            out.append((first, i))
            first = i + 1
    return out


def _sentence_starts(words):
    return {0} | {i + 1 for i, w in enumerate(words[:-1]) if w["eos"]}


def _sentence_ends(words):
    return {i for i, w in enumerate(words) if w["eos"]} | {len(words) - 1}


def _bare(word):
    return re.sub(r"[^\w']+", "", word.lower())


# --- pass 1: deterministic refinement ---------------------------------------------

def _padding(words, si, ei, cfg, video_duration):
    """Lead and tail (seconds) around the speech span words[si]..words[ei]:
    at most the configured padding, never into a neighbouring word, inside the
    media, and shrunk so the clip stays within ``max_duration``."""
    prev_end = words[si - 1]["e"] if si > 0 else None
    next_start = words[ei + 1]["s"] if ei + 1 < len(words) else None
    lead_room = (words[si]["s"] - prev_end - cfg.guard) if prev_end is not None else words[si]["s"]
    tail_room = (next_start - words[ei]["e"] - cfg.guard) if next_start is not None \
        else (float(video_duration) - words[ei]["e"] if video_duration else cfg.pad_end)
    lead = max(0.0, min(cfg.pad_start, lead_room))
    tail = max(0.0, min(cfg.pad_end, tail_room))
    span = words[ei]["e"] - words[si]["s"]
    excess = span + lead + tail - cfg.max_duration
    if excess > 0 and lead + tail > 0:
        keep = max(0.0, (lead + tail - excess) / (lead + tail))
        lead, tail = lead * keep, tail * keep
    return round(lead, 3), round(tail, 3)


def _edge_candidates(i0, k0, words, cfg):
    starts, ends = _sentence_starts(words), _sentence_ends(words)
    s_cands = [(i0, i0 in starts)]
    if i0 not in starts:
        back = max(j for j in starts if j <= i0)
        if words[i0]["s"] - words[back]["s"] <= cfg.max_start_extend:
            s_cands.append((back, True))
        fwd = min((j for j in starts if i0 < j <= k0), default=None)
        if fwd is not None and words[fwd]["s"] - words[i0]["s"] <= cfg.max_trim:
            s_cands.append((fwd, True))
    e_cands = [(k0, k0 in ends)]
    if k0 not in ends:
        fwd = min(j for j in ends if j >= k0)
        if words[fwd]["e"] - words[k0]["e"] <= cfg.max_end_extend:
            e_cands.append((fwd, True))
        back = max((j for j in ends if i0 <= j < k0), default=None)
        if back is not None and words[k0]["e"] - words[back]["e"] <= cfg.max_trim:
            e_cands.append((back, True))
    return s_cands, e_cands


def _choose(words, s_cands, e_cands, ai_start, ai_end, cfg):
    """Best (si, ei, start_ok, end_ok): whole sentences first, then a clip that
    reaches the minimum, then the most of the AI's range kept, then the
    smallest change. Only spans that fit ``max_duration`` qualify."""
    best = None
    for si, s_ok in s_cands:
        for ei, e_ok in e_cands:
            if ei < si:
                continue
            a, b = words[si]["s"], words[ei]["e"]
            span = b - a
            if span > cfg.max_duration:
                continue
            overlap = max(0.0, min(b, ai_end) - max(a, ai_start))
            change = abs(a - ai_start) + abs(b - ai_end)
            key = ((not s_ok) + (not e_ok), span < cfg.min_duration, -round(overlap, 3),
                   round(change, 3), si, ei)
            if best is None or key < best[0]:
                best = (key, si, ei, s_ok, e_ok)
    return None if best is None else best[1:]


def _fit_end(words, si, cfg):
    """The latest end that keeps a span from words[si] within max: a sentence
    end when one fits, else the last word that fits."""
    ends = _sentence_ends(words)
    fitting = [j for j in range(si, len(words))
               if words[j]["e"] - words[si]["s"] <= cfg.max_duration]
    if not fitting:
        return si, False
    whole = [j for j in fitting if j in ends]
    return (max(whole), True) if whole else (max(fitting), False)


def refine(ai_start, ai_end, words, video_duration, cfg=None):
    """Refine one clip. ``words`` from ``words_from_transcript``. Returns the
    ``boundaries`` dict; its ``start`` / ``end`` are the clip's new edges."""
    cfg = cfg or Config()
    ai_start, ai_end = float(ai_start), float(ai_end)
    out = {"version": VERSION, "refined": True, "ai": [round(ai_start, 3), round(ai_end, 3)],
           "start": round(ai_start, 3), "end": round(ai_end, 3), "speech": None,
           "pad": [0.0, 0.0], "reasons": [], "warnings": [], "llm_reviewed": False,
           "audio_checked": False,
           "config": {k: v for k, v in asdict(cfg).items() if k not in ("enabled",)}}
    if not words:
        out["warnings"].append("no_transcript")
        return out
    i0 = next((i for i, w in enumerate(words) if w["e"] > ai_start + 0.05), None)
    k0 = next((k for k in range(len(words) - 1, -1, -1) if words[k]["s"] < ai_end - 0.05), None)
    if i0 is None or k0 is None or k0 < i0:
        out["warnings"].append("no_speech")
        return out

    s_cands, e_cands = _edge_candidates(i0, k0, words, cfg)
    chosen = _choose(words, s_cands, e_cands, ai_start, ai_end, cfg)
    if chosen is None:
        # Even the AI's own words do not fit: keep its start, cut at the
        # last sentence end (or word) that fits.
        si, s_ok = s_cands[0]
        ei, e_ok = _fit_end(words, si, cfg)
        out["reasons"].append("end moved back to fit the maximum length")
    else:
        si, ei, s_ok, e_ok = chosen

    # Too short: add whole sentences after it, then before it, inside max.
    if words[ei]["e"] - words[si]["s"] < cfg.min_duration:
        for j in sorted(j for j in _sentence_ends(words) if j > ei):
            if words[j]["e"] - words[si]["s"] > cfg.max_duration:
                break
            ei, e_ok = j, True
            if words[ei]["e"] - words[si]["s"] >= cfg.min_duration:
                break
        if words[ei]["e"] - words[si]["s"] < cfg.min_duration:
            for j in sorted((j for j in _sentence_starts(words) if j < si), reverse=True):
                if words[ei]["e"] - words[j]["s"] > cfg.max_duration:
                    break
                si, s_ok = j, True
                if words[ei]["e"] - words[si]["s"] >= cfg.min_duration:
                    break
        if words[ei]["e"] - words[si]["s"] >= cfg.min_duration:
            out["reasons"].append("added whole sentences to reach the minimum length")

    _finish(out, words, si, ei, s_ok, e_ok, video_duration, cfg, i0, k0)
    return out


def _finish(out, words, si, ei, s_ok, e_ok, video_duration, cfg, i0=None, k0=None):
    """Write the chosen span (with padding, reasons and warnings) into ``out``."""
    speech_start, speech_end = words[si]["s"], words[ei]["e"]
    lead, tail = _padding(words, si, ei, cfg, video_duration)
    out["speech"] = [round(speech_start, 3), round(speech_end, 3)]
    out["pad"] = [lead, tail]
    out["start"] = round(max(0.0, speech_start - lead), 3)
    end = speech_end + tail
    if video_duration:
        end = min(end, float(video_duration))
    out["end"] = round(end, 3)
    out["words"] = [si, ei]

    ai_start, ai_end = out["ai"]
    if i0 is not None:
        if si < i0:
            out["reasons"].append(f"start moved back {ai_start - out['start']:.1f}s "
                                  f"to the start of the sentence")
        elif si > i0:
            out["reasons"].append(f"start moved forward {out['start'] - ai_start:.1f}s "
                                  f"to the next sentence")
    if k0 is not None:
        if ei > k0:
            out["reasons"].append(f"end extended {out['end'] - ai_end:.1f}s "
                                  f"to finish the sentence")
        elif ei < k0:
            out["reasons"].append(f"end moved back {ai_end - out['end']:.1f}s "
                                  f"to the end of the last whole sentence")
    out["reasons"].append(f"padding {lead:.1f}s before, {tail:.1f}s after the speech")

    warns = [w for w in out["warnings"] if w not in (
        "incomplete_sentence_start", "incomplete_sentence_end", "missing_context",
        "too_short", "excessive_silence")]
    if not s_ok:
        warns.append("incomplete_sentence_start")
    if not e_ok:
        warns.append("incomplete_sentence_end")
    if _bare(words[si]["w"]) in _CONTEXT_OPENERS:
        warns.append("missing_context")
    if out["end"] - out["start"] < cfg.min_duration:
        warns.append("too_short")
    if any(words[j + 1]["s"] - words[j]["e"] > cfg.excessive_silence for j in range(si, ei)):
        warns.append("excessive_silence")
    out["warnings"] = warns


def disabled(ai_start, ai_end, final):
    """``boundaries`` for a clip that went through the classic word snap."""
    return {"version": VERSION, "refined": False,
            "ai": [round(float(ai_start), 3), round(float(ai_end), 3)],
            "start": final[0], "end": final[1], "reasons": ["smart boundaries off"],
            "warnings": []}


# --- pass 2: LLM review of the clips still flagged -----------------------------------

def _review_schema():
    from typing import List
    from pydantic import BaseModel

    class BoundaryReviewItem(BaseModel):
        clip: int
        verdict: str          # keep | adjust | incoherent
        first_sentence: int
        last_sentence: int
        reason: str

    class BoundaryReviewResponse(BaseModel):
        clips: List[BoundaryReviewItem]

    return BoundaryReviewResponse


def _review_prompt(items, sents, words, cfg):
    blocks = []
    for n, (clip, lo, hi, cur_first, cur_last) in enumerate(items):
        lines = []
        for sid in range(lo, hi + 1):
            a, b = sents[sid]
            text = "".join(w["w"] for w in words[a:b + 1]).strip()
            mark = " <-- current clip" if cur_first <= sid <= cur_last else ""
            lines.append(f"  [{sid}] ({words[a]['s']:.1f}s-{words[b]['e']:.1f}s) {text}{mark}")
        title = (clip.get("video_title_for_youtube_short") or "").strip()[:120]
        blocks.append(f"CLIP {n} (title: {title!r}; currently sentences "
                      f"{cur_first}-{cur_last}):\n" + "\n".join(lines))
    return (
        "You check the edges of short vertical video clips cut from a longer video.\n"
        "For each clip you get the numbered transcript sentences around it, with "
        "their times. Pick the FIRST and LAST sentence id of the clip so that it:\n"
        "- starts with enough context to be understood on its own (it does not "
        "open on a reference to something it never shows);\n"
        "- contains the whole interesting moment, punchline or answer included;\n"
        "- ends naturally, on a finished thought.\n"
        f"The clip (last sentence end minus first sentence start) must be between "
        f"{cfg.min_duration:.0f} and {cfg.max_duration:.0f} seconds. Use only ids from "
        "the list. Keep the current sentences when they are already right.\n"
        "verdict: 'keep' (current edges are fine), 'adjust' (you moved them) or "
        "'incoherent' (no choice of sentences makes this a coherent clip).\n\n"
        + "\n\n".join(blocks)
        + "\n\nAnswer JSON: {\"clips\": [{\"clip\": n, \"verdict\": ..., "
          "\"first_sentence\": id, \"last_sentence\": id, \"reason\": short}]}"
    )


def _sentence_of(sents, index):
    for sid, (a, b) in enumerate(sents):
        if a <= index <= b:
            return sid
    return None


def review_flagged(shorts, words, video_duration, cfg, call, context_seconds=25.0,
                   per_call=6):
    """Ask the LLM about clips whose ``boundaries`` still carry a reviewable
    warning. ``call(prompt, schema) -> (parsed, cost)``. Returns the costs.
    Answers are validated: unknown ids, reversed ranges or spans outside the
    band are ignored, and the times always come from ``words``."""
    if not words:
        return []
    sents = sentences(words)
    flagged = []
    for clip in shorts:
        b = clip.get("boundaries") or {}
        if not b.get("refined") or not b.get("words") or b.get("llm_reviewed"):
            continue
        if not any(w in REVIEWABLE for w in b.get("warnings") or []):
            continue
        si, ei = b["words"]
        cur_first, cur_last = _sentence_of(sents, si), _sentence_of(sents, ei)
        lo = cur_first
        while lo > 0 and words[sents[lo - 1][0]]["s"] >= words[si]["s"] - context_seconds:
            lo -= 1
        hi = cur_last
        while hi + 1 < len(sents) and words[sents[hi + 1][1]]["e"] <= words[ei]["e"] + context_seconds:
            hi += 1
        flagged.append((clip, lo, hi, cur_first, cur_last))

    costs = []
    schema = _review_schema()
    for at in range(0, len(flagged), per_call):
        batch = flagged[at:at + per_call]
        try:
            parsed, cost = call(_review_prompt(batch, sents, words, cfg), schema)
        except Exception as e:  # the review is a bonus: never fail the job on it
            print(f"   ⚠️ Boundary review skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        if cost:
            costs.append(cost)
        answers = {}
        for item in (parsed or {}).get("clips") or []:
            try:
                answers[int(item["clip"])] = item
            except (KeyError, TypeError, ValueError):
                continue
        for n, (clip, lo, hi, cur_first, cur_last) in enumerate(batch):
            b = clip["boundaries"]
            b["llm_reviewed"] = True
            item = answers.get(n)
            if not item:
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            reason = str(item.get("reason") or "").strip()[:200]
            if verdict == "incoherent":
                b["warnings"].append("llm_incoherent")
                b["reasons"].append(f"LLM review: {reason or 'no coherent edges'}")
                continue
            try:
                first, last = int(item["first_sentence"]), int(item["last_sentence"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo <= first <= last <= hi) or (first, last) == (cur_first, cur_last):
                continue
            si, ei = sents[first][0], sents[last][1]
            span = words[ei]["e"] - words[si]["s"]
            if span > cfg.max_duration or span < min(cfg.min_duration, b["end"] - b["start"]):
                continue
            _finish(b, words, si, ei, True, True, video_duration, cfg)
            b["reasons"].append(f"LLM review moved the edges to sentences {first}-{last}"
                                + (f": {reason}" if reason else ""))
            clip["start"], clip["end"] = b["start"], b["end"]
    return costs


# --- pass 3: audio -------------------------------------------------------------------

_SIL_START_RX = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END_RX = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silences(stderr, offset=0.0, region_end=None):
    """``silencedetect`` output as absolute ``[(start, end), ...]``."""
    out, open_at = [], None
    for line in (stderr or "").splitlines():
        m = _SIL_START_RX.search(line)
        if m:
            open_at = float(m.group(1))
            continue
        m = _SIL_END_RX.search(line)
        if m and open_at is not None:
            out.append((round(max(0.0, open_at) + offset, 3), round(float(m.group(1)) + offset, 3)))
            open_at = None
    if open_at is not None and region_end is not None:
        out.append((round(max(0.0, open_at) + offset, 3), round(float(region_end), 3)))
    return out


_MEAN_RX = re.compile(r"mean_volume:\s*(-?[\d.]+)\s*dB")


def _ffmpeg_audio(media_path, start, length, af, timeout):
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
           "-i", media_path, "-map", "0:a:0", "-vn", "-af", af, "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stderr if r.returncode == 0 else None


def silence_threshold(mean_db, below=16.0):
    """Pause threshold relative to the region's mean loudness: a fixed
    -35 dB finds no pause at all in a recording with room tone, a quiet one
    calls every breath a pause. Clamped to -50..-20 dB."""
    return max(-50.0, min(-20.0, float(mean_db) - below))


def detect_silences(media_path, start, end, min_silence=0.2, timeout=120):
    """Pauses in ``media_path`` between ``start`` and ``end`` (absolute
    seconds), or None when ffmpeg cannot read the audio."""
    start = max(0.0, float(start))
    length = max(0.1, float(end) - start)
    stats = _ffmpeg_audio(media_path, start, length, "volumedetect", timeout)
    if stats is None:
        return None
    m = _MEAN_RX.search(stats)
    if not m:
        return None
    noise = silence_threshold(float(m.group(1)))
    out = _ffmpeg_audio(media_path, start, length,
                        f"silencedetect=noise={noise:.1f}dB:d={float(min_silence):.2f}", timeout)
    if out is None:
        return None
    return parse_silences(out, offset=start, region_end=start + length)


def _silent_at(t, silences):
    return any(a <= t <= b for a, b in silences)


def _quiet_near(t, silences, tolerance):
    """Any pause within ``tolerance`` of ``t``: the word's own timestamp may
    be that far off, so a pause there means the cut is between sounds."""
    return any(a <= t + tolerance and b >= t - tolerance for a, b in silences)


def apply_audio(boundaries, silences, video_duration=None, cfg=None):
    """Move padded edges into real pauses and flag edges where sound goes on.
    ``silences`` covers at least the clip plus its padding. No pause anywhere
    (music bed, noisy room) says nothing: the edges stay as they are."""
    cfg = cfg or Config()
    b = boundaries
    if not b.get("refined") or not b.get("speech") or b.get("audio_checked"):
        return b
    b["audio_checked"] = True
    if not silences:
        b["reasons"].append("audio: no pauses detected, edges kept")
        return b
    speech_start, speech_end = b["speech"]
    lead, tail = b["pad"]
    warns = b["warnings"]

    # START: the cut should be quiet. Otherwise use the earliest quiet point
    # in the lead-in window, or flag it.
    lo, hi = speech_start - lead, speech_start
    if lead > 0 and not _silent_at(lo, silences):
        inside = [(a, s_end) for a, s_end in silences if s_end >= lo and a <= hi]
        if inside:
            a, _ = inside[0]
            new = round(max(lo, a + 0.03), 3)
            b["reasons"].append(f"audio: start moved {new - b['start']:.2f}s into a pause")
            b["start"] = new
        elif not _quiet_near(speech_start, silences, cfg.edge_tolerance):
            warns.append("speech_cutoff_start")
    elif lead == 0 and not _quiet_near(speech_start, silences, cfg.edge_tolerance):
        warns.append("speech_cutoff_start")

    # END: the latest quiet point in the tail window, or flag it.
    lo, hi = speech_end, speech_end + tail
    if tail > 0 and not _silent_at(hi, silences):
        inside = [(a, s_end) for a, s_end in silences if s_end >= lo and a <= hi]
        if inside:
            a, s_end = inside[-1]
            new = round(min(hi, s_end - 0.03), 3)
            new = max(new, round(max(lo, a), 3))
            b["reasons"].append(f"audio: end moved {new - b['end']:.2f}s into a pause")
            b["end"] = new
        elif not _quiet_near(speech_end, silences, cfg.edge_tolerance):
            warns.append("speech_cutoff_end")
    elif tail == 0 and not _quiet_near(speech_end, silences, cfg.edge_tolerance):
        warns.append("speech_cutoff_end")

    if video_duration:
        b["end"] = min(b["end"], round(float(video_duration), 3))
    if "excessive_silence" not in warns and any(
            min(e, speech_end) - max(a, speech_start) > cfg.excessive_silence
            for a, e in silences):
        warns.append("excessive_silence")
    return b


def refine_clips_with_audio(shorts, media_path, video_duration, cfg=None):
    """Run ``apply_audio`` on every refined clip of a job (one short ffmpeg
    read per clip, around its edges only)."""
    cfg = cfg or config_from_env()
    if not cfg.enabled or not cfg.audio or not media_path:
        return
    for clip in shorts or []:
        b = clip.get("boundaries")
        if not b or not b.get("refined") or b.get("audio_checked") or not b.get("speech"):
            continue
        silences = detect_silences(media_path, b["start"] - 1.0, b["end"] + 1.0)
        if silences is None:
            continue
        apply_audio(b, silences, video_duration, cfg)
        clip["start"], clip["end"] = b["start"], b["end"]

"""
Recut engine for the clip editor: a per-clip EDL (list of source segments) is
turned into a rendered clip.

The pure helpers here are standard-library only (plus ffmpeg_utils, which is
also light) so the module imports in the thin CI environment; everything heavy
(reframing, watermark, auto-captions live in main.py) is imported lazily and
only on the paths that need it.

Two render paths:

- FAST: every segment falls inside the range the canonical clip was originally
  cut from, so the recut can be cut straight out of the already-reframed
  canonical file. No ML, no source video needed, seconds instead of minutes.
- SOURCE: at least one segment reaches outside that range (or there is no
  canonical file), so the recut is cut from the source video and re-reframed
  with the same engine the pipeline used.
"""

import json
import os
import shutil
import subprocess
import time
import uuid

from ffmpeg_utils import (METADATA_SCRUB, QUALITY_FAST, audio_encode_args,
                          video_encode_args)

# EDL limits. Deliberately generous — the editor is for humans fixing cuts,
# not for stitching feature films.
MAX_SEGMENTS = 12
MIN_SEGMENT_SECONDS = 0.5
MAX_TOTAL_SECONDS = 180.0

# Segments may start/end a hair outside the canonical range through float
# round-tripping; treat them as inside.
RANGE_TOLERANCE = 0.05

# Hybrid render: when most of a new cut is already in a rendered file and only
# a few stretches need the source, only those stretches are reframed and
# spliced in, instead of reframing the whole cut again (a 0.5 s extension of a
# 35 s clip used to cost the full 35 s of scene detection, tracking and
# render). Each stretch is reframed with some context on both sides, so scene
# detection and the tracker see the shot rather than a sliver of it.
HYBRID_PAD_SECONDS = 2.0
HYBRID_MAX_SOURCE_PIECES = 3
# Past this share of the cut, the padded windows cost about as much as the
# full render, which also keeps one continuous camera.
HYBRID_MAX_SHARE = 0.6


class RecutError(ValueError):
    """Invalid EDL — safe to surface verbatim as a 400 detail."""


def normalize_segments(segments, source_duration=None):
    """Validate and clamp an EDL. Returns [{'start': float, 'end': float}, ...].

    Order is preserved — the segment order IS the clip order, and reusing a
    source range twice is legal (an echo/replay is a real editing move).
    """
    if not isinstance(segments, (list, tuple)) or not segments:
        raise RecutError("segments must be a non-empty list")
    if len(segments) > MAX_SEGMENTS:
        raise RecutError(f"too many segments (max {MAX_SEGMENTS})")

    normalized = []
    for i, seg in enumerate(segments):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            raise RecutError(f"segment {i + 1}: start/end must be numbers")
        if start != start or end != end:  # NaN guard
            raise RecutError(f"segment {i + 1}: start/end must be numbers")
        start = max(0.0, start)
        if source_duration is not None:
            end = min(float(source_duration), end)
            start = min(start, float(source_duration))
        if end - start < MIN_SEGMENT_SECONDS:
            raise RecutError(
                f"segment {i + 1} is shorter than {MIN_SEGMENT_SECONDS}s")
        normalized.append({"start": round(start, 3), "end": round(end, 3)})

    if total_duration(normalized) > MAX_TOTAL_SECONDS:
        raise RecutError(f"clip would exceed {MAX_TOTAL_SECONDS:.0f}s")
    return normalized


def total_duration(segments):
    return round(sum(s["end"] - s["start"] for s in segments), 3)


def within_range(segments, range_start, range_end, tolerance=RANGE_TOLERANCE):
    """True when every segment fits inside [range_start, range_end]."""
    return all(
        s["start"] >= float(range_start) - tolerance
        and s["end"] <= float(range_end) + tolerance
        for s in segments
    )


def rebase_segments(segments, range_start, range_end=None):
    """Map source-absolute segments onto a file cut at ``range_start``.

    Used by the fast path: the canonical clip's t=0 is the source's
    ``range_start``. Clamps to the file bounds so tolerance-admitted segments
    never produce negative seek times.
    """
    rebased = []
    for seg in segments:
        start = max(0.0, seg["start"] - float(range_start))
        end = seg["end"] - float(range_start)
        if range_end is not None:
            end = min(end, float(range_end) - float(range_start))
        rebased.append({"start": round(start, 3), "end": round(end, 3)})
    return rebased


def plan_pieces(segments, bases, tolerance=RANGE_TOLERANCE):
    """Where each stretch of a new cut can come from, in clip order.

    ``bases``: [(path, base_segments)], best first. Each is a rendered file
    whose timeline is ``base_segments`` (source-absolute) laid back to back.
    Returns pieces ``{"path", "start", "end"}``: times ON ``path`` for a
    stretch a rendered file already holds, or ``path`` None and SOURCE times
    for a stretch only the source has. Gaps of up to ``tolerance`` count as
    covered (float round-tripping), so no source piece is shorter than that.
    """
    spans = []
    for path, base_segments in bases:
        offset = 0.0
        for b in base_segments:
            b_start, b_end = float(b["start"]), float(b["end"])
            if b_end > b_start:
                spans.append((b_start, b_end, path, offset))
                offset += b_end - b_start

    pieces = []
    for seg in segments:
        cursor, end = float(seg["start"]), float(seg["end"])
        while end - cursor > tolerance:
            holding = [sp for sp in spans if sp[0] - tolerance <= cursor < sp[1]]
            if holding:
                # The best base that holds the rest of the segment, else the
                # span reaching furthest (earliest base on a tie).
                whole = [sp for sp in holding if end <= sp[1] + tolerance]
                b_start, b_end, path, offset = (
                    whole[0] if whole else max(holding, key=lambda sp: sp[1]))
                stop = end if end <= b_end + tolerance else b_end
                pieces.append({
                    "path": path,
                    "start": round(offset + max(0.0, cursor - b_start), 3),
                    "end": round(offset + min(stop, b_end) - b_start, 3)})
            else:
                later = [sp[0] for sp in spans if cursor < sp[0] < end]
                stop = min(later) if later else end
                pieces.append({"path": None, "start": round(cursor, 3),
                               "end": round(stop, 3)})
            cursor = stop

    merged = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if (last and last["path"] == piece["path"]
                and abs(last["end"] - piece["start"]) < 0.002):
            last["end"] = piece["end"]
        else:
            merged.append(dict(piece))
    return merged


def hybrid_window(piece, source_duration=None, pad=HYBRID_PAD_SECONDS):
    """The source stretch reframed for a source piece: the piece plus context."""
    start = max(0.0, float(piece["start"]) - pad)
    end = float(piece["end"]) + pad
    if source_duration is not None:
        end = min(float(source_duration), end)
    return {"start": round(start, 3), "end": round(end, 3)}


def worth_hybrid(pieces, source_duration=None):
    """True when splicing reframed source stretches into already-rendered
    footage is clearly cheaper than reframing the whole cut again."""
    from_source = [p for p in pieces if p["path"] is None]
    if not from_source or len(from_source) == len(pieces):
        return False
    if len(from_source) > HYBRID_MAX_SOURCE_PIECES:
        return False
    window = sum(w["end"] - w["start"] for w in
                 (hybrid_window(p, source_duration) for p in from_source))
    return window < HYBRID_MAX_SHARE * total_duration(pieces)


def snap_segments(segments, transcript, source_duration):
    """Snap each segment's bounds onto word boundaries (ground truth beats
    millisecond arithmetic — same rationale as the pipeline's snapping)."""
    from clip_selection import snap_clip_to_words

    words = transcript_words(transcript)
    if not words:
        return segments
    snapped = []
    for seg in segments:
        start, end = snap_clip_to_words(
            seg["start"], seg["end"], words, source_duration,
            min_duration=MIN_SEGMENT_SECONDS, max_duration=MAX_TOTAL_SECONDS)
        snapped.append({"start": start, "end": end})
    return snapped


def transcript_words(transcript):
    """Flatten a Whisper transcript to [{'w','s','e'}, ...] sorted by start."""
    words = []
    for segment in (transcript or {}).get("segments", []):
        for w in segment.get("words", []) or []:
            try:
                words.append({
                    "w": str(w.get("word", "")).strip(),
                    "s": float(w["start"]),
                    "e": float(w["end"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    words.sort(key=lambda w: w["s"])
    return words


def virtual_transcript(transcript, segments):
    """Remap a source-absolute transcript onto the concatenated clip timeline.

    Each EDL segment becomes one synthetic transcript segment whose words are
    shifted so t=0 is the start of the recut clip. This is what lets
    auto-captions and subtitle restyles work on multi-segment clips: they keep
    slicing "words between clip_start and clip_end" exactly as before, against
    this transcript with clip_start=0.
    """
    out_segments = []
    offset = 0.0
    for seg in segments:
        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        seg_duration = seg_end - seg_start
        words = []
        for w in transcript_words(transcript):
            if w["e"] <= seg_start or w["s"] >= seg_end:
                continue
            words.append({
                # Leading space = Whisper's word-boundary convention.
                # transcript_words() strips it, and without it the caption
                # block collector treats every word as a continuation fragment
                # and burns the whole line glued together.
                "word": " " + w["w"],
                "start": round(max(0.0, w["s"] - seg_start) + offset, 3),
                "end": round(min(seg_duration, w["e"] - seg_start) + offset, 3),
            })
        out_segments.append({
            "start": round(offset, 3),
            "end": round(offset + seg_duration, 3),
            "text": "".join(w["word"] for w in words).strip(),
            "words": words,
        })
        offset += seg_duration
    return {
        "language": (transcript or {}).get("language", "en"),
        "segments": out_segments,
    }


def cut_commands(input_path, segments, part_paths, video_filter=None):
    """ffmpeg argv for each segment cut. Re-encodes for frame-accurate cuts
    with uniform parameters so the parts concat cleanly. ``input_path`` is
    one file for every segment, or a list with one file per segment;
    ``video_filter`` (built here, never from stored data) conforms parts from
    different renders to one size and frame rate."""
    inputs = (input_path if isinstance(input_path, (list, tuple))
              else [input_path] * len(segments))
    commands = []
    for seg, part, source in zip(segments, part_paths, inputs):
        commands.append([
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-i", source,
            *(["-vf", video_filter] if video_filter else []),
            *video_encode_args(QUALITY_FAST),
            *audio_encode_args(),
            # Every final-artifact producer in the repo scrubs source metadata
            # and fronts the moov atom (a fast-path recut IS the delivered
            # file, and without +faststart the browser preview hangs). Also on
            # intermediate parts: harmless, and it keeps the source's handler
            # metadata from ever entering the chain.
            *METADATA_SCRUB, "-movflags", "+faststart",
            part,
        ])
    return commands


def concat_command(list_path, out_path):
    """Concat demuxer over identically-encoded parts — stream copy, no
    generation loss on the join."""
    return [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy",
        *METADATA_SCRUB, "-movflags", "+faststart", out_path,
    ]


def run_cut_concat(input_path, segments, out_path, workdir, runner=None,
                   video_filter=None):
    """Cut every segment from ``input_path`` (one file, or one per segment)
    and join them into ``out_path``."""
    run = runner or _run_ffmpeg
    if len(segments) == 1:
        run(cut_commands(input_path, segments, [out_path], video_filter)[0])
        return out_path

    # Unique per invocation: two concurrent recuts in the same job dir must
    # not overwrite each other's parts (or delete them via the finally below).
    token = uuid.uuid4().hex[:8]
    part_paths = [
        os.path.join(workdir, f"temp_recut_part_{token}_{i}.mp4")
        for i in range(len(segments))
    ]
    list_path = os.path.join(workdir, f"temp_recut_concat_{token}.txt")
    try:
        for command in cut_commands(input_path, segments, part_paths,
                                    video_filter):
            run(command)
        with open(list_path, "w") as f:
            for part in part_paths:
                # Absolute paths: the concat demuxer resolves relative entries
                # against the LIST FILE's directory, not the process cwd.
                f.write(f"file '{os.path.abspath(part)}'\n")
        run(concat_command(list_path, out_path))
    finally:
        for path in part_paths + [list_path]:
            if os.path.exists(path):
                os.remove(path)
    return out_path


# Same ceiling as apply_watermark's: a hung ffmpeg must not pin the executor
# thread (and the caller's quota reservation) until a server restart.
FFMPEG_TIMEOUT_SECONDS = 1800


def _run_ffmpeg(command):
    try:
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s ({command[1:6]}...)")
    if result.returncode != 0:
        tail = (result.stderr or b"").decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"ffmpeg failed ({command[1:6]}...): {tail}")


def perform_recut(*, input_path, segments, output_dir, clean_name,
                  reframe=False, output_format="auto", watermark=False,
                  captions_transcript=None, force_strategy=None,
                  crop_overrides=None, runner=None, renderer=None,
                  watermarker=None, captioner=None, subtitle_band=None,
                  pieces=None, source_duration=None, reuse_input=False):
    """Render a recut clip. Returns (served_filename, clean_filename).

    - ``input_path``/``segments``: the file to cut from and the times ON THAT
      FILE (the caller rebases for the fast path).
    - ``reframe``: run the reframe engine on the joined cut (source path only —
      the canonical file is already framed).
    - ``watermark``: re-apply the free-plan watermark (source path only — the
      canonical file already carries it).
    - ``captions_transcript``: a clip-relative transcript (see
      ``virtual_transcript``); when given and non-empty, captions are burned
      LAST onto a ``subtitled_<ts>_`` derivative, preserving the invariant
      that the clean file stays clean for later re-styling.
    - ``crop_overrides``: scene index -> crop centre as a fraction of the
      source width, for scenes the user framed by hand. Source path only, for
      the same reason as ``reframe``: the canonical file is already cropped, so
      its framing can no longer be changed.
    - ``reuse_input``: ``input_path`` already IS the cut (same segments, same
      framing): nothing is cut or encoded, only the captions are burned again.
      Returns that file as the clean name.
    - ``pieces`` (see ``plan_pieces``): splice the cut together from files
      that already hold most of it; ``input_path`` is then the source, and
      only the source pieces are reframed (each in its ``hybrid_window``,
      clamped to ``source_duration``). ``segments`` stays the source cut, for
      the caption timeline.

    The renderer/watermarker/captioner hooks default to main.py's
    implementations, imported lazily so this module stays importable without
    the ML stack; tests inject fakes.
    """
    if reuse_input:
        clean = os.path.basename(input_path)
        served = clean
        if captions_transcript and captions_transcript.get("segments"):
            caption = captioner or _main_attr("auto_caption_clip")
            captioned = caption(input_path, captions_transcript,
                                0.0, total_duration(segments))
            if captioned:
                served = os.path.basename(captioned)
        return served, clean

    # The uuid token keeps two same-second saves of one clip from writing (and
    # then serving) the same filename; the timestamp keeps "newest derived
    # file" resolution working in _canonical_clip_file.
    out_name = f"recut_{int(time.time())}_{uuid.uuid4().hex[:6]}_{clean_name}"
    out_path = os.path.join(output_dir, out_name)
    work_name = f"temp_{out_name}"
    work_path = os.path.join(output_dir, work_name)

    def render(i, o, f):
        if renderer is not None:
            return renderer(i, o, f)  # injected fakes keep the 3-arg contract
        return _main_attr("render_clip")(
            i, o, f, force_strategy=force_strategy,
            crop_overrides=crop_overrides, subtitle_band=subtitle_band)

    try:
        if pieces:
            _splice_pieces(input_path, pieces, work_path, out_path, output_dir,
                           output_format, render, runner, source_duration)
            shutil.move(work_path, out_path)
        elif reframe:
            run_cut_concat(input_path, segments, work_path, output_dir,
                           runner=runner)
            if not render(work_path, out_path, output_format):
                raise RuntimeError("reframe failed on the recut clip")
        else:
            run_cut_concat(input_path, segments, work_path, output_dir,
                           runner=runner)
            shutil.move(work_path, out_path)
            # No reframe means no fresh layout sidecar; the captions would fall
            # back to the bottom on a stacked clip. Carry the input's layout
            # ranges through the cut instead (empty when the input has none).
            import layout_ranges
            layout_ranges.write(out_path, [
                (r["start"], r["end"], r["layout"], r.get("content"))
                for r in layout_ranges.remap(layout_ranges.read(input_path), segments)])

        if watermark:
            (watermarker or _main_attr("apply_watermark"))(out_path)

        served_name = out_name
        if captions_transcript and captions_transcript.get("segments"):
            caption = captioner or _main_attr("auto_caption_clip")
            captioned = caption(out_path, captions_transcript,
                                0.0, total_duration(segments))
            if captioned:
                served_name = os.path.basename(captioned)
        return served_name, out_name
    finally:
        if os.path.exists(work_path):
            os.remove(work_path)


def _splice_pieces(source_path, pieces, work_path, out_path, workdir,
                   output_format, render, runner, source_duration):
    """Join ``pieces`` into ``work_path``, reframing only the source ones, and
    write ``out_path``'s layout sidecar from the layouts of the files used."""
    import layout_ranges

    token = uuid.uuid4().hex[:8]
    temps = []
    parts = []
    try:
        for i, piece in enumerate(pieces):
            if piece["path"] is not None:
                parts.append(piece)
                continue
            window = hybrid_window(piece, source_duration)
            cut = os.path.join(workdir, f"temp_recut_win_{token}_{i}.mp4")
            framed = os.path.join(workdir, f"temp_recut_framed_{token}_{i}.mp4")
            temps += [cut, framed, layout_ranges.sidecar_path(framed)]
            run_cut_concat(source_path, [window], cut, workdir, runner=runner)
            if not render(cut, framed, output_format):
                raise RuntimeError("reframe failed on a recut stretch")
            parts.append({"path": framed,
                          "start": round(piece["start"] - window["start"], 3),
                          "end": round(piece["end"] - window["start"], 3)})

        reference = next((p["path"] for p in pieces if p["path"]), None)
        run_cut_concat([p["path"] for p in parts],
                       [{"start": p["start"], "end": p["end"]} for p in parts],
                       work_path, workdir, runner=runner,
                       video_filter=_conform_filter(reference))

        ranges, offset = [], 0.0
        for part in parts:
            for r in layout_ranges.remap(layout_ranges.read(part["path"]),
                                         [part]):
                ranges.append((r["start"] + offset, r["end"] + offset,
                               r["layout"], r.get("content")))
            offset += part["end"] - part["start"]
        layout_ranges.write(out_path, ranges)
    finally:
        for path in temps:
            if os.path.exists(path):
                os.remove(path)


def _conform_filter(reference_path):
    """scale/fps filter matching ``reference_path`` (one of our own renders),
    so stretches reframed now concat cleanly with frames rendered earlier.
    None when it cannot be probed: the parts are encoded alike anyway."""
    if not reference_path:
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", reference_path],
            capture_output=True, timeout=30)
        stream = json.loads(result.stdout or b"{}")["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        num, den = (int(v) for v in str(stream["r_frame_rate"]).split("/"))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError,
            IndexError, TypeError):
        return None
    if width <= 0 or height <= 0 or num <= 0 or den <= 0:
        return None
    return f"scale={width}:{height},setsar=1,fps={num}/{den}"


def _main_attr(name):
    import main  # heavy — resolved only when a real render/caption runs
    return getattr(main, name)

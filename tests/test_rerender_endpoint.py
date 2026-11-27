"""HTTP tests for the clip editor's backend: GET .../edl and POST /api/clip/rerender.

Same conventions as test_mcp_endpoint.py: a real ASGI round-trip against the
imported app (BILLING_ENABLED=0 via conftest, so the self-host branch runs and
no cloud auth or database is involved), with the actual render work stubbed at
the recut.perform_recut seam — these tests own the ENDPOINT contract: request
validation, fast/source path choice, 409 on a gone source, and the persistence
of the recipe into metadata.json and the in-memory job.
"""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
import recut  # noqa: E402

JOB_ID = "recut-endpoint-test-job"

TRANSCRIPT = {
    "language": "en",
    "segments": [
        {
            "start": 0.0, "end": 60.0, "text": "hello world again",
            "words": [
                {"word": "hello", "start": 12.0, "end": 12.5},
                {"word": "world", "start": 20.0, "end": 20.4},
                {"word": "again", "start": 50.0, "end": 50.5},
            ],
        },
    ],
}


def _request(method, path, json_body=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, json=json_body)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    """A completed job on disk + in memory: canonical clip, metadata with
    transcript, and a retained source video in the job dir (URL-job style)."""
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))

    clip = {
        "start": 10.0,
        "end": 40.0,
        "video_title_for_youtube_short": "test clip",
        "video_url": f"/videos/{JOB_ID}/mytitle_clip_1.mp4",
    }
    meta = {
        "shorts": [clip],
        "transcript": TRANSCRIPT,
        "source_video": "src.mp4",
        "output_format": "auto",
        "cost_analysis": {},
    }
    (job_dir / "mytitle_metadata.json").write_text(json.dumps(meta))
    (job_dir / "mytitle_clip_1.mp4").write_bytes(b"canonical")
    (job_dir / "src.mp4").write_bytes(b"source")

    app_module.jobs[JOB_ID] = {
        "status": "completed",
        "logs": [],
        "result": {"clips": [dict(clip)], "cost_analysis": {}},
        "user_id": None,
        "watermark": False,
    }
    try:
        yield {"dir": job_dir, "meta_path": job_dir / "mytitle_metadata.json"}
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def fake_recut(monkeypatch):
    """Stub the render seam; records kwargs and materializes the output file."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        name = f"recut_1_{kwargs['clean_name']}"
        with open(os.path.join(kwargs["output_dir"], name), "wb") as f:
            f.write(b"recut")
        return name, name

    monkeypatch.setattr(recut, "perform_recut", fake)
    return calls


class TestGetEdl:
    def test_synthesizes_recipe_for_pristine_clip(self, job):
        resp = _request("GET", f"/api/clip/{JOB_ID}/0/edl")
        assert resp.status_code == 200
        data = resp.json()
        assert data["segments"] == [{"start": 10.0, "end": 40.0}]
        assert data["canonical_range"] == {"start": 10.0, "end": 40.0}
        assert data["duration"] == 30.0
        assert data["current_file"] == "mytitle_clip_1.mp4"
        assert data["has_captions"] is False
        assert data["source"]["available"] is True
        assert data["source"]["url"] == f"/api/source/{JOB_ID}"
        # The editor gets the whole source transcript, not a window around
        # the clip: a cut can only be extended into words it was sent.
        assert [w["w"] for w in data["words"]] == ["hello", "world", "again"]
        assert data["limits"]["max_segments"] == recut.MAX_SEGMENTS
        # Self-host meters nothing.
        assert data["rerender_minutes"] == 0

    def test_ships_words_far_outside_the_clip(self, job):
        """The EDL carries the WHOLE source transcript. The old ±45s window
        would drop these words (clip ends at 40s, they start at 200s), and a
        cut can only be extended into material whose words were sent."""
        meta = json.loads(job["meta_path"].read_text())
        meta["transcript"]["segments"].append({
            "start": 200.0, "end": 260.0, "text": "distant tail",
            "words": [
                {"word": "distant", "start": 200.0, "end": 200.4},
                {"word": "tail", "start": 259.0, "end": 259.5},
            ],
        })
        job["meta_path"].write_text(json.dumps(meta))

        data = _request("GET", f"/api/clip/{JOB_ID}/0/edl").json()
        assert [w["w"] for w in data["words"]] == [
            "hello", "world", "again", "distant", "tail"]
        # Word order is the binary-search invariant the editor's follow-along
        # highlight depends on.
        starts = [w["s"] for w in data["words"]]
        assert starts == sorted(starts)

    def test_reports_missing_source(self, job):
        os.remove(job["dir"] / "src.mp4")
        data = _request("GET", f"/api/clip/{JOB_ID}/0/edl").json()
        assert data["source"]["available"] is False
        assert data["source"]["url"] is None
        assert data["source"]["duration_estimated"] is True

    def test_404s(self, job):
        assert _request("GET", "/api/clip/nope/0/edl").status_code == 404
        assert _request("GET", f"/api/clip/{JOB_ID}/7/edl").status_code == 404


class TestRerenderValidation:
    def test_unknown_job_404(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": "nope", "clip_index": 0,
            "segments": [{"start": 10, "end": 20}]})
        assert resp.status_code == 404

    def test_clip_index_out_of_range_404(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 5,
            "segments": [{"start": 10, "end": 20}]})
        assert resp.status_code == 404

    def test_invalid_segments_400(self, job, fake_recut):
        for segments in ([], [{"start": 15, "end": 15.1}]):
            resp = _request("POST", "/api/clip/rerender", {
                "job_id": JOB_ID, "clip_index": 0, "segments": segments})
            assert resp.status_code == 400, segments
        assert fake_recut == []

    def test_gone_source_with_outside_segment_409(self, job, fake_recut):
        os.remove(job["dir"] / "src.mp4")
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 45, "end": 55}]})
        assert resp.status_code == 409
        assert fake_recut == []

    def test_invalid_framing_400(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "framing": "cinematic",
            "segments": [{"start": 12, "end": 30}]})
        assert resp.status_code == 400
        assert fake_recut == []

    def test_framing_without_source_409(self, job, fake_recut):
        os.remove(job["dir"] / "src.mp4")
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "framing": "full",
            "segments": [{"start": 12, "end": 30}]})  # in-range, but framing needs source
        assert resp.status_code == 409
        assert "framing" in resp.json()["detail"]
        assert fake_recut == []

    def test_snap_clamp_inverted_segment_400_not_500(self, job, fake_recut,
                                                     monkeypatch):
        # No source + snap_to_words: a segment fully outside the canonical
        # range clamps to an inverted (end < start) window. That must be
        # rejected as a 400 by the post-snap re-validation, never reach
        # ffmpeg as `-ss 150 -to 30` and surface as a 500.
        os.remove(job["dir"] / "src.mp4")
        monkeypatch.setattr(recut, "snap_segments",
                            lambda segments, transcript, bound: segments)
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "snap_to_words": True,
            "segments": [{"start": 200, "end": 210}]})
        assert resp.status_code == 400
        assert fake_recut == []


class TestRerenderPaths:
    def test_fast_path_cuts_from_canonical_clip(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 12, "end": 22}, {"start": 30, "end": 35}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["render_path"] == "fast"
        assert data["new_video_url"] == f"/videos/{JOB_ID}/recut_1_mytitle_clip_1.mp4"
        assert data["duration"] == 15.0

        call = fake_recut[0]
        assert call["input_path"].endswith("mytitle_clip_1.mp4")
        assert call["reframe"] is False
        # Segments arrive rebased onto the canonical file (t=0 is source 10s).
        assert call["segments"] == [{"start": 2.0, "end": 12.0},
                                    {"start": 20.0, "end": 25.0}]
        # Captions re-applied by default, against the clip-relative transcript.
        assert call["captions_transcript"]["segments"][0]["words"]

    def test_source_path_reframes_from_retained_source(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 45, "end": 55}]})
        assert resp.status_code == 200
        assert resp.json()["render_path"] == "source"
        call = fake_recut[0]
        assert call["input_path"].endswith("src.mp4")
        assert call["reframe"] is True
        assert call["output_format"] == "auto"
        # The free-plan mark is applied at delivery (app._deliver), never
        # inside the recut, so the served twins stay clean for an upgrade.
        assert not call.get("watermark")
        # Source-absolute times, not rebased.
        assert call["segments"] == [{"start": 45.0, "end": 55.0}]

    def test_reapply_captions_false_skips_the_transcript(self, job, fake_recut):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "reapply_captions": False,
            "segments": [{"start": 12, "end": 22}]})
        assert resp.status_code == 200
        assert fake_recut[0]["captions_transcript"] is None

    def test_framing_full_forces_source_path_and_persists(self, job, fake_recut):
        # In-range segments would take the fast path, but a framing override
        # must re-reframe from the source with the forced layout.
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "framing": "full",
            "segments": [{"start": 12, "end": 22}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["render_path"] == "source"
        assert data["framing"] == "full"
        assert data["recipe"]["framing"] == "full"
        call = fake_recut[0]
        assert call["input_path"].endswith("src.mp4")
        assert call["reframe"] is True
        assert call["force_strategy"] == "WIDE"

        # A follow-up trim WITHOUT a framing field inherits the recipe's
        # framing (the look must not silently revert on a plain trim), and is
        # cut from the WIDE render it trims instead of reframing again...
        resp2 = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 13, "end": 21}]})
        assert resp2.status_code == 200
        assert resp2.json()["framing"] == "full"
        assert resp2.json()["render_path"] == "fast"
        assert fake_recut[1]["input_path"].endswith("recut_1_mytitle_clip_1.mp4")
        assert fake_recut[1]["segments"] == [{"start": 1.0, "end": 9.0}]

        # ...while an explicit 'auto' resets to the classifier and the fast path.
        resp3 = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "framing": "auto",
            "segments": [{"start": 13, "end": 21}]})
        assert resp3.status_code == 200
        assert resp3.json()["render_path"] == "fast"
        assert "framing" not in resp3.json()["recipe"]


class TestRerenderReuse:
    """A cut is taken from the files that already hold it; only footage none
    of them holds is reframed, and only that."""

    def _post(self, segments):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "segments": segments})
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_small_extension_reframes_only_the_new_stretch(self, job, fake_recut):
        data = self._post([{"start": 10, "end": 41}])
        assert data["render_path"] == "hybrid"
        call = fake_recut[0]
        assert call["input_path"].endswith("src.mp4")
        assert call["pieces"] == [
            {"path": str(job["dir"] / "mytitle_clip_1.mp4"), "start": 0.0, "end": 30.0},
            {"path": None, "start": 40.0, "end": 41.0}]
        assert data["recipe"]["rendered"] == "recut_1_mytitle_clip_1.mp4"

    def test_trim_after_an_extension_cuts_from_the_last_render(self, job, fake_recut):
        self._post([{"start": 30, "end": 50}])  # mostly new: whole render
        assert fake_recut[0]["reframe"] is True and not fake_recut[0].get("pieces")
        data = self._post([{"start": 42, "end": 48}])
        assert data["render_path"] == "fast"
        assert fake_recut[1]["input_path"].endswith("recut_1_mytitle_clip_1.mp4")
        assert fake_recut[1]["segments"] == [{"start": 12.0, "end": 18.0}]

    def test_unchanged_cut_only_reburns_captions(self, job, fake_recut):
        # A fixed transcript word: same cut, same framing. Nothing is cut or
        # encoded; the file already rendered gets its captions again.
        data = self._post([{"start": 10, "end": 40}])
        assert data["render_path"] == "captions"
        call = fake_recut[0]
        assert call["reuse_input"] is True
        assert call["input_path"].endswith("mytitle_clip_1.mp4")
        assert call["captions_transcript"]["segments"][0]["words"]

    def test_a_stored_rendered_name_is_never_a_path(self, job, fake_recut):
        meta = json.loads(job["meta_path"].read_text())
        meta["shorts"][0]["recipe"] = {
            "v": 1, "segments": [{"start": 10, "end": 40}],
            "canonical_range": {"start": 10, "end": 40},
            "rendered": "../../src.mp4"}
        job["meta_path"].write_text(json.dumps(meta))
        self._post([{"start": 12, "end": 20}])
        assert fake_recut[0]["input_path"].endswith("mytitle_clip_1.mp4")


class TestRerenderPersistence:
    SEGMENTS = [{"start": 10, "end": 15}, {"start": 48, "end": 52}]

    def _rerender(self):
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0, "segments": self.SEGMENTS})
        assert resp.status_code == 200
        return resp.json()

    def test_recipe_lands_in_metadata_and_memory(self, job, fake_recut):
        data = self._rerender()
        expected_segments = [{"start": 10.0, "end": 15.0},
                             {"start": 48.0, "end": 52.0}]
        assert data["recipe"]["segments"] == expected_segments
        assert data["recipe"]["canonical_range"] == {"start": 10.0, "end": 40.0}
        # Covering range, so downstream start/end stay a sane positive window.
        assert (data["start"], data["end"]) == (10.0, 52.0)

        meta = json.loads(job["meta_path"].read_text())
        stored = meta["shorts"][0]
        assert stored["recipe"]["segments"] == expected_segments
        assert stored["video_url"] == data["new_video_url"]
        mem = app_module.jobs[JOB_ID]["result"]["clips"][0]
        assert mem["recipe"]["segments"] == expected_segments
        assert mem["video_url"] == data["new_video_url"]

    def test_edl_reflects_the_recut(self, job, fake_recut):
        self._rerender()
        data = _request("GET", f"/api/clip/{JOB_ID}/0/edl").json()
        assert data["segments"] == [{"start": 10.0, "end": 15.0},
                                    {"start": 48.0, "end": 52.0}]
        # The canonical range survives, so the fast path stays available.
        assert data["canonical_range"] == {"start": 10.0, "end": 40.0}

    def test_transcript_endpoint_remaps_piecewise(self, job, fake_recut):
        self._rerender()
        data = _request("GET", f"/api/clip/{JOB_ID}/0/transcript").json()
        assert data["durationSec"] == 9.0
        by_text = {c["text"]: c for c in data["captions"]}
        # "hello" (12.0s in source) → 2.0s into the clip.
        assert by_text["hello"]["startMs"] == 2000
        # "again" (50.0s) sits in the second segment → 5.0 + 2.0 = 7.0s.
        assert by_text["again"]["startMs"] == 7000
        # "world" (20.0s) was cut out.
        assert "world" not in by_text


class TestMcpTool:
    def test_recut_clip_is_registered(self):
        import mcp_server
        names = [t["name"] for t in mcp_server.TOOLS]
        assert "recut_clip" in names
        assert "recut_clip" in mcp_server._TOOL_IMPLS


class TestFixTranscriptWord:
    """A misheard word is corrected at the source: same timestamps, new text,
    and every caption made afterwards says it."""

    def _fix(self, body):
        return _request("POST", f"/api/job/{JOB_ID}/transcript/word", body)

    def test_fixes_the_text_and_keeps_the_timing(self, job):
        resp = self._fix({"start": 20.0, "end": 20.4, "text": "  off   me "})
        assert resp.status_code == 200, resp.text
        assert resp.json()["word"] == {"w": "off me", "s": 20.0, "e": 20.4}
        seg = json.loads(job["meta_path"].read_text())["transcript"]["segments"][0]
        assert seg["words"][1] == {"word": " off me", "start": 20.0, "end": 20.4}
        assert seg["text"] == "hello off me again"
        # The editor's word list (and so every recut's captions) follows.
        data = _request("GET", f"/api/clip/{JOB_ID}/0/edl").json()
        assert [w["w"] for w in data["words"]] == ["hello", "off me", "again"]

    def test_unknown_word_404_and_bad_text_400(self, job):
        assert self._fix({"start": 21.0, "end": 21.4, "text": "x"}).status_code == 404
        assert self._fix({"start": 20.0, "end": 20.4, "text": "   "}).status_code == 400
        assert self._fix({"start": 20.0, "end": 20.4, "text": "a" * 61}).status_code == 400
        assert _request("POST", "/api/job/nope/transcript/word",
                        {"start": 20.0, "end": 20.4, "text": "x"}).status_code == 404

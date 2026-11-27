"""Cancelling a job: queued or running, it stops, its minutes are released,
it is never auto-retried nor counted as a failure, and a running one takes
the processes it started (WhisperX, ffmpeg) down with it."""
import asyncio
import json
import os
import sys
import time

import pytest

app_module = pytest.importorskip("app")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_stopping", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_remote_jobs", set())
    monkeypatch.setattr(app_module, "_admitting", set())
    monkeypatch.setattr(app_module, "_delivering_jobs", set())
    monkeypatch.setattr(app_module, "_job_processes", {})
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    calls = []

    def fake(name):
        async def step(job_id, *a, **k):
            calls.append(name)
        return step

    for attr, name in (("_settle_reservation", "settle"), ("_archive_managed_job", "archive"),
                       ("_unmark_if_owner_paid", "unmark"),
                       ("_autopilot_job_finished", "autopilot"),
                       ("_notify_job_webhook", "webhook"), ("_record_job_alert", "alert"),
                       ("_track_proxy_usage", "proxy"), ("_notify_clips_ready", "email"),
                       ("_notify_clip_activity", "activity")):
        monkeypatch.setattr(app_module, attr, fake(name))
    return root, calls


def _job(root, job_id, cmd, status="queued"):
    d = root / job_id
    d.mkdir()
    (d / app_module._RESUME_FILE).write_text(json.dumps(
        {"cmd": cmd, "priority": 2, "user_id": None, "reservation_id": None,
         "watermark": False, "attempts": 0}))
    job = {"status": status, "logs": [], "cmd": cmd, "env": {}, "output_dir": str(d),
           "result": None}
    app_module.jobs[job_id] = job
    return job


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def test_queued_job_is_dropped_and_released(env):
    root, calls = env

    async def go():
        job = _job(root, "q1", ["python", "main.py"])
        app_module.job_queue.put_nowait((2, 1.0, "q1"))
        out = await app_module.cancel_job("q1", None)
        await asyncio.sleep(0.05)
        return job, out

    job, out = asyncio.run(go())
    assert out == {"status": "cancelled"}
    assert app_module.job_queue.empty()
    assert job["status"] == "failed" and job["logs"][-1] == app_module.CANCELLED_LOG
    assert calls == ["settle", "webhook"]          # no alert, no email
    assert not job.get("delivering")
    assert not os.path.exists(root / "q1" / app_module._RESUME_FILE)
    assert not app_module._should_auto_retry(job)


def test_finished_job_is_left_alone(env):
    root, _calls = env
    job = _job(root, "done", ["python", "main.py"], status="completed")
    out = asyncio.run(app_module.cancel_job("done", None))
    assert out["cancelled"] is False and job["status"] == "completed"


def test_running_job_kills_its_whole_process_tree(env, tmp_path):
    root, calls = env
    pidfile = tmp_path / "child.pid"
    # main.py stand-in that starts a long child, like main.py starts WhisperX.
    script = (f"import subprocess,time;p=subprocess.Popen(['sleep','60']);"
              f"open({str(pidfile)!r},'w').write(str(p.pid));time.sleep(60)")
    cmd = [sys.executable, "-c", script]

    async def go():
        job = _job(root, "r1", cmd)
        app_module.job_queue.put_nowait((2, 1.0, "r1"))
        await app_module.job_queue.get()
        app_module._running_jobs.add("r1")
        await app_module.concurrency_semaphore.acquire()
        wrapper = asyncio.create_task(app_module.run_job_wrapper("r1"))
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text():
                break
            await asyncio.sleep(0.05)
        child = int(pidfile.read_text())
        started = time.monotonic()
        assert (await app_module.cancel_job("r1", None)) == {"status": "cancelled"}
        await asyncio.wait_for(wrapper, 10)
        return job, child, time.monotonic() - started

    job, child, took = asyncio.run(go())
    assert took < 8
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.05)
    assert not _alive(child), "the grandchild (WhisperX stand-in) kept running"
    assert job["status"] == "failed" and job["logs"][-1] == app_module.CANCELLED_LOG
    assert job.get("auto_retries") is None         # not retried as a transient kill
    assert calls == ["settle", "webhook"]
    assert "r1" not in app_module._running_jobs

import concurrent.futures
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import captcha_agent
import captcha_brain
import pure_captcha_farm
import vlm_farm_worker


def _reset_farm_state(monkeypatch, *, same_gap=0.0, cross_seed=0.0, cross_worker=0.0):
    with pure_captcha_farm._LOCK:
        pure_captcha_farm._TASKS.clear()
        pure_captcha_farm._QUEUE.clear()
        pure_captcha_farm._SEED_FP_MAP.clear()
        pure_captcha_farm._RELEASED_SEEDS.clear()
        pure_captcha_farm._SEED_LAST_SOLVE.clear()
        pure_captcha_farm._WORKER_LAST_SOLVE.clear()
    monkeypatch.setattr(pure_captcha_farm, "SAME_STAGE_GAP_SECONDS", same_gap)
    monkeypatch.setattr(pure_captcha_farm, "CROSS_STAGE_SEED_COOLDOWN_SECONDS", cross_seed)
    monkeypatch.setattr(pure_captcha_farm, "CROSS_STAGE_WORKER_COOLDOWN_SECONDS", cross_worker)


def test_farm_queue_roundtrip_and_seed_fp_inheritance(monkeypatch):
    _reset_farm_state(monkeypatch)
    t1 = pure_captcha_farm.enqueue_task(
        email="demo@example.com",
        scenario="register",
        account_seed="seed_alpha",
    )
    claimed = pure_captcha_farm.claim_next_task(worker_id="slot_0")
    assert claimed["status"] == "claimed"
    assert claimed["task_id"] == t1["task_id"]
    assert claimed["account_seed"] == "seed_alpha"

    valid_fp = "B" + "x" * 24
    ok = pure_captcha_farm.complete_task(
        t1["task_id"],
        token="P1_tok_register_" + "x" * 20,
        device_id=valid_fp,
        device_fingerprint='{"source":"test"}',
    )
    assert ok
    assert pure_captcha_farm._TASKS[t1["task_id"]]["device_id"] == valid_fp

    # Login scenario inherits cached fingerprint for same seed and sets stage_switch=True
    t2 = pure_captcha_farm.enqueue_task(
        email="demo@example.com",
        scenario="login",
        account_seed="seed_alpha",
    )
    claimed2 = pure_captcha_farm.claim_next_task(worker_id="slot_0")
    assert claimed2["status"] == "claimed"
    assert claimed2["stage_switch"] is True
    assert pure_captcha_farm.complete_task(t2["task_id"], token="P1_tok_login_" + "y" * 20, device_id="")
    assert pure_captcha_farm._TASKS[t2["task_id"]]["device_id"] == valid_fp


def test_farm_cross_stage_isolation_cooldown(monkeypatch):
    _reset_farm_state(monkeypatch, same_gap=0.0, cross_seed=60.0, cross_worker=60.0)
    t1 = pure_captcha_farm.enqueue_task("a@example.com", scenario="register", account_seed="seed_a")
    t_reg2 = pure_captcha_farm.enqueue_task("b@example.com", scenario="register", account_seed="seed_b")

    c1 = pure_captcha_farm.claim_next_task(worker_id="slot_0")
    assert c1["status"] == "claimed" and c1["task_id"] == t1["task_id"]
    assert pure_captcha_farm.complete_task(t1["task_id"], token="P1_tok1_" + "a" * 20, device_id="B" + "a" * 20)

    # Enqueue cross-stage login while same-stage register for seed_b is also pending
    t2 = pure_captcha_farm.enqueue_task("a@example.com", scenario="login", account_seed="seed_a")
    same_stage = pure_captcha_farm.claim_next_task(worker_id="slot_0")
    assert same_stage["status"] == "claimed"
    assert same_stage["task_id"] == t_reg2["task_id"]
    assert same_stage["stage_switch"] is False
    assert pure_captcha_farm.complete_task(t_reg2["task_id"], token="P1_tok3_" + "c" * 20, device_id="B" + "c" * 20)

    # Even a fresh slot_1 sees cooldown on seed_a login until cross-stage seed cooldown expires
    blocked = pure_captcha_farm.claim_next_task(worker_id="slot_1")
    assert blocked["status"] == "cooldown"
    assert blocked["retry_after"] > 0

    monkeypatch.setattr(pure_captcha_farm, "CROSS_STAGE_SEED_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(pure_captcha_farm, "CROSS_STAGE_WORKER_COOLDOWN_SECONDS", 0.0)
    unblocked = pure_captcha_farm.claim_next_task(worker_id="slot_0")
    assert unblocked["status"] == "claimed"
    assert unblocked["task_id"] == t2["task_id"]
    assert unblocked["stage_switch"] is True
    assert pure_captcha_farm.complete_task(t2["task_id"], token="P1_tok2_" + "b" * 20)


def test_vlm_semaphore_stagger_and_per_page_cdp_locks(monkeypatch):
    captcha_agent.set_vlm_concurrency(2)
    monkeypatch.setattr(captcha_agent, "VLM_STAGGER_SECONDS", 0.0)
    active = 0
    peak = 0
    mu = threading.Lock()

    def _work():
        nonlocal active, peak
        with captcha_agent.vlm_slot():
            with mu:
                active += 1
                peak = max(peak, active)
            time.sleep(0.08)
            with mu:
                active -= 1

    threads = [threading.Thread(target=_work) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert peak == 2

    class _FakePage:
        pass

    p1, p2 = _FakePage(), _FakePage()
    assert captcha_agent._get_cdp_lock(p1) is captcha_agent._get_cdp_lock(p1)
    assert captcha_agent._get_cdp_lock(p1) is not captcha_agent._get_cdp_lock(p2)


def test_captcha_brain_cells_descanline_guard_and_probe(tmp_path, monkeypatch):
    clicked = []
    tools = captcha_brain.Tools(page=None, verbose=False)
    tools.panel = Image.new("RGB", (400, 480), "white")
    monkeypatch.setattr(
        tools,
        "click",
        lambda x, y, space="panel": clicked.append((round(x, 2), round(y, 2))) or {"ok": True},
    )
    res = tools.commit(cells=[6, 8, 9], submit_after=False)
    assert res["ok"] is True
    assert res["cells"] == [6, 8, 9]
    assert clicked == [(0.82, 0.53), (0.5, 0.75), (0.82, 0.75)]

    # Scanline detector + vertical k=10 cleaner
    arr = np.full((300, 300, 3), 210, dtype=np.uint8)
    for y in range(0, 300, 6):
        arr[y : y + 2, :, :] = 40
    striped_im = Image.fromarray(arr, mode="RGB")
    detected, v_diff, h_diff = captcha_brain._has_horizontal_scanlines(striped_im)
    assert detected is True
    assert v_diff > h_diff
    cleaned = captcha_brain._clean_horizontal_scanlines(striped_im)
    assert cleaned.size == striped_im.size

    # Zero-distance drag guard
    brain = captcha_brain.CaptchaBrain(page=None, verbose=False)
    msg = brain._guard("commit", {"tool": "commit", "drag": [0.5, 0.5, 0.51, 0.5]})
    assert isinstance(msg, str) and len(msg) > 0

    # Captcha probe JSONL summary
    probe_path = tmp_path / "captcha_probe.jsonl"
    monkeypatch.setattr(captcha_brain, "PROBE_FILE", str(probe_path))
    captcha_brain.record_captcha_probe({
        "event": "turn",
        "solve_id": "s1",
        "turn": 1,
        "challenge_kind": "drag_letter_scanline",
        "has_scanlines": True,
    })
    captcha_brain.record_captcha_probe({
        "event": "solve_summary",
        "solve_id": "s1",
        "challenge_kind": "drag_letter_scanline",
        "turns": 1,
        "final_ok": True,
        "total_elapsed_s": 21.5,
    })
    summary = captcha_brain.get_captcha_probe_summary()
    assert summary["total_turns"] == 1
    assert summary["total_solves"] == 1
    assert summary["by_kind"]["drag_letter_scanline"]["pass_rate"] == 1.0


def test_worker_respects_stage_switch_isolation_flag(monkeypatch):
    worker = vlm_farm_worker.VlmFarmWorker(farm_url="http://127.0.0.1:9599")
    calls = []
    monkeypatch.setattr(worker, "_ensure_browser", lambda seed, force_isolate=False: calls.append((seed, force_isolate)) or None)
    monkeypatch.setattr(vlm_farm_worker.requests, "post", lambda *a, **kw: None)
    ok = worker.solve_one({"task_id": "t1", "account_seed": "s1", "stage_switch": True})
    assert ok is False
    assert calls == [("s1", True)]

# -*- coding: utf-8 -*-
"""
producer/pure_captcha_farm.py —— 100% 独立纯粹打码农场 (彻底剥离 DeepSeek 官网)
================================================================================
【核心特性】：
  1. 页面中绝无任何 DeepSeek 网页、不打开官网、不填邮箱密码！
  2. 纯粹引入官方 hCaptcha SDK，仅凭 SiteKey 原地渲染真实人机题目。
  3. 线程安全任务队列：支持多路流水线并发向 POST /farm/solve 派单阻塞取票。
  4. 支持【人工网页点选】与【VLM 全自动 Worker】双模无缝衔接：
     - GET  /                  独立题面容器（含 Canvas/WebGL + account_seed 指纹采集）
     - GET  /farm/task_status  返回当前队首待解任务 {task_id, status, email, scenario, account_seed, queue_len}
     - GET  /farm/health       农场健康与排队状态
     - POST /farm/solve        流水线阻塞取票接口
     - POST /farm/submit_token 提交 hCaptcha Token 与实时数美 device_id
     - POST /farm/fail         VLM Worker 失败上报（立即唤醒结束阻塞，避免空等超时）
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 9527

# DeepSeek 官方海外 hCaptcha SiteKey
HCAPTCHA_SITEKEY = os.environ.get("HCAPTCHA_SITEKEY", "352e5376-f2cc-43fe-a744-e51640449610")

_LOCK = threading.Lock()
_TASKS: Dict[str, Dict[str, Any]] = {}
_QUEUE: Deque[str] = deque()
_SEED_FP_MAP: Dict[str, Dict[str, str]] = {}
_RELEASED_SEEDS: set[str] = set()
_SEED_LAST_SOLVE: Dict[str, Tuple[str, float]] = {}
_WORKER_LAST_SOLVE: Dict[str, Tuple[str, float]] = {}

SAME_STAGE_GAP_SECONDS = max(0.0, float(os.environ.get("FARM_SAME_STAGE_GAP_S", "2.0") or "2.0"))
CROSS_STAGE_SEED_COOLDOWN_SECONDS = max(0.0, float(os.environ.get("FARM_CROSS_STAGE_SEED_COOLDOWN_S", "12.0") or "12.0"))
CROSS_STAGE_WORKER_COOLDOWN_SECONDS = max(0.0, float(os.environ.get("FARM_CROSS_STAGE_WORKER_COOLDOWN_S", "15.0") or "15.0"))
CLAIM_TTL_SECONDS = max(60.0, float(os.environ.get("FARM_CLAIM_TTL_S", "240.0") or "240.0"))


def is_valid_shumei_device_id(device_id: Any) -> bool:
    s = str(device_id or "").strip()
    return len(s) >= 16 and s.startswith("B") and not any(ch.isspace() for ch in s)


def _reclaim_expired_claims_locked(now: Optional[float] = None) -> None:
    ts = now if now is not None else time.time()
    for tid in list(_QUEUE):
        t = _TASKS.get(tid)
        if not t:
            continue
        if t.get("status") == "claimed":
            claimed_at = float(t.get("claimed_at") or 0.0)
            if claimed_at > 0 and (ts - claimed_at) > CLAIM_TTL_SECONDS:
                t["status"] = "pending"
                t["claimed_by"] = ""
                t["claimed_at"] = 0.0


def _prune_head_locked() -> Optional[Dict[str, Any]]:
    _reclaim_expired_claims_locked()
    while _QUEUE:
        tid = _QUEUE[0]
        t = _TASKS.get(tid)
        if not t or t.get("status") not in ("pending", "claimed"):
            _QUEUE.popleft()
            continue
        return t
    return None


def enqueue_task(email: str = "", scenario: str = "register", account_seed: str = "") -> Dict[str, Any]:
    tid = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    seed_str = str(account_seed or "")
    task = {
        "task_id": tid,
        "email": str(email or ""),
        "scenario": str(scenario or "register"),
        "account_seed": seed_str,
        "status": "pending",
        "claimed_by": "",
        "claimed_at": 0.0,
        "token": "",
        "device_id": "",
        "device_fingerprint": "",
        "error": "",
        "created_at": time.time(),
        "event": threading.Event(),
    }
    with _LOCK:
        if seed_str:
            _RELEASED_SEEDS.discard(seed_str)
        _TASKS[tid] = task
        _QUEUE.append(tid)
    return task


def _task_wait_seconds_locked(task: Dict[str, Any], wid: str, now: float) -> float:
    """计算某任务在当前 worker 上还需等待的频率隔离冷却秒数（同阶段不同号不跨阶段冷却）。"""
    scen = str(task.get("scenario") or "register")
    seed_str = str(task.get("account_seed") or "").strip()
    waits = [0.0]
    if seed_str and seed_str in _SEED_LAST_SOLVE:
        last_scen, last_ts = _SEED_LAST_SOLVE[seed_str]
        if last_scen and last_scen != scen and CROSS_STAGE_SEED_COOLDOWN_SECONDS > 0:
            waits.append((last_ts + CROSS_STAGE_SEED_COOLDOWN_SECONDS) - now)
    if wid in _WORKER_LAST_SOLVE:
        w_scen, w_ts = _WORKER_LAST_SOLVE[wid]
        if w_ts > 0:
            if w_scen == scen and SAME_STAGE_GAP_SECONDS > 0:
                waits.append((w_ts + SAME_STAGE_GAP_SECONDS) - now)
            elif w_scen and w_scen != scen and CROSS_STAGE_WORKER_COOLDOWN_SECONDS > 0:
                waits.append((w_ts + CROSS_STAGE_WORKER_COOLDOWN_SECONDS) - now)
    return max(waits)


def claim_next_task(worker_id: str = "default", warm_seeds: Optional[list] = None) -> Dict[str, Any]:
    """原子认领一条待处理任务：同阶段不同号优先排队，跨阶段强制冷却与工位隔离。"""
    wid = str(worker_id or "default").strip() or "default"
    warm_set = {str(s).strip() for s in (warm_seeds or []) if str(s).strip()}
    now = time.time()
    with _LOCK:
        _reclaim_expired_claims_locked(now=now)
        released = sorted(_RELEASED_SEEDS)
        _RELEASED_SEEDS.clear()

        chosen: Optional[Dict[str, Any]] = None
        # 若该 worker 已有未完成 claimed 任务，直接幂等返回
        for tid in _QUEUE:
            t = _TASKS.get(tid)
            if t and t.get("status") == "claimed" and t.get("claimed_by") == wid:
                chosen = t
                break

        w_last_scen = (_WORKER_LAST_SOLVE.get(wid) or ("", 0.0))[0]
        eligible: list[Dict[str, Any]] = []
        min_wait: Optional[float] = None
        if chosen is None:
            for tid in _QUEUE:
                t = _TASKS.get(tid)
                if not t or t.get("status") != "pending":
                    continue
                wait_s = _task_wait_seconds_locked(t, wid, now)
                if wait_s <= 0.0:
                    eligible.append(t)
                elif min_wait is None or wait_s < min_wait:
                    min_wait = wait_s

        if chosen is None and eligible:
            # 1) 同阶段（w_last_scen 匹配）优先，让多槽位按阶段自然分流；内部仍优先 warm_seeds 再 FIFO
            if w_last_scen:
                same_stage = [t for t in eligible if t.get("scenario") == w_last_scen]
                if same_stage:
                    if warm_set:
                        for t in same_stage:
                            if t.get("account_seed") in warm_set:
                                chosen = t
                                break
                    if chosen is None:
                        chosen = same_stage[0]

            # 2) 新工位或同阶段空闲（且已过跨阶段冷却）：优先 warm_seeds，其次 FIFO
            if chosen is None and warm_set:
                for t in eligible:
                    if t.get("account_seed") in warm_set:
                        chosen = t
                        break
            if chosen is None:
                chosen = eligible[0]

        pending_count = sum(
            1 for qid in _QUEUE
            if (_TASKS.get(qid) or {}).get("status") in ("pending", "claimed")
        )
        if chosen is None:
            is_cooldown = bool(min_wait is not None and min_wait > 0.0)
            return {
                "task_id": "",
                "status": "cooldown" if is_cooldown else "idle",
                "retry_after": round(min_wait, 2) if is_cooldown and min_wait is not None else 0.0,
                "email": "",
                "scenario": "",
                "account_seed": "",
                "queue_len": pending_count,
                "released_seeds": released,
                "stage_switch": False,
            }

        chosen["status"] = "claimed"
        chosen["claimed_by"] = wid
        chosen["claimed_at"] = now
        seed_str = str(chosen.get("account_seed") or "")
        bound_fp = _SEED_FP_MAP.get(seed_str) or {}
        stage_switch = bool(w_last_scen and w_last_scen != chosen["scenario"])
        return {
            "task_id": chosen["task_id"],
            "status": "claimed",
            "worker_id": wid,
            "email": chosen["email"],
            "scenario": chosen["scenario"],
            "account_seed": seed_str,
            "device_id": bound_fp.get("device_id", ""),
            "device_fingerprint": bound_fp.get("device_fingerprint", ""),
            "queue_len": pending_count,
            "released_seeds": released,
            "stage_switch": stage_switch,
        }


def release_seed(account_seed: str) -> bool:
    seed_str = str(account_seed or "").strip()
    if not seed_str:
        return False
    with _LOCK:
        _RELEASED_SEEDS.add(seed_str)
        _SEED_FP_MAP.pop(seed_str, None)
        _SEED_LAST_SOLVE.pop(seed_str, None)
    return True


def bind_seed_fingerprint(account_seed: str, device_id: str, device_fingerprint: Any = "") -> bool:
    seed_str = str(account_seed or "").strip()
    dev = str(device_id or "").strip()
    if not seed_str or not is_valid_shumei_device_id(dev):
        return False
    fp_str = ""
    if isinstance(device_fingerprint, dict):
        fp_str = json.dumps(device_fingerprint, ensure_ascii=False)
    elif isinstance(device_fingerprint, str):
        fp_str = device_fingerprint.strip()
    with _LOCK:
        _SEED_FP_MAP[seed_str] = {"device_id": dev, "device_fingerprint": fp_str}
    return True


def get_current_pending(worker_id: str = "", task_id: str = "") -> Optional[Dict[str, Any]]:
    wid = str(worker_id or "").strip()
    tid_query = str(task_id or "").strip()
    with _LOCK:
        _reclaim_expired_claims_locked()
        target: Optional[Dict[str, Any]] = None
        if tid_query:
            cand = _TASKS.get(tid_query)
            if not cand or cand.get("status") not in ("pending", "claimed"):
                return None
            target = cand
        elif wid:
            for qid in _QUEUE:
                cand = _TASKS.get(qid)
                if cand and cand.get("status") == "claimed" and cand.get("claimed_by") == wid:
                    target = cand
                    break
            if target is None:
                return None
        else:
            target = _prune_head_locked()
        if not target:
            return None
        pending_count = sum(
            1 for qid in _QUEUE
            if (_TASKS.get(qid) or {}).get("status") in ("pending", "claimed")
        )
        seed_str = str(target.get("account_seed") or "")
        bound_fp = _SEED_FP_MAP.get(seed_str) or {}
        return {
            "task_id": target["task_id"],
            "status": "pending",
            "raw_status": target.get("status", "pending"),
            "claimed_by": target.get("claimed_by", ""),
            "email": target["email"],
            "scenario": target["scenario"],
            "account_seed": seed_str,
            "device_id": bound_fp.get("device_id", ""),
            "device_fingerprint": bound_fp.get("device_fingerprint", ""),
            "queue_len": pending_count,
        }


def complete_task(
    task_id: str,
    token: str,
    device_id: str = "",
    device_fingerprint: str = "",
) -> Optional[str]:
    tok = str(token or "").strip()
    dev = str(device_id or "").strip()
    fp_str = ""
    if isinstance(device_fingerprint, dict):
        fp_str = json.dumps(device_fingerprint, ensure_ascii=False)
    elif isinstance(device_fingerprint, str):
        fp_str = device_fingerprint.strip()
    if not tok:
        return None
    with _LOCK:
        task = _TASKS.get(task_id) if task_id else None
        if not task or task.get("status") not in ("pending", "claimed"):
            task = _prune_head_locked()
        if not task:
            return None
        seed_str = str(task.get("account_seed") or "").strip()
        if not is_valid_shumei_device_id(dev) and seed_str and seed_str in _SEED_FP_MAP:
            dev = _SEED_FP_MAP[seed_str].get("device_id", "")
            if not fp_str:
                fp_str = _SEED_FP_MAP[seed_str].get("device_fingerprint", "")
        if not is_valid_shumei_device_id(dev):
            return None
        if seed_str:
            if seed_str in _SEED_FP_MAP:
                # 同一 account_seed 强制保持首次绑定的 device_id 一致，避免纯并发打码不同工位造成 dev_2 != dev_1
                dev = _SEED_FP_MAP[seed_str]["device_id"]
                if not fp_str:
                    fp_str = _SEED_FP_MAP[seed_str].get("device_fingerprint", "")
            else:
                _SEED_FP_MAP[seed_str] = {"device_id": dev, "device_fingerprint": fp_str}
        fin_ts = time.time()
        scen = str(task.get("scenario") or "register")
        wid = str(task.get("claimed_by") or "").strip()
        if wid:
            _WORKER_LAST_SOLVE[wid] = (scen, fin_ts)
        if seed_str:
            _SEED_LAST_SOLVE[seed_str] = (scen, fin_ts)
        task["token"] = tok
        task["device_id"] = dev
        task["device_fingerprint"] = fp_str
        task["status"] = "solved"
        task["event"].set()
        _prune_head_locked()
        return task["task_id"]


def fail_task(task_id: str, error: str = "solve_failed") -> Optional[str]:
    with _LOCK:
        task = _TASKS.get(task_id) if task_id else _prune_head_locked()
        if not task or task.get("status") not in ("pending", "claimed"):
            return None
        fin_ts = time.time()
        scen = str(task.get("scenario") or "register")
        wid = str(task.get("claimed_by") or "").strip()
        seed_str = str(task.get("account_seed") or "").strip()
        if wid:
            _WORKER_LAST_SOLVE[wid] = (scen, fin_ts)
        if seed_str:
            _SEED_LAST_SOLVE[seed_str] = (scen, fin_ts)
        task["error"] = str(error or "solve_failed")
        task["status"] = "failed"
        task["event"].set()
        _prune_head_locked()
        return task["task_id"]


def cancel_or_cleanup_task(task_id: str, status_if_pending: str = "timeout") -> None:
    with _LOCK:
        t = _TASKS.pop(task_id, None)
        if t and t.get("status") in ("pending", "claimed"):
            t["status"] = status_if_pending
            t["event"].set()
        _prune_head_locked()


HTML_PURE_FARM = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Blacklight VLM 独立打码农场 (Pure Captcha Farm)</title>
    <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
    <style>
        :root {
            --bg: #090d16; --card: #111726; --border: #1e293b;
            --primary: #3b82f6; --accent: #10b981; --text: #f8fafc; --sub: #94a3b8;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg); color: var(--text);
            margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 90vh;
        }
        .card {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 12px; padding: 30px; box-shadow: 0 4px 25px rgba(0,0,0,0.5);
            text-align: center; max-width: 480px; width: 100%;
        }
        h2 { margin: 0 0 10px 0; color: var(--primary); font-size: 18px; }
        p { color: var(--sub); font-size: 13px; margin: 0 0 14px 0; }
        .badge {
            display: inline-block; padding: 5px 12px; border-radius: 12px;
            font-size: 12px; font-weight: 600; margin-bottom: 16px;
        }
        .badge-idle { background: #334155; color: #cbd5e1; }
        .badge-pending { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid var(--primary); }
        .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid var(--accent); }
        .captcha-box {
            display: flex; justify-content: center; margin: 20px 0; min-height: 80px;
        }
        .meta { font-size: 11px; color: #64748b; margin-top: 8px; word-break: break-all; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🛡️ Blacklight VLM 独立打码农场工位</h2>
        <p>100% 独立题面容器，支持人工点选或 VLM 视觉全自动过码</p>
        <div id="statusBadge" class="badge badge-idle">等待流水线索题...</div>

        <div class="captcha-box">
            <div class="h-captcha"
                 data-sitekey="__SITEKEY__"
                 data-callback="onCaptchaSuccess"
                 data-theme="dark">
            </div>
        </div>
        <div id="tipText" style="font-size: 12px; color: var(--sub);">点选通过后，Token 将自动派发给流水线发信！</div>
        <div id="metaText" class="meta"></div>
    </div>

    <script>
        let currentTaskId = "";
        let currentAccountSeed = "";
        window.__FARM_ACTIVE_TASK__ = null;

        function pollTask() {
            fetch('/farm/task_status' + (window.location.search || ''))
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    const badge = document.getElementById('statusBadge');
                    const meta = document.getElementById('metaText');
                    window.__FARM_ACTIVE_TASK__ = d;
                    if (d.status === 'pending') {
                        if (d.task_id !== currentTaskId) {
                            currentTaskId = d.task_id;
                            currentAccountSeed = d.account_seed || "";
                            badge.className = 'badge badge-pending';
                            const tag = d.scenario ? ('[' + d.scenario + '] ') : '';
                            badge.innerText = '⚡ 发现待解题目 ' + tag + (d.email || '') + '，请立即点选！';
                            meta.innerText = 'Task: ' + currentTaskId + (d.queue_len > 1 ? (' | 排队: ' + d.queue_len) : '');
                            if (window.hcaptcha) {
                                try { hcaptcha.reset(); } catch(e) {}
                            }
                        }
                    } else if (d.status === 'idle') {
                        currentTaskId = "";
                        currentAccountSeed = "";
                        badge.className = 'badge badge-idle';
                        badge.innerText = '等待流水线索题...';
                        meta.innerText = '';
                    }
                })
                .catch(function() {})
                .finally(function() { setTimeout(pollTask, 800); });
        }

        function getBrowserFingerprint(seedOverride) {
            try {
                // 仅读取由 Worker 从官方 chat.deepseek.com 数美 SDK 采集并注入的真实云端备案指纹；
                // 严禁在前端本地通过 SHA/UUID 伪造虚假 'B...' 字符串（会直接触发 biz_code=11）。
                const realFp = (typeof window.__VLM_REAL_FP__ === 'string') ? window.__VLM_REAL_FP__.trim() : '';
                if (realFp && realFp.startsWith('B') && realFp.length >= 16) {
                    return Promise.resolve(realFp);
                }
            } catch(e) {}
            return Promise.resolve("");
        }
        window.getBrowserFingerprint = getBrowserFingerprint;

        function onCaptchaSuccess(token) {
            const badge = document.getElementById('statusBadge');
            const tip = document.getElementById('tipText');
            badge.className = 'badge badge-success';
            badge.innerText = '✅ 解题通过！正在同步 Token 与真实数美设备指纹...';
            tip.innerText = 'Token 与数美真实设备指纹已自动同步回流给纯 HTTP 流水线！';

            const submitTaskId = currentTaskId;
            const submitSeed = currentAccountSeed;
            const fpJson = (typeof window.__VLM_REAL_FP_JSON__ === 'string') ? window.__VLM_REAL_FP_JSON__ : '';
            getBrowserFingerprint(submitSeed).then(function(devId) {
                if (!devId) {
                    console.warn('尚无真实数美 device_id，等待 Worker 兜底携带完整数美指纹上报');
                    return;
                }
                fetch('/farm/submit_token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        token: token,
                        task_id: submitTaskId,
                        device_id: devId,
                        device_fingerprint: fpJson
                    })
                }).then(function(r) { return r.json(); }).then(function(d) {
                    console.log('Token 与真实数美指纹上报成功:', d);
                }).catch(function(err) { console.error('上报失败:', err); });
            });
        }

        pollTask();
    </script>
</body>
</html>
""".replace("__SITEKEY__", HCAPTCHA_SITEKEY)


class PureFarmHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = HTML_PURE_FARM.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/farm/task_status":
            from urllib.parse import parse_qs

            qs = parse_qs(parsed.query or "")
            wid = (qs.get("worker_id") or qs.get("worker") or [""])[0]
            tid_q = (qs.get("task_id") or [""])[0]
            cur = get_current_pending(worker_id=wid, task_id=tid_q)
            if cur:
                self._send_json(200, cur)
            else:
                self._send_json(
                    200,
                    {
                        "task_id": "",
                        "status": "idle",
                        "email": "",
                        "scenario": "",
                        "account_seed": "",
                        "queue_len": 0,
                    },
                )
        elif parsed.path == "/farm/health":
            cur = get_current_pending()
            with _LOCK:
                qlen = sum(
                    1 for qid in _QUEUE
                    if (_TASKS.get(qid) or {}).get("status") in ("pending", "claimed")
                )
            self._send_json(
                200,
                {
                    "ok": True,
                    "queue_len": qlen,
                    "active_task_id": (cur or {}).get("task_id", ""),
                    "port": PORT,
                },
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0) or 0)
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            body = {}

        if parsed.path == "/farm/claim":
            wid = str(body.get("worker_id") or "default").strip()
            warm_seeds = body.get("warm_seeds") if isinstance(body.get("warm_seeds"), list) else []
            claimed = claim_next_task(worker_id=wid, warm_seeds=warm_seeds)
            self._send_json(200, claimed)
            return

        if parsed.path == "/farm/release_seed":
            seed = str(body.get("account_seed") or body.get("seed") or "").strip()
            ok = release_seed(seed)
            self._send_json(200 if ok else 400, {"ok": ok, "account_seed": seed})
            return

        if parsed.path == "/farm/bind_seed_fp":
            seed = str(body.get("account_seed") or body.get("seed") or "").strip()
            dev = str(body.get("device_id") or "").strip()
            fp_raw = body.get("device_fingerprint") or ""
            ok = bind_seed_fingerprint(seed, dev, fp_raw)
            self._send_json(200 if ok else 400, {"ok": ok, "account_seed": seed})
            return

        if parsed.path == "/farm/solve":
            email = str(body.get("email") or "")
            scenario = str(body.get("scenario") or "register")
            account_seed = str(body.get("account_seed") or body.get("seed") or email or "")
            try:
                timeout_s = max(10.0, float(body.get("timeout") or 180.0))
            except Exception:
                timeout_s = 180.0

            task = enqueue_task(email=email, scenario=scenario, account_seed=account_seed)
            tid = task["task_id"]
            print(
                f"\n[PureFarm] 📥 接到代打任务 (TaskID: {tid} | scenario={scenario} | email={email or '-'})，等待解决..."
            )
            try:
                passed = task["event"].wait(timeout=timeout_s)
                if passed and task.get("token"):
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "task_id": tid,
                            "token": task["token"],
                            "device_id": task.get("device_id") or "",
                            "device_fingerprint": task.get("device_fingerprint") or "",
                        },
                    )
                    return
                if task.get("status") == "failed":
                    self._send_json(
                        502,
                        {"ok": False, "task_id": tid, "error": task.get("error") or "solve_failed"},
                    )
                    return
                self._send_json(408, {"ok": False, "task_id": tid, "error": "timeout"})
            finally:
                cancel_or_cleanup_task(tid)
            return

        if parsed.path == "/farm/submit_token":
            tok = str(body.get("token") or "").strip()
            tid = str(body.get("task_id") or "").strip()
            dev = str(body.get("device_id") or "").strip()
            fp_raw = body.get("device_fingerprint") or ""
            if not tok:
                self._send_json(400, {"ok": False, "error": "missing_token"})
                return
            matched = complete_task(tid, tok, dev, device_fingerprint=fp_raw)
            if matched is None:
                # 区分无有效数美指纹还是无匹配任务
                with _LOCK:
                    t_chk = _TASKS.get(tid) if tid else _prune_head_locked()
                    seed_chk = str((t_chk or {}).get("account_seed") or "").strip()
                    has_bound_fp = bool(seed_chk and seed_chk in _SEED_FP_MAP)
                if not is_valid_shumei_device_id(dev) and not has_bound_fp:
                    self._send_json(400, {"ok": False, "error": "missing_or_invalid_device_id"})
                    return
                self._send_json(409, {"ok": False, "error": "no_matching_pending_task"})
                return
            print(
                f"\n🎉 [PureFarm] 收到回调 Token! (TaskID: {matched} | len={len(tok)} | fp={(dev or 'bound')[:28]}...)"
            )
            self._send_json(200, {"ok": True, "task_id": matched})
            return

        if parsed.path == "/farm/fail":
            tid = str(body.get("task_id") or "").strip()
            err = str(body.get("error") or "worker_failed")
            matched = fail_task(tid, err)
            if matched:
                print(f"\n[PureFarm] ⚠️ 任务标记失败 (TaskID: {matched} | reason={err})")
                self._send_json(200, {"ok": True, "task_id": matched})
            else:
                self._send_json(404, {"ok": False, "error": "task_not_found"})
            return

        self._send_json(404, {"ok": False, "error": "unknown_endpoint"})


def start_pure_farm(host: str = HOST, port: int = PORT, daemon_thread: bool = False):
    server = ThreadingHTTPServer((host, port), PureFarmHandler)
    print(f"[PureFarm] 🚀 纯粹独立打码农场多线程并发服务启动: http://{host}:{port}")
    if daemon_thread:
        th = threading.Thread(target=server.serve_forever, daemon=True, name="PureCaptchaFarm")
        th.start()
        return server
    server.serve_forever()
    return server


if __name__ == "__main__":
    start_pure_farm()

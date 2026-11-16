# -*- coding: utf-8 -*-
"""
producer/vlm_farm_worker.py —— 接入 Pure Captcha Farm (127.0.0.1:9527) 的 VLM 全自动打码端
================================================================================
【核心能力】：
  1. 轮询打码农场 GET /farm/task_status，一旦纯 HTTP 产号流水线派发新任务，自动拉起或复用浏览器；
  2. 按 account_seed（邮箱/订单号）隔离与复用浏览器沙箱 Profile：
     - 同一账号的第 1 次（注册）与第 2 次（登录 MFA）复用同一实例，节省启动耗时并保持指纹一致；
     - 切换下一个账号时自动销毁旧实例、清理临时 Profile 并切换新视口与独立设备指纹，杜绝串号；
  3. 自动点击 hCaptcha 复选框：
     - 若直接秒过（has_token=True），农场前端回调秒级回传 Token；
     - 若弹出挑战弹窗，立即移交 captcha_agent / captcha_brain / captcha_shapes 视觉 Agent 自动解题；
  4. 支持单命令一体启动（--with-farm 同时启动 9527 农场服务 + VLM Worker）。

用法示例：
  # 方式 A：单终端一键同时启动 9527 农场 + VLM 自动打码端（推荐）
  python producer/vlm_farm_worker.py --with-farm --proxy http://127.0.0.1:7890

  # 方式 B：独立 Worker 连接已运行的 pure_captcha_farm.py
  python producer/vlm_farm_worker.py --farm-url http://127.0.0.1:9527 --proxy http://127.0.0.1:7890
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import random
import shutil
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from _secrets import get as _secret_get
except Exception:
    def _secret_get(k, d=""):
        return os.environ.get(str(k).upper(), d) or d

PROFILES_ROOT = pathlib.Path(HERE) / "vlm_profiles"
_FP_CACHE_LOCK = threading.Lock()
_SEED_FP_CACHE: Dict[str, tuple[str, str]] = {}
_USED_DEVICE_IDS: set[str] = set()
_PORT_ALLOC_LOCK = threading.Lock()
_RESERVED_PORTS: set[int] = set()
_BROWSER_LAUNCH_LOCK = threading.Lock()


def get_cached_seed_fp(account_seed: str) -> tuple[str, str]:
    if not account_seed:
        return "", ""
    with _FP_CACHE_LOCK:
        return _SEED_FP_CACHE.get(str(account_seed), ("", ""))


def remember_seed_fp(account_seed: str, device_id: str, fp_json: str) -> bool:
    dev = str(device_id or "").strip()
    if not dev or not dev.startswith("B") or len(dev) < 16:
        return False
    seed_str = str(account_seed or "").strip()
    with _FP_CACHE_LOCK:
        if seed_str and seed_str in _SEED_FP_CACHE:
            return True
        if dev in _USED_DEVICE_IDS:
            return False
        _USED_DEVICE_IDS.add(dev)
        if seed_str:
            _SEED_FP_CACHE[seed_str] = (dev, str(fp_json or ""))
        return True


def release_cached_seed(account_seed: str) -> None:
    if not account_seed:
        return
    with _FP_CACHE_LOCK:
        _SEED_FP_CACHE.pop(str(account_seed), None)


def _prune_old_profiles(keep_dir: Optional[pathlib.Path] = None, keep_max: int = 3) -> None:
    if not PROFILES_ROOT.exists():
        return
    dirs = [p for p in PROFILES_ROOT.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    kept = 0
    for d in dirs:
        if keep_dir and d.resolve() == keep_dir.resolve():
            continue
        if kept < keep_max:
            kept += 1
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _seed_hash(seed: str) -> str:
    return hashlib.sha256((seed or "default").encode("utf-8")).hexdigest()[:12]


def _get_farm_status(
    farm_url: str,
    timeout: float = 5.0,
    worker_id: str = "",
    task_id: str = "",
) -> Optional[Dict[str, Any]]:
    try:
        params = {}
        if worker_id:
            params["worker_id"] = worker_id
        if task_id:
            params["task_id"] = task_id
        r = requests.get(f"{farm_url}/farm/task_status", params=params or None, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _claim_farm_task(
    farm_url: str,
    worker_id: str = "slot_0",
    warm_seeds: Optional[list[str]] = None,
    timeout: float = 5.0,
) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(
            f"{farm_url}/farm/claim",
            json={"worker_id": worker_id, "warm_seeds": list(warm_seeds or [])},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            for rel_seed in data.get("released_seeds") or []:
                release_cached_seed(rel_seed)
            if data.get("status") in ("claimed", "pending") and data.get("task_id"):
                return data
            return None
        if r.status_code == 404:
            return _get_farm_status(farm_url, timeout=timeout)
    except Exception:
        pass
    return None


def _task_no_longer_pending(farm_url: str, task_id: str, worker_id: str = "") -> bool:
    st = _get_farm_status(farm_url, timeout=3.0, worker_id=worker_id, task_id=task_id)
    if not st:
        return False
    if st.get("status") != "pending":
        return True
    return bool(st.get("task_id") and st.get("task_id") != task_id)


def _checkbox_rect(page) -> Optional[Dict[str, float]]:
    js = r"""
    (() => {
      const frames = Array.from(document.querySelectorAll('iframe'));
      for (const f of frames) {
        const src = (f.src || '');
        if (!/frame=checkbox/i.test(src)) continue;
        const r = f.getBoundingClientRect();
        const st = window.getComputedStyle(f);
        if (!r || r.width < 100 || r.height < 40) continue;
        if (st && (st.visibility === 'hidden' || st.display === 'none')) continue;
        const apiReady = Boolean(window.hcaptcha && typeof window.hcaptcha.getResponse === 'function');
        return { x: r.x, y: r.y, w: r.width, h: r.height, apiReady: apiReady };
      }
      return null;
    })()
    """
    try:
        r = page.run_js(js, as_expr=True)
        if isinstance(r, dict) and r.get("w") and r.get("h"):
            out = {k: float(r[k]) for k in ("x", "y", "w", "h")}
            if "apiReady" in r:
                out["apiReady"] = bool(r["apiReady"])
            return out
    except Exception:
        pass
    return None


def _wait_checkbox_ready(page, stop_event=None, timeout: float = 15.0) -> Optional[Dict[str, float]]:
    """等待 hCaptcha 复选框 iframe 与 JS API 就绪且位置稳定，防止刚 reset 完过早点击失效。"""
    t0 = time.time()
    last_rect = None
    while time.time() - t0 < timeout:
        if stop_event and stop_event.is_set():
            return None
        r = _checkbox_rect(page)
        if r and r.get("apiReady", True):
            if last_rect is not None and abs(r["x"] - last_rect["x"]) < 2.0 and abs(r["y"] - last_rect["y"]) < 2.0:
                return r
            last_rect = r
        else:
            last_rect = None
        time.sleep(0.35)
    return _checkbox_rect(page)


def _ensure_page_task_synced(
    page,
    farm_url: str,
    task_id: str,
    account_seed: str,
    timeout: float = 8.0,
    worker_id: str = "",
) -> bool:
    """确保浏览器页签已经通过轮询拿到当前 task_id 并处于 pending 状态，否则强制同步变量并重置 widget。"""
    target_page_url = f"{farm_url}/?worker_id={worker_id}" if worker_id else farm_url
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            active = page.run_js("window.__FARM_ACTIVE_TASK__ || null", as_expr=True)
            if isinstance(active, dict) and active.get("status") == "pending" and active.get("task_id") == task_id:
                return True
        except Exception:
            pass
        time.sleep(0.35)

    # 兜底：若页面轮询略慢，等待下一次刷新并确认
    try:
        page.get(target_page_url)
    except Exception:
        return False
    t1 = time.time()
    while time.time() - t1 < 6.0:
        try:
            active = page.run_js("window.__FARM_ACTIVE_TASK__ || null", as_expr=True)
            if isinstance(active, dict) and active.get("status") == "pending" and active.get("task_id") == task_id:
                return True
        except Exception:
            pass
        time.sleep(0.35)
    return False


def _fallback_submit_if_needed(
    page,
    farm_url: str,
    task_id: str,
    account_seed: str,
    captcha_agent,
    real_device_id: str = "",
    real_fp_json: str = "",
) -> bool:
    """若 hCaptcha 已发放 token 但页面 JS 回调未及时清掉队列，则直接提取 token 与真实数美指纹上报农场。"""
    time.sleep(1.2)
    if _task_no_longer_pending(farm_url, task_id):
        return True
    st = captcha_agent.page_status(page)
    tok = (st or {}).get("token") or ""
    if len(tok) < 30:
        return False
    dev = (real_device_id or "").strip()
    fp_json = (real_fp_json or "").strip()
    if not dev:
        try:
            page.run_js(
                f"""
                window.__VLM_FP = '';
                if (window.getBrowserFingerprint) {{
                  window.getBrowserFingerprint({repr(account_seed)}).then(v => {{ window.__VLM_FP = v || ''; }});
                }}
                """,
            )
            time.sleep(0.5)
            dev = page.run_js("window.__VLM_FP || ''", as_expr=True) or ""
        except Exception:
            dev = ""
    if not fp_json:
        try:
            fp_json = page.run_js("window.__VLM_REAL_FP_JSON__ || ''", as_expr=True) or ""
        except Exception:
            fp_json = ""
    try:
        r = requests.post(
            f"{farm_url}/farm/submit_token",
            json={
                "task_id": task_id,
                "token": tok,
                "device_id": dev,
                "device_fingerprint": fp_json,
            },
            timeout=10,
        )
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


class VlmFarmWorker:
    def __init__(
        self,
        farm_url: str = "http://127.0.0.1:9527",
        proxy: str = "",
        headless: bool = False,
        max_steps: int = 12,
        max_retries: int = 2,
        fail_fast: bool = False,
        no_vlm: bool = False,
        manual_timeout: float = 300.0,
        stop_event: Optional[threading.Event] = None,
        worker_id: str = "",
    ) -> None:
        self.farm_url = farm_url.rstrip("/")
        self.proxy = (proxy or "").strip()
        self.no_vlm = bool(no_vlm)
        # 人工打码模式下必须可见窗口，强制关闭 headless
        self.headless = False if self.no_vlm else bool(headless)
        self.manual_timeout = max(30.0, float(manual_timeout))
        self.stop_event = stop_event
        self.max_steps = max(4, int(max_steps))
        self.max_retries = max(1, int(max_retries))
        self.fail_fast = bool(fail_fast)
        self.worker_id = str(worker_id or "").strip()
        self.page = None
        self.current_port: Optional[int] = None
        self.current_seed_hash: Optional[str] = None
        self.current_profile_dir: Optional[pathlib.Path] = None
        self.current_device_id: str = ""
        self.current_device_fp_json: str = ""

    @property
    def worker_page_url(self) -> str:
        return f"{self.farm_url}/?worker_id={self.worker_id}" if self.worker_id else self.farm_url

    def close_browser(self) -> None:
        if self.page is not None:
            try:
                self.page.quit()
            except Exception:
                pass
            self.page = None
        if self.current_port is not None:
            with _PORT_ALLOC_LOCK:
                _RESERVED_PORTS.discard(self.current_port)
            self.current_port = None
        self.current_seed_hash = None
        self.current_device_id = ""
        self.current_device_fp_json = ""

    def _capture_real_shumei_fp(self, page, timeout: Optional[float] = None) -> tuple[str, str]:
        """在沙箱浏览器中打开 chat.deepseek.com/sign_in，等待官方数美 SDK 在云端备案并写入 .thumbcache_*。"""
        import json

        if timeout is None:
            timeout = 60.0 if self.no_vlm else 20.0
        max_loads = 1 if self.no_vlm else 2

        print(f"[VLMWorker] 🛡️ 正在加载官方数美 SDK 采集真实设备指纹 (chat.deepseek.com/sign_in, 最长等待 {int(timeout)}s)...")
        js_extract = r"""
        (() => {
            const ls = {};
            try {
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (!k) continue;
                    if (k.indexOf('.thumbcache_') === 0 || k === 'smidV2'
                        || k === 'deepseek-device-id:chat'
                        || k === '__ds_remote_feature_did'
                        || k.indexOf('__tea_cache_') === 0) {
                        ls[k] = localStorage.getItem(k) || '';
                    }
                }
            } catch(e) {}
            const ck = {};
            try {
                document.cookie.split(';').forEach(function(item) {
                    const idx = item.indexOf('=');
                    if (idx > 0) {
                        const name = item.slice(0, idx).trim();
                        const val = item.slice(idx + 1).trim();
                        if (name) {
                            try { ck[name] = decodeURIComponent(val); } catch(e) { ck[name] = val; }
                        }
                    }
                });
            } catch(e) {}
            let devId = '';
            for (const k of Object.keys(ls)) {
                if (k.indexOf('.thumbcache_') === 0 && ls[k] && ls[k].length >= 16) {
                    devId = 'B' + ls[k];
                    break;
                }
            }
            const hasWaf = !!(document.querySelector('script[src*="awswaf"], script[src*="challenge.js"], #challenge-container'));
            return { device_id: devId, localStorage: ls, cookies: ck, title: document.title || '', hasWaf: hasWaf };
        })()
        """
        for load_try in range(max_loads):
            try:
                page.get("https://chat.deepseek.com/sign_in")
            except Exception as e:
                print(f"[VLMWorker] [!] 访问 chat.deepseek.com/sign_in 异常 ({load_try + 1}/{max_loads}): {e}")
                continue
            t0 = time.time()
            waf_warned = False
            while time.time() - t0 < timeout:
                if self.stop_event and self.stop_event.is_set():
                    return "", ""
                try:
                    info = page.run_js(js_extract, as_expr=True)
                except Exception:
                    info = None
                if isinstance(info, dict):
                    dev_id = str(info.get("device_id") or "").strip()
                    if len(dev_id) >= 16 and dev_id.startswith("B") and not any(ch.isspace() for ch in dev_id):
                        fp_obj = {
                            "device_id": dev_id,
                            "localStorage": info.get("localStorage") or {},
                            "cookies": info.get("cookies") or {},
                            "source": "shumei_sdk",
                            "captured_at": int(time.time()),
                        }
                        fp_json = json.dumps(fp_obj, ensure_ascii=False)
                        print(f"[VLMWorker] ✅ 成功取得云端备案真实数美指纹: {dev_id[:28]}... (耗时 {time.time() - t0:.1f}s)")
                        return dev_id, fp_json
                    title_low = str(info.get("title") or "").lower()
                    is_cf_or_waf = bool(info.get("hasWaf")) or (
                        "error: the request could not be satisfied" in title_low
                        or "human verification" in title_low
                    )
                    if is_cf_or_waf:
                        if not waf_warned:
                            waf_warned = True
                            if self.no_vlm:
                                print("\a", end="", flush=True)
                                print(
                                    f"[ManualWorker] ⏳ 检测到 WAF / 验证页 (title={info.get('title')!r})，"
                                    f"绝不刷新打断！请等待自动放行或在浏览器中完成验证（最多等 {int(timeout)}s）..."
                                )
                            else:
                                print("[VLMWorker] ⏳ 检测到 WAF/CloudFront 挑战页，等待放行...")
                        # 仅非人工模式下，若非 JS WAF 且持续超过 8s 仍为静态拦截才放弃本轮
                        if (not self.no_vlm) and (not info.get("hasWaf")) and (time.time() - t0 > 8.0):
                            print("[VLMWorker] [!] chat.deepseek.com/sign_in 被 CloudFront 持续拦截，检查代理直连分流配置")
                            break
                time.sleep(0.5)
        print("[VLMWorker] [-] 未能从官方数美 SDK 采集到 .thumbcache_* 指纹")
        return "", ""

    def _clear_deepseek_origin_storage(self, page) -> None:
        """在现有浏览器槽位内瞬时清空 chat.deepseek.com 存储与 Cookie，实现仅在采集新数美指纹那 1s 做隔离。"""
        try:
            page.run_cdp(
                "Storage.clearDataForOrigin",
                origin="https://chat.deepseek.com",
                storageTypes="all",
            )
        except Exception:
            pass
        try:
            page.run_cdp("Network.clearBrowserCookies")
        except Exception:
            pass

    def _allocate_free_port(self) -> int:
        import captcha_agent

        with _PORT_ALLOC_LOCK:
            for cand in range(19520, 19680):
                if cand in _RESERVED_PORTS:
                    continue
                p = captcha_agent._pick_free_port(start=cand, end=cand + 1)
                if p == cand and p not in _RESERVED_PORTS:
                    _RESERVED_PORTS.add(p)
                    return p
            p = captcha_agent._pick_free_port(start=19520, end=19680)
            _RESERVED_PORTS.add(p)
            return p

    def _ensure_browser(self, account_seed: str, force_isolate: bool = False):
        from DrissionPage import ChromiumOptions, ChromiumPage

        sh = _seed_hash(account_seed)
        cached_dev, cached_fp = get_cached_seed_fp(account_seed)
        target_url = self.worker_page_url

        # 1) 若当前槽位浏览器已存活：
        #    - 若该 account_seed 已有缓存数美指纹（如 tok_2 或预绑定），直接保留一致性指纹；
        #      若发生跨阶段切换 (force_isolate=True)，仍强制清理域存储与 Cookie 做阶段隔离。
        #    - 若是新 account_seed，仅瞬时清空 chat.deepseek.com 存储采集新指纹，无需关停重启 chrome.exe！
        if self.page is not None:
            try:
                if cached_dev:
                    if force_isolate:
                        self._clear_deepseek_origin_storage(self.page)
                    self.current_device_id = cached_dev
                    self.current_device_fp_json = cached_fp
                elif force_isolate or self.current_seed_hash != sh or not self.current_device_id:
                    self._clear_deepseek_origin_storage(self.page)
                    dev_id, fp_json = self._capture_real_shumei_fp(self.page)
                    if dev_id and remember_seed_fp(account_seed, dev_id, fp_json):
                        self.current_device_id = dev_id
                        self.current_device_fp_json = fp_json
                    else:
                        # 极罕见：若 CDP 清理后仍命中旧指纹，则降级重建全新 Profile
                        self.close_browser()
                if self.page is not None:
                    url_now = self.page.url or ""
                    if self.farm_url not in url_now or (self.worker_id and f"worker_id={self.worker_id}" not in url_now):
                        self.page.get(target_url)
                    self.current_seed_hash = sh
                    return self.page
            except Exception:
                self.close_browser()

        self.close_browser()
        PROFILES_ROOT.mkdir(parents=True, exist_ok=True)
        slot_tag = f"{self.worker_id}_{sh}" if self.worker_id else f"seed_{sh}"
        prof_dir = PROFILES_ROOT / slot_tag
        if prof_dir.exists():
            shutil.rmtree(prof_dir, ignore_errors=True)
        prof_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_profiles(keep_dir=prof_dir, keep_max=6)

        free_port = self._allocate_free_port()
        self.current_port = free_port
        co = ChromiumOptions()
        co.set_local_port(free_port)
        co.set_user_data_path(str(prof_dir))
        co.set_argument("--no-first-run")
        co.set_argument("--no-default-browser-check")
        co.set_argument("--disable-infobars")
        co.set_argument("--disable-blink-features=AutomationControlled")
        # 防后台遮挡/多窗口并发渲染冻结参数
        co.set_argument("--disable-backgrounding-occluded-windows")
        co.set_argument("--disable-renderer-backgrounding")
        co.set_argument("--disable-background-timer-throttling")
        co.set_argument("--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling")
        if sys.platform.startswith("linux"):
            co.set_argument("--no-sandbox")
            co.set_argument("--disable-dev-shm-usage")
        h_int = int(sh, 16)
        win_w = 1180 + (h_int % 140)
        win_h = 820 + ((h_int >> 8) % 90)
        co.set_argument(f"--window-size={win_w},{win_h}")
        if self.proxy:
            co.set_proxy(self.proxy)
            co.set_argument(
                "--proxy-bypass-list=127.0.0.1;localhost;chat.deepseek.com;*.deepseek.com;*.fengkongcloud.cn;*.fengkongcloud.com"
            )
        if self.headless:
            co.headless()

        print(
            f"[VLMWorker{(':' + self.worker_id) if self.worker_id else ''}] 🌐 启动隔离浏览器 "
            f"(seed={sh}, port={free_port}, win={win_w}x{win_h}, headless={self.headless})"
        )
        with _BROWSER_LAUNCH_LOCK:
            page = ChromiumPage(co)
            time.sleep(0.5)
        if cached_dev:
            dev_id, fp_json = cached_dev, cached_fp
        else:
            dev_id, fp_json = self._capture_real_shumei_fp(page)
            if dev_id:
                remember_seed_fp(account_seed, dev_id, fp_json)
        self.current_device_id = dev_id
        self.current_device_fp_json = fp_json
        page.get(target_url)
        self.page = page
        self.current_seed_hash = sh
        self.current_profile_dir = prof_dir
        return page

    def solve_one(self, task: Dict[str, Any]) -> bool:
        import json
        import captcha_agent

        task_id = str(task.get("task_id") or "")
        email = str(task.get("email") or "")
        scenario = str(task.get("scenario") or "register")
        account_seed = str(task.get("account_seed") or email or task_id)
        bound_dev = str(task.get("device_id") or "").strip()
        bound_fp = str(task.get("device_fingerprint") or "").strip()
        if bound_dev:
            remember_seed_fp(account_seed, bound_dev, bound_fp)

        wtag = f"[VLMWorker:{self.worker_id}]" if self.worker_id else "[VLMWorker]"
        print(f"\n{wtag} 🎯 接单: task={task_id} | scenario={scenario} | email={email or '-'}")
        page = (
            self._ensure_browser(account_seed, force_isolate=True)
            if task.get("stage_switch")
            else self._ensure_browser(account_seed)
        )

        if not self.current_device_id:
            print(f"{wtag} [-] 未获取到官方数美真实指纹，中止该打码任务")
            if self.fail_fast:
                try:
                    requests.post(
                        f"{self.farm_url}/farm/fail",
                        json={"task_id": task_id, "error": "shumei_fp_missing"},
                        timeout=5,
                    )
                except Exception:
                    pass
            return False

        if not _ensure_page_task_synced(
            page,
            self.farm_url,
            task_id,
            account_seed,
            worker_id=self.worker_id,
        ):
            print(f"{wtag} [-] 浏览器未能同步到任务状态")
            if self.fail_fast:
                try:
                    requests.post(
                        f"{self.farm_url}/farm/fail",
                        json={"task_id": task_id, "error": "farm_page_sync_failed"},
                        timeout=5,
                    )
                except Exception:
                    pass
            return False

        try:
            page.run_js(
                f"window.__VLM_REAL_FP__ = {json.dumps(self.current_device_id)};"
                f"window.__VLM_REAL_FP_JSON__ = {json.dumps(self.current_device_fp_json)};"
            )
        except Exception as e:
            print(f"[VLMWorker] [!] 注入数美指纹变量异常: {e}")

        if self.no_vlm:
            return self._solve_one_manual(
                page=page,
                task_id=task_id,
                email=email,
                scenario=scenario,
                account_seed=account_seed,
                captcha_agent=captcha_agent,
            )

        for attempt in range(1, self.max_retries + 1):
            if _task_no_longer_pending(self.farm_url, task_id, worker_id=self.worker_id):
                return True

            print(f"{wtag} 🔄 第 {attempt}/{self.max_retries} 轮尝试...")
            if attempt > 1:
                try:
                    page.run_js("if (window.hcaptcha) { hcaptcha.reset(); }")
                    time.sleep(1.5)
                except Exception:
                    page.get(self.worker_page_url)
                    _ensure_page_task_synced(
                        page,
                        self.farm_url,
                        task_id,
                        account_seed,
                        worker_id=self.worker_id,
                    )

            # 等待 checkbox iframe 与 hCaptcha JS API 就绪且位置稳定，防止 reset 后过早点击失效
            cbox = _wait_checkbox_ready(page, stop_event=self.stop_event, timeout=25.0)
            if not cbox:
                print("[VLMWorker] [-] 未检测到 hCaptcha 复选框 iframe（检查代理/hCaptcha 连通性）")
                continue

            # 拟人停留等 cross-origin anchor/hsw.js 完全挂好点击监听，并确认面板尚未自动展开
            time.sleep(random.uniform(1.6, 2.5))
            panel_appeared = bool(captcha_agent.panel_rect(page))
            instant_ok = bool(captcha_agent.page_status(page).get("has_token"))
            if not instant_ok and not panel_appeared:
                for click_try in range(2):
                    cbox_now = _checkbox_rect(page) or cbox
                    click_x = cbox_now["x"] + random.uniform(22.0, 34.0)
                    click_y = cbox_now["y"] + cbox_now["h"] / 2.0 + random.uniform(-4.0, 4.0)
                    print(
                        f"[VLMWorker] 🖱️ 点击 hCaptcha 复选框 ({click_x:.1f}, {click_y:.1f})"
                        + (" [补点]" if click_try > 0 else "")
                    )
                    captcha_agent.human_click(page, click_x, click_y, jitter=2.0)

                    # 代理下 HSW PoW 与 challenge iframe 展开常需 6~15s，耐心观察，不打断加载动画
                    t_obs = time.time()
                    obs_limit = 16.0 if click_try == 0 else 14.0
                    while time.time() - t_obs < obs_limit:
                        if self.stop_event and self.stop_event.is_set():
                            return False
                        if _task_no_longer_pending(self.farm_url, task_id, worker_id=self.worker_id):
                            return True
                        st = captcha_agent.page_status(page)
                        if st.get("has_token"):
                            instant_ok = True
                            break
                        ch_active, _ = captcha_agent.challenge_active(page)
                        if ch_active or captcha_agent.panel_rect(page):
                            panel_appeared = True
                            break
                        time.sleep(0.4)
                    if instant_ok or panel_appeared:
                        break
                    if click_try == 0:
                        time.sleep(random.uniform(0.8, 1.5))

            if instant_ok:
                print("[VLMWorker] ⚡ 复选框直接通过，无需二次答题！")
                if _fallback_submit_if_needed(
                    page,
                    self.farm_url,
                    task_id,
                    account_seed,
                    captcha_agent,
                    real_device_id=self.current_device_id,
                    real_fp_json=self.current_device_fp_json,
                ):
                    return True
                continue

            if not panel_appeared:
                st_wp = captcha_agent.wait_panel(page, timeout=12.0, verbose=True)
                if st_wp.get("has_token") and _fallback_submit_if_needed(
                    page,
                    self.farm_url,
                    task_id,
                    account_seed,
                    captcha_agent,
                    real_device_id=self.current_device_id,
                    real_fp_json=self.current_device_fp_json,
                ):
                    return True
                if not st_wp.get("panel"):
                    print("[VLMWorker] [-] 复选框点击后挑战面板仍未展开，跳过 VLM 盲点并重置 hCaptcha 准备下一轮")
                    try:
                        page.run_js("if (window.hcaptcha) { hcaptcha.reset(); }")
                    except Exception:
                        pass
                    time.sleep(1.5)
                    continue
            else:
                # 确认题图渲染就绪（避免刚打开瞬间的白板期就发 VLM）
                st_wp = captcha_agent.wait_panel(page, timeout=12.0, verbose=True)
                if st_wp.get("has_token") and _fallback_submit_if_needed(
                    page,
                    self.farm_url,
                    task_id,
                    account_seed,
                    captcha_agent,
                    real_device_id=self.current_device_id,
                    real_fp_json=self.current_device_fp_json,
                ):
                    return True

            res = captcha_agent.solve_captcha(
                page,
                max_steps=max(self.max_steps, 14),
                hint="当前页面仅渲染了标准 hCaptcha，目标是让 window.hcaptcha.getResponse() 拿到 token。",
                verbose=True,
                should_stop=lambda: _task_no_longer_pending(self.farm_url, task_id, worker_id=self.worker_id),
                deadline=time.time() + 220.0,
                task_meta={
                    "task_id": task_id,
                    "email": str(task.get("email") or ""),
                    "scenario": str(task.get("scenario") or ""),
                    "worker_id": str(self.worker_id or ""),
                },
            )
            ok = bool((res or {}).get("ok")) or bool(captcha_agent.page_status(page).get("has_token"))
            if ok:
                if _fallback_submit_if_needed(
                    page,
                    self.farm_url,
                    task_id,
                    account_seed,
                    captcha_agent,
                    real_device_id=self.current_device_id,
                    real_fp_json=self.current_device_fp_json,
                ):
                    print(f"[VLMWorker] ✅ 任务 {task_id} 已成功通过并回传 Token！")
                    return True
            print(f"[VLMWorker] ⚠️ 本轮未通过: {(res or {}).get('reason', 'unknown')}")

        if self.fail_fast:
            try:
                requests.post(
                    f"{self.farm_url}/farm/fail",
                    json={"task_id": task_id, "error": "vlm_exhausted_retries"},
                    timeout=5,
                )
            except Exception:
                pass
        return False

    def _solve_one_manual(
        self,
        page,
        task_id: str,
        email: str,
        scenario: str,
        account_seed: str,
        captcha_agent,
    ) -> bool:
        """无 VLM 纯人工点选模式：自动展开复选框（秒过直接提交），弹出题面后等待用户手动过码并回传真实数美指纹。"""
        if _task_no_longer_pending(self.farm_url, task_id):
            return True

        cbox = _wait_checkbox_ready(page, stop_event=self.stop_event, timeout=15.0)

        instant_ok = False
        panel_appeared = bool(captcha_agent.panel_rect(page))
        if cbox and not panel_appeared:
            time.sleep(random.uniform(1.0, 1.5))
            st0 = captcha_agent.page_status(page)
            if st0.get("has_token"):
                instant_ok = True
            elif not captcha_agent.panel_rect(page):
                click_x = cbox["x"] + random.uniform(22.0, 34.0)
                click_y = cbox["y"] + cbox["h"] / 2.0 + random.uniform(-4.0, 4.0)
                print(f"[ManualWorker] 🖱️ 自动展开 hCaptcha 复选框 ({click_x:.1f}, {click_y:.1f})...")
                try:
                    captcha_agent.human_click(page, click_x, click_y, jitter=2.5)
                except Exception as e:
                    print(f"[ManualWorker] [!] 自动点击复选框异常（请直接在浏览器内手动点击）: {e}")
                t_obs = time.time()
                while time.time() - t_obs < 4.5:
                    if self.stop_event and self.stop_event.is_set():
                        return False
                    if _task_no_longer_pending(self.farm_url, task_id):
                        return True
                    if captcha_agent.page_status(page).get("has_token"):
                        instant_ok = True
                        break
                    if captcha_agent.panel_rect(page):
                        panel_appeared = True
                        break
                    time.sleep(0.35)

        if instant_ok:
            print("[ManualWorker] ⚡ 复选框直接通过，无需手动点选！")
            if _fallback_submit_if_needed(
                page,
                self.farm_url,
                task_id,
                account_seed,
                captcha_agent,
                real_device_id=self.current_device_id,
                real_fp_json=self.current_device_fp_json,
            ):
                return True

        print("\a", end="", flush=True)
        print("=" * 72)
        print(f"[ManualWorker] 👤 请在弹出的 Chrome 窗口内手动完成 hCaptcha 验证！")
        print(f"  • 场景: {scenario} | 邮箱: {email or '-'} | TaskID: {task_id}")
        print(f"  • 指纹: {self.current_device_id[:28]}... (已绑定真实数美 SDK 指纹)")
        print(f"  • 提示: 点选通过后会自动回传 Token 并继续纯流量流水线，最长等待 {int(self.manual_timeout)}s")
        print("=" * 72)

        deadline = time.time() + self.manual_timeout
        while time.time() < deadline:
            if self.stop_event and self.stop_event.is_set():
                return False
            if _task_no_longer_pending(self.farm_url, task_id):
                print(f"[ManualWorker] ✅ 任务 {task_id} 已收到人工过码回传 Token！")
                return True
            try:
                st = captcha_agent.page_status(page)
            except Exception as e:
                print(f"[ManualWorker] [-] 浏览器窗口已关闭或连接断开: {e}")
                self.close_browser()
                return False
            if (st or {}).get("has_token"):
                if _fallback_submit_if_needed(
                    page,
                    self.farm_url,
                    task_id,
                    account_seed,
                    captcha_agent,
                    real_device_id=self.current_device_id,
                    real_fp_json=self.current_device_fp_json,
                ):
                    print(f"[ManualWorker] ✅ 任务 {task_id} 已人工通过并回传 Token！")
                    return True
            time.sleep(0.4)

        print(f"[ManualWorker] [-] 等待人工过码超时 ({int(self.manual_timeout)}s)")
        if self.fail_fast:
            try:
                requests.post(
                    f"{self.farm_url}/farm/fail",
                    json={"task_id": task_id, "error": "manual_timeout"},
                    timeout=5,
                )
            except Exception:
                pass
        return False

    def run_forever(self) -> None:
        import captcha_agent

        if self.no_vlm:
            print("[ManualWorker] 👤 无 VLM 纯人工点选模式已就绪（只采集真实数美指纹 + 弹出 hCaptcha 窗口，不调用 VLM）")
        elif not getattr(captcha_agent, "VLM_FALLBACKS", None):
            print("[VLMWorker] ⚠️ 警告: 未检测到可用 vlm_api_key！请在 producer/secrets.json 或环境变量 VLM_API_KEY 配置。")
        else:
            eps = ", ".join(f"{e['model']}@{e['base']}" for e in captcha_agent.VLM_FALLBACKS)
            print(f"[VLMWorker] 🧠 VLM 通道就绪: {eps}")

        base_tag = "ManualWorker" if self.no_vlm else "VLMWorker"
        tag = f"{base_tag}:{self.worker_id}" if self.worker_id else base_tag
        print(f"[{tag}] 👀 开始监听打码农场: {self.farm_url}")
        seen_failed: set[str] = set()
        try:
            while not (self.stop_event and self.stop_event.is_set()):
                if self.worker_id:
                    with _FP_CACHE_LOCK:
                        warm_seeds = list(_SEED_FP_CACHE.keys())
                    st = _claim_farm_task(self.farm_url, worker_id=self.worker_id, warm_seeds=warm_seeds, timeout=4.0)
                else:
                    st = _get_farm_status(self.farm_url, timeout=4.0)
                if st and st.get("status") in ("pending", "claimed"):
                    tid = str(st.get("task_id") or "")
                    if tid and tid not in seen_failed:
                        try:
                            ok = self.solve_one(st)
                        except Exception as e:
                            print(f"[{tag}] [!] solve_one 异常 (task={tid}): {e}")
                            self.close_browser()
                            ok = False
                            if self.fail_fast:
                                try:
                                    requests.post(
                                        f"{self.farm_url}/farm/fail",
                                        json={"task_id": tid, "error": f"worker_exception:{type(e).__name__}"},
                                        timeout=5,
                                    )
                                except Exception:
                                    pass
                        if not ok:
                            seen_failed.add(tid)
                            if len(seen_failed) > 100:
                                seen_failed.clear()
                elif st and st.get("status") == "cooldown":
                    wait_s = min(max(0.25, float(st.get("retry_after") or 0.5)), 2.0)
                    time.sleep(wait_s)
                    continue
                time.sleep(0.5 if self.no_vlm else 0.6)
        finally:
            self.close_browser()


class MultiSlotFarmRunner:
    """多槽位并发打码调度器：启动 C 个独立打码槽位错峰并行等题图，共享 VLM 信号量卡时间差排队推理。"""

    def __init__(
        self,
        workers: int = 2,
        farm_url: str = "http://127.0.0.1:9527",
        proxy: str = "",
        headless: bool = False,
        max_steps: int = 12,
        max_retries: int = 2,
        fail_fast: bool = False,
        no_vlm: bool = False,
        manual_timeout: float = 300.0,
        stagger_secs: float = 2.0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.num_workers = max(1, int(workers or 1))
        self.stop_event = stop_event or threading.Event()
        self.stagger_secs = max(0.0, float(stagger_secs))
        self.workers: list[VlmFarmWorker] = [
            VlmFarmWorker(
                farm_url=farm_url,
                proxy=proxy,
                headless=headless,
                max_steps=max_steps,
                max_retries=max_retries,
                fail_fast=fail_fast,
                no_vlm=no_vlm,
                manual_timeout=manual_timeout,
                stop_event=self.stop_event,
                worker_id=(f"slot_{i}" if self.num_workers > 1 else ""),
            )
            for i in range(self.num_workers)
        ]

    def close_all(self) -> None:
        self.stop_event.set()
        for w in self.workers:
            try:
                w.close_browser()
            except Exception:
                pass

    def run_forever(self) -> None:
        if self.num_workers == 1:
            self.workers[0].run_forever()
            return
        print(
            f"[VLMWorkerPool] 🚀 启动 {self.num_workers} 个并发打码槽位 "
            f"(错峰间隔={self.stagger_secs:.1f}s, POST /farm/claim 原子抢单)..."
        )
        threads: list[threading.Thread] = []
        try:
            for idx, w in enumerate(self.workers):
                if idx > 0 and self.stagger_secs > 0 and not self.stop_event.is_set():
                    time.sleep(self.stagger_secs)
                t = threading.Thread(target=w.run_forever, daemon=True, name=f"VlmSlot-{idx}")
                t.start()
                threads.append(t)
            while not self.stop_event.is_set():
                if not any(t.is_alive() for t in threads):
                    break
                time.sleep(0.5)
        finally:
            self.close_all()


def main() -> None:
    default_proxy = _secret_get("proxy_url", "http://127.0.0.1:7890")
    ap = argparse.ArgumentParser(description="Blacklight VLM Browser Captcha Farm Worker")
    ap.add_argument("--farm-url", default="http://127.0.0.1:9527", help="打码农场地址 (默认 http://127.0.0.1:9527)")
    ap.add_argument("--with-farm", action="store_true", help="在同一进程内同时启动 9527 打码农场 HTTP 服务")
    ap.add_argument("--proxy", default=default_proxy, help="浏览器代理地址 (例如 http://127.0.0.1:7890，传 none 禁用)")
    ap.add_argument("--headless", action="store_true", help="以无头模式启动 Chromium")
    ap.add_argument("--workers", type=int, default=1, help="并发打码浏览器槽位数 C (默认 1，推荐 2~4)")
    ap.add_argument("--vlm-concurrency", type=int, default=1, help="VLM 同时推理并发上限 (默认 1，卡时间差排队防 429)")
    ap.add_argument("--max-steps", type=int, default=12, help="每轮挑战 VLM 最大决策步数")
    ap.add_argument("--max-retries", type=int, default=2, help="单次任务失败最大重试轮数")
    ap.add_argument("--fail-fast", action="store_true", help="重试耗尽后立即通知农场 fail，不等待人工接管超时")
    ap.add_argument("--allow-fallback", action="store_true", help="允许 VLM 失败时回落到 ZEN_API_KEY")
    ap.add_argument("--no-vlm", "--manual", dest="no_vlm", action="store_true", help="无 VLM 纯人工点选模式：弹出 Chrome 让用户手动点选 hCaptcha，保留真实数美指纹采集")
    ap.add_argument("--manual-timeout", type=float, default=300.0, help="人工模式单次验证码最大等待秒数 (默认 300)")
    args = ap.parse_args()

    if not args.allow_fallback:
        os.environ.setdefault("VLM_ONLY", "gateway")
    import captcha_agent

    captcha_agent.set_vlm_concurrency(max(1, int(args.vlm_concurrency or 1)))
    proxy_val = "" if (args.proxy or "").strip().lower() == "none" else args.proxy
    headless_val = bool(args.headless)
    if args.no_vlm and headless_val:
        print("[ManualWorker] [!] --no-vlm 人工模式必须显示浏览器窗口，已自动忽略 --headless")
        headless_val = False

    if args.with_farm:
        from urllib.parse import urlparse
        import pure_captcha_farm

        parsed_farm = urlparse(args.farm_url)
        farm_host = parsed_farm.hostname or "127.0.0.1"
        farm_port = parsed_farm.port or 9527
        pure_captcha_farm.start_pure_farm(host=farm_host, port=farm_port, daemon_thread=True)
        time.sleep(0.3)

    runner = MultiSlotFarmRunner(
        workers=max(1, int(args.workers or 1)),
        farm_url=args.farm_url,
        proxy=proxy_val,
        headless=headless_val,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        fail_fast=args.fail_fast,
        no_vlm=bool(args.no_vlm),
        manual_timeout=args.manual_timeout,
    )

    def _handle_sig(_sig, _frame):
        print("\n[VLMWorker] 正在优雅退出并关闭浏览器...")
        runner.close_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)
    runner.run_forever()


if __name__ == "__main__":
    main()

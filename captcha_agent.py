# -*- coding: utf-8 -*-
"""
captcha_agent.py — 万能视觉验证码 Agent（v1.0）
================================================================================
设计哲学
--------------------------------------------------------------------------------
验证码的**物理动作只有两种：点击(click) 和 拖拽(drag)**。所有题型差异
（九宫格选图、滑块、拼图、文字题、复选框）都只是"给 VLM 看什么、点哪里"的
语义差别，不需要为每种题型写 DOM 解析器。

因此本模块彻底抛弃 DOM 几何：
    全视口截图  →  VLM 识题型 + 输出归一化坐标  →  CDP 派发真实鼠标事件
                    ↑____________________循环观察____________________|

为什么能绕开历史所有坑
--------------------------------------------------------------------------------
· iframe 是跨域 OOPIF 时，`frame.run_js()` 会静默回落到主页面上下文，
  `getBoundingClientRect()` 读到的是主文档的视口尺寸，几何恒错。
  本模块**完全不读 iframe DOM**，因此该坑不存在。
· hCaptcha 的 challenge iframe 未激活时位于 y=-9999、visibility:hidden，
  任何"先取几何再截图"的实现都会失败。本模块只截"眼睛看得到的视口"。
· 坐标用**归一化 0~1**，与 devicePixelRatio / 截图缩放完全解耦：
      css_x = norm_x * viewport_width
  实测 deepseek-flash 在 420x420 九宫格上平均误差 0.2px（9/9 命中）。

对外接口
--------------------------------------------------------------------------------
    from captcha_agent import solve_captcha
    result = solve_captcha(page, max_steps=12)      # 传 DrissionPage 的 page
    result["ok"]       -> bool   是否通过
    result["steps"]    -> list   每步的决策与动作（可审计）

HTTP 接口（供打号主程序或任何外部程序调用）
--------------------------------------------------------------------------------
    python captcha_agent.py --serve --listen 127.0.0.1:8848
    POST /solve   {"port": 9333, "max_steps": 12, "url": "可选"}
    POST /probe   {"port": 9333}          # 只观察不动作，返回当前状态
    GET  /health
================================================================================
"""
import argparse
import base64
from contextlib import contextmanager
import io
import json
import os
import random
import sys
import threading
import time

import requests
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from _secrets import get as _secret_get
except Exception:
    def _secret_get(k, d=""):
        return os.environ.get(str(k).upper(), d) or d

# 证书：本机有这个自定义 CA 才用；换机器（没这个文件）时回落到 certifi，
# 否则 SSL_CERT_FILE 指向不存在的路径会让所有 HTTPS 请求直接报错。
_CA_FILE = os.environ.get("SSL_CERT_FILE") or r"C:\Temp\cacert.pem"
if os.path.isfile(_CA_FILE):
    os.environ.setdefault("SSL_CERT_FILE", _CA_FILE)
    os.environ.setdefault("CURL_CA_BUNDLE", _CA_FILE)
else:
    # 指向不存在的证书反而会让所有 HTTPS 失败，直接清掉回落到 certifi
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("CURL_CA_BUNDLE", None)

def _norm_completions_url(raw_url: str, default_url: str) -> str:
    u = (raw_url or default_url or "").strip().rstrip("/")
    if not u:
        return default_url
    if not u.endswith("/chat/completions"):
        if u.endswith("/v1"):
            u = u + "/chat/completions"
        else:
            u = u + "/v1/chat/completions"
    return u

# ------------------------------------------------------------------ 配置
VLM_BASE = _norm_completions_url(
    _secret_get("vlm_base_url", "http://127.0.0.1:8000/v1/chat/completions"),
    "http://127.0.0.1:8000/v1/chat/completions",
)
VLM_KEY = _secret_get("vlm_api_key", "")
VLM_MODEL = _secret_get("vlm_model", "deepseek-v4.1-flash")
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "120"))

# ------------------------------------------------------------------ 视觉模型通道
# 主通道：OpenAI 兼容多模态接口（可对接本地 ds-gateway 或任意 VLM 服务）
# 兜底  ：可选 OpenCode Zen / Go 套餐（仅当配置了 ZEN_API_KEY 时启用）
ZEN_BASE = _secret_get("zen_base_url", "https://opencode.ai/zen/go/v1")
ZEN_KEY = _secret_get("zen_api_key", "")
ZEN_MODEL = _secret_get("zen_model", "gemini-4-argon")
# zen 要求带会话头，否则 400 MissingSessionID
ZEN_HEADERS = {"x-opencode-session": "ses_captcha"}

# 主通道顺序可用环境变量调：VLM_ORDER=gateway,zen / zen,gateway
_ORDER = [x.strip().lower() for x in
          os.environ.get("VLM_ORDER", "gateway,zen").split(",") if x.strip()]

_GATEWAY = {"base": VLM_BASE, "key": VLM_KEY, "model": VLM_MODEL, "headers": {}}
_ZEN = [
    {"base": ZEN_BASE.rstrip("/") + "/chat/completions", "key": ZEN_KEY,
     "model": ZEN_MODEL, "headers": ZEN_HEADERS},
    {"base": ZEN_BASE.rstrip("/") + "/chat/completions", "key": ZEN_KEY,
     "model": "claude-haiku-5-5", "headers": ZEN_HEADERS},
]
VLM_FALLBACKS = []
for _k in _ORDER:
    if _k == "gateway" and _GATEWAY["key"]:
        VLM_FALLBACKS.append(dict(_GATEWAY))
    elif _k == "zen" and ZEN_KEY:
        VLM_FALLBACKS.extend(dict(e) for e in _ZEN)
# 若某通道配置了密钥且尚未加入，保留为兜底
if _GATEWAY["key"] and not any(
    e["base"] == _GATEWAY["base"] and e["model"] == _GATEWAY["model"]
    for e in VLM_FALLBACKS
):
    VLM_FALLBACKS.append(dict(_GATEWAY))
if ZEN_KEY and not any(e["model"] == ZEN_MODEL for e in VLM_FALLBACKS):
    VLM_FALLBACKS.append(dict(_ZEN[0]))

# 子调用链：默认与主脑完全一致（整体走用户自己的网关）。
if os.environ.get("VLM_SUB_SPLIT", "").strip().lower() in ("1", "true", "yes"):
    VLM_SUB = ([e for e in VLM_FALLBACKS if "opencode.ai" in e["base"]]
               + [e for e in VLM_FALLBACKS if "opencode.ai" not in e["base"]])
else:
    VLM_SUB = list(VLM_FALLBACKS)

# VLM_ONLY=gateway → 彻底不要 Zen 兜底，只用用户自己的网关。
if os.environ.get("VLM_ONLY", "").strip().lower() == "gateway":
    VLM_FALLBACKS = [dict(_GATEWAY)] if _GATEWAY["key"] else []
    VLM_SUB = list(VLM_FALLBACKS)

DEBUG_DIR = os.environ.get("CAPTCHA_DEBUG_DIR", os.path.join(HERE, "captcha_debug"))
try:
    os.makedirs(DEBUG_DIR, exist_ok=True)
except Exception:
    DEBUG_DIR = HERE

_cdp_lock = threading.Lock()
_PAGE_CDP_LOCKS: dict[int, threading.Lock] = {}
_PAGE_CDP_LOCKS_GUARD = threading.Lock()

# VLM 信号量与错峰排队（卡时间差排队打：多个槽位并行等题图加载/跑鼠标动画，仅在调用 VLM 那 2~4s 排队）
VLM_CONCURRENCY = max(1, int(os.environ.get("VLM_CONCURRENCY", "1") or "1"))
VLM_STAGGER_SECONDS = max(0.0, float(os.environ.get("VLM_STAGGER_SECONDS", "0.25") or "0.25"))
_VLM_SEM = threading.BoundedSemaphore(VLM_CONCURRENCY)
_VLM_PACER_LOCK = threading.Lock()
_LAST_VLM_ENTER_TS = 0.0


def set_vlm_concurrency(limit: int) -> None:
    global VLM_CONCURRENCY, _VLM_SEM
    VLM_CONCURRENCY = max(1, int(limit or 1))
    _VLM_SEM = threading.BoundedSemaphore(VLM_CONCURRENCY)


@contextmanager
def vlm_slot():
    """获取一个 VLM 并发槽位并执行错峰间隔控制。"""
    global _LAST_VLM_ENTER_TS
    with _VLM_SEM:
        if VLM_STAGGER_SECONDS > 0:
            with _VLM_PACER_LOCK:
                now = time.time()
                gap = (_LAST_VLM_ENTER_TS + VLM_STAGGER_SECONDS) - now
                if gap > 0:
                    time.sleep(gap)
                _LAST_VLM_ENTER_TS = time.time()
        yield


def _get_cdp_lock(page) -> threading.Lock:
    if page is None:
        return _cdp_lock
    pid = id(page)
    with _PAGE_CDP_LOCKS_GUARD:
        lk = _PAGE_CDP_LOCKS.get(pid)
        if lk is None:
            if len(_PAGE_CDP_LOCKS) > 64:
                _PAGE_CDP_LOCKS.clear()
            lk = threading.Lock()
            _PAGE_CDP_LOCKS[pid] = lk
        return lk

# 截图最长边：过小看不清格子，过大浪费 token。归一化坐标与缩放无关。
MAX_SHOT_SIDE = int(os.environ.get("CAPTCHA_MAX_SHOT", "1400"))

# ------------------------------------------------------------------ Prompt
SYSTEM_PROMPT = """你是自动化验证码求解器。输入是浏览器**当前视口的完整截图**。

你必须同时完成「状态判定」和「动作规划」两件事。

【状态 state】六选一：
- "no_captcha" : 页面完全没有验证码
- "checkbox"   : 只有未勾选的"我不是机器人"复选框（hCaptcha/reCAPTCHA 外层）
- "challenge"  : 已出现题目（图片九宫格 / 滑块 / 拼图 / 文字题 / 图形题）
- "loading"    : 验证码区域是空白/骨架屏，题目正在加载
- "solved"     : 已通过（绿色对勾、success 提示、或验证码消失且页面已进入下一步）
- "blocked"    : 提示"请稍后再试/被拒绝/机器人检测失败"，无法继续

【动作 actions】按执行顺序排列，每项只能是下面之一：
- {"type":"click","x":0.5,"y":0.5}
- {"type":"drag","x1":0.2,"y1":0.5,"x2":0.7,"y2":0.5}
- {"type":"wait"}
- {"type":"done"}

【坐标规则 —— 最重要】
1. x、y 是**归一化坐标**，范围 0.0 ~ 1.0。
2. (0,0) = 截图左上角，(1,1) = 截图右下角。
3. 必须给目标元素的**几何中心**，精度到小数点后 3 位。
4. 不要输出像素、不要输出百分比、不要估算偏移。

【题型处理规范】
· 复选框：点**左侧小方块**的中心，不要点右边的说明文字。
· 图片九宫格：题目形如"请点击所有包含 X 的图片"。逐格判断，
  命中格点其**格子中心**；不命中的格子绝对不要点。
· 滑块 / 拼图：drag 从滑块中心拖到缺口中心（两者通常同一水平线）。
· 文字题 / 图形题：按提示点对应的字符或图形中心。
· 若同时存在外层复选框与内层题目，**只处理当前最上层可见的那个**。

【底部按钮状态机 —— 决定成败，必须逐字读按钮文案】
题目面板底部右侧的按钮只有三种文案，含义完全不同：
· 「跳过 / Skip」      → 本题尚未选中任何目标。**绝对不要点它**（点 = 放弃本题）。
· 「下一步 / Next」    → 本页已选好，点它进入下一小题。
· 「检查 / 验证 / Verify / 提交」 → 最后一页，点它提交答案。
动作策略（严格按此执行）：
- 按钮文案是「跳过」  → 本轮 actions 只包含格子/目标的点击。
- 按钮文案是「下一步」或「检查/验证」 → 本轮 actions 只包含**这一个按钮的 click**，
  不要再重复点格子（重复点会把已选中的格子取消掉）。
- 题目可能分多页（顶部有一排小圆点，实心=当前页）：每页先选目标 → 点「下一步」，
  最后一页点「检查」。页数没走完就点「检查」通常会被判错。

【跨页记忆 —— 移动动物类题目必备】
· 题目「点击移动动物从未到访过的岛屿」中：动物当前站着的岛 = 已经到访。
· 分多页时动物会在不同页出现在不同岛；**前面页里出现过动物的岛也不要点**。
· 因此正确集合 = 所有岛 −（本页动物所在岛 ∪ 前几页动物出现过的岛）。

【已选状态识别 —— 防止把选中点掉】
· 被选中的格子会出现：白色勾选角标 / 半透明高亮遮罩 / 边框变色。
· 某格已被选中且确实命中题目 → **跳过它，绝对不要再点**（再点会取消选择）。
· 某格已被选中但不应命中 → 再点一次取消。

【语义题规范（新版 hCaptcha 常见）】
· 「点击移动动物未曾到访的岛屿」：动物所在的岛=已到访，**其余岛=未到访**，点其余。
· 「点击移动动物曾到访的岛屿」：只点动物所在 / 有脚印痕迹的岛。
· 决策顺序（务必按此执行，别凭直觉）：
    ① 先读题目原文，确定要"有痕迹"还是"没痕迹"；
    ② 再定位动物在哪一格（它的位置就是"已到访"）；
    ③ 然后只点符合条件的目标，逐个确认坐标确实落在目标图形内部。

【复位识别】
· 如果你刚答过题、现在又回到「我是真人访客」复选框，说明上一轮答案被判错。
  这一轮必须重新逐格核对题目要求，不要复用上一次的答案。

【输出格式】严格输出单个 JSON，禁止 Markdown 代码块，禁止任何解释文字：
{"state":"challenge","captcha_type":"hcaptcha_grid","instruction":"题目原文","actions":[{"type":"click","x":0.123,"y":0.456}],"confidence":"high","note":"一句话理由"}

captcha_type 取值：none / hcaptcha_checkbox / hcaptcha_grid / recaptcha_checkbox /
recaptcha_grid / slider / puzzle / text / other"""


# ================================================================== 底层：截图
def _djpeg(img_bytes, quality=82):
    try:
        im = Image.open(io.BytesIO(img_bytes))
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > MAX_SHOT_SIDE:
            scale = MAX_SHOT_SIDE / float(longest)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), im.size
    except Exception:
        return img_bytes, None


def _raw_screenshot(page):
    """取视口截图原始字节。三条通路依次降级。"""
    lk = _get_cdp_lock(page)
    # 1) DrissionPage 原生
    try:
        with lk:
            b = page.get_screenshot(as_bytes=True)
        if isinstance(b, (bytes, bytearray)) and len(b) > 512:
            return bytes(b)
    except Exception:
        pass
    # 2) CDP Page.captureScreenshot
    try:
        with lk:
            r = page.run_cdp("Page.captureScreenshot", format="jpeg", quality=88)
        data = r.get("data") if isinstance(r, dict) else None
        if data:
            raw = base64.b64decode(data)
            if len(raw) > 512:
                return raw
    except Exception:
        pass
    # 3) 落盘再读
    try:
        with lk:
            p = page.get_screenshot()
        if isinstance(p, str) and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    except Exception:
        pass
    return None


def viewport(page):
    """视口 CSS 尺寸与 DPR。注意 run_js 必须 as_expr=True。"""
    js = ("(() => ({w: window.innerWidth, h: window.innerHeight, "
          "dpr: window.devicePixelRatio || 1}))()")
    try:
        with _get_cdp_lock(page):
            r = page.run_js(js, as_expr=True)
        if isinstance(r, dict) and r.get("w"):
            return {"w": float(r["w"]), "h": float(r["h"]),
                    "dpr": float(r.get("dpr") or 1.0)}
    except Exception:
        pass
    return {"w": 1280.0, "h": 800.0, "dpr": 1.0}


# 只做「存在性」判定，不读任何几何 —— 与"几何恒错"的历史坑彻底解耦。
_HAS_CAPTCHA_JS = r"""
(() => {
  const hits = [];
  const scan = (root) => {
    let frames = [];
    try { frames = Array.from(root.querySelectorAll('iframe')); } catch (e) { return; }
    for (const f of frames) {
      const src = (f.src || '').toLowerCase();
      if (/hcaptcha|recaptcha|turnstile|challenges\.cloudflare|geetest|nc_1_n1z|slide/.test(src)) {
        hits.push(src.slice(0, 90));
      }
    }
  };
  scan(document);
  for (const el of Array.from(document.querySelectorAll('*'))) {
    if (el.shadowRoot) scan(el.shadowRoot);
  }
  const marks = ['.h-captcha', '#cf-chl-widget', '[data-hcaptcha-widget-id]',
                 '.g-recaptcha', '#nc_1_wrapper', '.geetest_panel', '.captcha'];
  for (const sel of marks) {
    try { if (document.querySelector(sel)) hits.push('dom:' + sel); } catch (e) {}
  }
  return { present: hits.length > 0, hits: hits.slice(0, 6) };
})()
"""

# 挑战面板是否已从"屏外占位"变成"真正可见"。
# 注意：这里只取一个 bool（computed style + rect 高度），不做任何坐标推算，
# 因此不会踩"iframe 几何读到主页面视口"的坑。
_CHALLENGE_ACTIVE_JS = r"""
(() => {
  const out = { active: false, n: 0, detail: [] };
  const scan = (root) => {
    let frames = [];
    try { frames = Array.from(root.querySelectorAll('iframe')); } catch (e) { return; }
    for (const f of frames) {
      const src = (f.src || '');
      if (!/frame=challenge/i.test(src)) continue;
      out.n += 1;
      let st = null, r = null;
      try { st = window.getComputedStyle(f); } catch (e) {}
      try { r = f.getBoundingClientRect(); } catch (e) {}
      const vis = !!st && st.visibility !== 'hidden' && st.display !== 'none';
      const tall = !!r && r.height > 40 && r.width > 80;
      const onscreen = !!r && r.top > -500 && r.top < 4000;
      out.detail.push({ vis: vis, tall: tall, onscreen: onscreen });
      if (vis && tall && onscreen) out.active = true;
    }
  };
  scan(document);
  for (const el of Array.from(document.querySelectorAll('*'))) {
    if (el.shadowRoot) scan(el.shadowRoot);
  }
  return out;
})()
"""


def challenge_active(page):
    """挑战面板是否已真正可见（bool 判定）。"""
    try:
        with _get_cdp_lock(page):
            r = page.run_js(_CHALLENGE_ACTIVE_JS, as_expr=True)
        if isinstance(r, dict):
            return bool(r.get("active")), r
    except Exception:
        pass
    return False, {}


def wait_challenge_active(page, timeout=8.0):
    """点完复选框后，等待挑战面板真正展开。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok, _detail = challenge_active(page)
        if ok:
            return True
        time.sleep(0.3)
    return False


def has_captcha(page):
    """页面上是否存在验证码控件（存在性判定，零几何依赖）。"""
    try:
        with _get_cdp_lock(page):
            r = page.run_js(_HAS_CAPTCHA_JS, as_expr=True)
        if isinstance(r, dict):
            return bool(r.get("present")), r.get("hits") or []
    except Exception:
        pass
    return False, []


def wait_for_captcha(page, timeout=25.0, verbose=True):
    """等待验证码控件注入（hCaptcha 的 js/iframe 加载常需 5~10s）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok, hits = has_captcha(page)
        if ok:
            if verbose:
                print("  [+] 检测到验证码控件（%.1fs）: %s"
                      % (time.time() - t0, hits[:2]))
            return True
        time.sleep(0.7)
    if verbose:
        print("  [!] %.0fs 内未检测到验证码控件" % timeout)
    return False


# ================================================================== 底层：动作
_LAST_CURSOR: dict = {}


def _mouse(page, mtype, x, y, button="none", buttons=0, clicks=0):
    args = {"type": mtype, "x": float(x), "y": float(y),
            "button": button, "buttons": int(buttons)}
    if clicks:
        args["clickCount"] = int(clicks)
    with _get_cdp_lock(page):
        return page.run_cdp("Input.dispatchMouseEvent", **args)


def human_click(page, x, y, jitter=2.0):
    """拟人化点击：从上一光标位置沿缓变贝塞尔曲线平滑移入目标后按下抬起。"""
    try:
        jx = float(x) + random.uniform(-jitter, jitter)
        jy = float(y) + random.uniform(-jitter, jitter)
        pid = id(page)
        prev = _LAST_CURSOR.get(pid)
        if prev is None:
            ang = random.uniform(0, 6.28318)
            dist0 = random.uniform(140.0, 320.0)
            import math
            sx = max(24.0, min(1180.0, jx + math.cos(ang) * dist0))
            sy = max(24.0, min(740.0, jy + math.sin(ang) * dist0))
        else:
            sx, sy = prev

        dist = ((jx - sx) ** 2 + (jy - sy) ** 2) ** 0.5
        if dist >= 18.0:
            # 轻微弯曲的二次贝塞尔轨迹（避免直线瞬移触发风控）
            mid_x = (sx + jx) * 0.5 + random.uniform(-0.18, 0.18) * dist
            mid_y = (sy + jy) * 0.5 + random.uniform(-0.18, 0.18) * dist
            steps = max(8, min(20, int(dist / 26.0)))
            for i in range(1, steps):
                t = i / float(steps)
                # smoothstep ease-in-out
                u = t * t * (3.0 - 2.0 * t)
                omt = 1.0 - u
                px = omt * omt * sx + 2.0 * omt * u * mid_x + u * u * jx + random.uniform(-1.0, 1.0)
                py = omt * omt * sy + 2.0 * omt * u * mid_y + u * u * jy + random.uniform(-1.0, 1.0)
                _mouse(page, "mouseMoved", px, py)
                time.sleep(random.uniform(0.008, 0.022))

        _mouse(page, "mouseMoved", jx, jy)
        _LAST_CURSOR[pid] = (jx, jy)
        if len(_LAST_CURSOR) > 32:
            _LAST_CURSOR.clear()
            _LAST_CURSOR[pid] = (jx, jy)
        time.sleep(random.uniform(0.07, 0.17))
        _mouse(page, "mousePressed", jx, jy, "left", 1, 1)
        time.sleep(random.uniform(0.06, 0.14))
        _mouse(page, "mouseReleased", jx, jy, "left", 0, 1)
        return True
    except Exception as e:
        print("  [!] click 派发失败 (%.0f,%.0f): %s" % (x, y, e))
        return False


def human_drag(page, x1, y1, x2, y2, steps=None, jitter=1.5):
    """拟人化拖拽：按下 → 多段变速移动（含轻微抖动）→ 抬起。"""
    try:
        if steps is None:
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            steps = max(12, min(38, int(dist / 9)))
        _mouse(page, "mouseMoved", x1 + random.uniform(-3, 3), y1 + random.uniform(-3, 3))
        time.sleep(random.uniform(0.05, 0.12))
        _mouse(page, "mousePressed", x1, y1, "left", 1, 1)
        time.sleep(random.uniform(0.08, 0.18))
        for i in range(1, steps + 1):
            t = i / float(steps)
            # ease-out：起步快，末段慢，贴近真人把滑块对齐缺口的手感
            e = 1.0 - (1.0 - t) ** 2.2
            cx = x1 + (x2 - x1) * e + random.uniform(-jitter, jitter)
            cy = y1 + (y2 - y1) * e + random.uniform(-jitter, jitter)
            _mouse(page, "mouseMoved", cx, cy, "left", 1)
            time.sleep(random.uniform(0.006, 0.020))
        _mouse(page, "mouseMoved", x2, y2, "left", 1)
        time.sleep(random.uniform(0.06, 0.14))
        _mouse(page, "mouseReleased", x2, y2, "left", 0, 1)
        return True
    except Exception as e:
        print("  [!] drag 派发失败: %s" % e)
        return False


# ================================================================== VLM 调用
def _parse_json(content):
    if not content:
        return None
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    i = content.find("{")
    while i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(content[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = content.find("{", i + 1)
    return None


def call_vlm_endpoints(
    messages,
    *,
    chain=None,
    effort=None,
    timeout=None,
    deadline=None,
    ask_budget=None,
    max_attempts=3,
    extra_parsers=None,
    include_raw=False,
    return_on_parse_fail=False,
):
    """通过共享 _VLM_SEM 信号量与错峰控制依次调用 VLM 端点并解析 JSON 决策。"""
    last = "缺少 vlm_api_key：请在 producer/secrets.json 或环境变量 VLM_API_KEY 配置"
    endpoints = [ep for ep in (chain if chain is not None else VLM_FALLBACKS) if ep.get("key")]
    if not endpoints:
        meta = {"error": last}
        if deadline is not None or ask_budget is not None:
            meta["elapsed"] = 0.0
        return None, meta

    t_overall = time.time()
    req_timeout = float(timeout if timeout is not None else VLM_TIMEOUT)
    with vlm_slot():
        eff_deadline = deadline
        if ask_budget is not None:
            budget_deadline = time.time() + float(ask_budget)
            eff_deadline = min(eff_deadline, budget_deadline) if eff_deadline is not None else budget_deadline

        for ep in endpoints:
            if eff_deadline is not None:
                rem = eff_deadline - time.time()
                if rem < 8:
                    last = "总耗时超预算（剩 %.0fs）" % max(0.0, rem)
                    break
                call_timeout = max(8.0, min(req_timeout, rem))
            else:
                call_timeout = req_timeout

            base, key, model = ep["base"], ep["key"], ep["model"]
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
            }
            # reasoning_effort 是 DeepSeek 专有字段，其他厂商收到可能直接 400
            if model.startswith("deepseek"):
                payload["reasoning_effort"] = effort or os.environ.get("VLM_EFFORT", "off")

            t0 = time.time()
            r = None
            for attempt in range(max_attempts):
                if eff_deadline is not None and attempt > 0 and (eff_deadline - time.time()) < 8:
                    break
                try:
                    r = requests.post(
                        base,
                        headers=dict({"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"},
                                     **(ep.get("headers") or {})),
                        json=payload, timeout=call_timeout)
                except Exception as e:
                    last = "%s: %s" % (type(e).__name__, str(e)[:70])
                    r = None
                    if attempt < max_attempts - 1:
                        wait_s = 1.0 * (attempt + 1)
                        if eff_deadline is not None:
                            wait_s = min(wait_s, max(0.0, eff_deadline - time.time() - 8.0))
                        if wait_s > 0:
                            time.sleep(wait_s)
                    continue
                if r.status_code in (429, 500, 502, 503, 504):
                    last = "HTTP %d" % r.status_code
                    r = None
                    if attempt < max_attempts - 1:
                        backoff = (1.5 ** attempt) + random.uniform(0.15, 0.55)
                        if eff_deadline is not None:
                            backoff = min(backoff, max(0.0, eff_deadline - time.time() - 8.0))
                        if backoff > 0:
                            time.sleep(backoff)
                    continue
                break

            if r is None:
                print("  [-] VLM %s 不可用（%s），切换备用" % (model, last))
                continue
            if r.status_code != 200:
                last = "HTTP %d %s" % (r.status_code, (r.text or "")[:120])
                print("  [-] VLM %s %s" % (model, last))
                continue

            try:
                content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                last = "解析失败 %s" % e
                continue

            plan = _parse_json(content)
            if plan is None and extra_parsers:
                for parser in extra_parsers:
                    plan = parser(content)
                    if plan is not None:
                        break
            elapsed = round(time.time() - (t_overall if eff_deadline is not None else t0), 1)
            meta = {"model": model, "elapsed": elapsed}
            if include_raw:
                meta["raw"] = content
            if not plan:
                last = "JSON 解析失败" if return_on_parse_fail else "无有效 JSON"
                print("  [-] VLM 无有效 JSON: %s" % (content or "")[:160].replace("\n", " "))
                if return_on_parse_fail:
                    meta["error"] = last
                    return None, meta
                continue
            return plan, meta

    meta = {"error": last}
    if deadline is not None or ask_budget is not None:
        meta["elapsed"] = round(time.time() - t_overall, 1)
    return None, meta


def ask_vlm(jpeg_bytes, hint="", extra="", raw_prompt=False, force=False,
            effort=None, chain=None):
    """把截图交给 VLM，返回 (plan_dict, meta)。失败返回 (None, meta)。

    raw_prompt=True 时 hint 直接作为完整 prompt（不再拼接 SYSTEM_PROMPT），
    用于 locate / sheet 这类“只做单一子任务”的子调用。
    通过 _VLM_SEM 信号量与错峰间隔实现“多槽位并行等题图、VLM 卡时间差排队推理”。
    """
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    if raw_prompt:
        prompt = hint or ""
    else:
        prompt = SYSTEM_PROMPT
        if hint:
            prompt += "\n\n【补充信息】%s" % hint
    if extra:
        prompt += "\n\n【上一轮反馈】%s" % extra

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + b64}},
        ],
    }]
    return call_vlm_endpoints(messages, chain=chain, effort=effort)


# ================================================================== 坐标换算
def to_css(plan_action, vp):
    """归一化坐标 → CSS 像素。"""
    def cx(v):
        return float(v) * vp["w"]

    def cy(v):
        return float(v) * vp["h"]

    t = (plan_action.get("type") or "").lower()
    if t == "click":
        return ("click", cx(plan_action.get("x", 0)), cy(plan_action.get("y", 0)))
    if t == "drag":
        return ("drag", cx(plan_action.get("x1", 0)), cy(plan_action.get("y1", 0)),
                cx(plan_action.get("x2", 0)), cy(plan_action.get("y2", 0)))
    return (t or "wait",)


# ================================================================== 面板裁剪
# 关键：跨域 iframe 的**内容**读不到，但 iframe 元素本身的 getBoundingClientRect
# 是主文档的东西，完全可读。这给了我们一个可靠的"题目面板矩形"，
# 既能裁剪放大给 VLM 看，也能把面板内坐标精确映射回视口坐标。
_PANEL_RECT_JS = r"""
(() => {
  let best = null;
  const scan = (root) => {
    let frames = [];
    try { frames = Array.from(root.querySelectorAll('iframe')); } catch (e) { return; }
    for (const f of frames) {
      const src = (f.src || '');
      if (!/frame=challenge/i.test(src)) continue;
      let r = null, st = null;
      try { r = f.getBoundingClientRect(); } catch (e) {}
      try { st = window.getComputedStyle(f); } catch (e) {}
      if (!r) continue;
      if (!st || st.visibility === 'hidden' || st.display === 'none') continue;
      if (r.width < 200 || r.height < 200) continue;
      if (r.top < -500 || r.left < -500) continue;
      if (!best || r.width * r.height > best.w * best.h) {
        best = {x: Math.round(r.x), y: Math.round(r.y),
                w: Math.round(r.width), h: Math.round(r.height)};
      }
    }
  };
  scan(document);
  for (const el of Array.from(document.querySelectorAll('*'))) {
    if (el.shadowRoot) scan(el.shadowRoot);
  }
  return best;
})()
"""


def panel_rect(page):
    """可见的 challenge 面板矩形（CSS 像素），不可见时返回 None。"""
    try:
        with _get_cdp_lock(page):
            r = page.run_js(_PANEL_RECT_JS, as_expr=True)
        if isinstance(r, dict) and r.get("w") and r.get("h"):
            return {"x": float(r["x"]), "y": float(r["y"]),
                    "w": float(r["w"]), "h": float(r["h"])}
    except Exception:
        pass
    return None


_TOKEN_JS = r"""
(() => {
  const out = {api: false, token: '', countdown: false, panel: false,
               btn: '', err: ''};
  try {
    if (window.hcaptcha && window.hcaptcha.getResponse) {
      out.api = true;
      out.token = String(window.hcaptcha.getResponse() || '');
    }
  } catch (e) { out.err = String(e); }
  try {
    const el = document.querySelector('.ds-verify-code-input-countdown');
    out.countdown = !!(el && el.offsetParent !== null);
    out.countdown_text = el ? String(el.innerText || '').trim().slice(0, 24) : '';
    const frames = Array.from(document.querySelectorAll('iframe'));
    out.panel = frames.some(f => {
      if (!/frame=challenge/i.test(f.src || '')) return false;
      const r = f.getBoundingClientRect();
      const st = getComputedStyle(f);
      return r.width > 200 && r.height > 200 && r.top > -500
             && st.visibility !== 'hidden';
    });
  } catch (e) {}
  try {
    document.querySelectorAll('button,[role=button],div>span').forEach(e => {
      const s = (e.innerText || '').trim();
      if (['跳过', '下一个', '下一步', '检查', '验证', 'Skip', 'Next', 'Verify'].includes(s)) {
        out.btn = s;
      }
    });
  } catch (e) {}
  return out;
})()
"""


def page_status(page):
    """一次性读回判据：hCaptcha token / 面板是否可见 / 发码倒计时 / 底部按钮文案。"""
    try:
        with _get_cdp_lock(page):
            r = page.run_js(_TOKEN_JS, as_expr=True)
        if isinstance(r, dict):
            r["has_token"] = len(r.get("token") or "") >= 30
            # 真·放行判据：倒计时可见且带秒数（说明验证码已过、码已发出）
            import re as _re
            r["passed"] = bool(r["has_token"] or (
                r.get("countdown") and _re.search(r"(\d+\s*[秒s]|resend|重新获取)",
                                                  r.get("countdown_text") or "")))
            return r
    except Exception:
        pass
    return {"api": False, "token": "", "countdown": False, "panel": False,
            "btn": "", "err": "", "has_token": False, "passed": False,
            "countdown_text": ""}


def panel_ready(img, min_std=14.0):
    """面板里的题图是否真的渲染出来了。

    刚展开时 hCaptcha 有个 0.5~2s 的动画，期间题图区是纯白，此时 locate 必然
    得到 0 个目标。用题图区的颜色标准差做判据（纯白/纯灰 → 标准差极低）。
    """
    try:
        W, H = img.size
        box = (int(W * 0.16), int(H * 0.22), int(W * 0.84), int(H * 0.82))
        c = img.crop(box).convert("RGB").resize((72, 72), Image.BILINEAR)
        px = list(c.getdata())
        n = float(len(px))
        mean = [sum(p[i] for p in px) / n for i in range(3)]
        var = sum(((p[0] - mean[0]) ** 2 + (p[1] - mean[1]) ** 2
                   + (p[2] - mean[2]) ** 2) / 3.0 for p in px) / n
        std = var ** 0.5
        return std >= min_std, round(std, 1)
    except Exception:
        return True, -1


def find_button(panel, min_frac=0.02):
    """在面板底部找到「跳过/下一个/检查」按钮的真实矩形（归一化坐标）。

    为什么必须检测而不是写死：
      hCaptcha 面板布局随语言/题面变化，按钮在底部条里的位置不是固定比例。
      实测把 y 写死成 0.95 时，点击落在按钮下沿外 2~3 像素 —— 看起来点到了，
      其实什么都没发生，最后只能等 120 秒超时，白丢一次机会。
    按钮是底部条里唯一的**较大块纯色区域**（白色背景上的深色圆角矩形），
    用「底部 15% 内非白像素的最大连通块」定位，稳且与语言无关。
    """
    try:
        import numpy as np
        im = panel.convert("RGB")
        W, H = im.size
        y0 = int(H * 0.84)
        arr = np.asarray(im.crop((0, y0, W, H)), dtype=np.int16)
        # 非白：任一通道明显偏暗（按钮深色 / 文字深色），白底三个通道都接近 255
        mask = (arr.min(axis=2) < 205).astype(np.uint8)
        if mask.sum() < 0.002 * mask.size:
            return None
        # 简单两遍扫描做连通域（不引入 scipy）
        h, w = mask.shape
        lab = np.zeros((h, w), dtype=np.int32)
        cur = 0
        best = None
        from collections import deque
        for i in range(h):
            for j in range(w):
                if mask[i, j] and not lab[i, j]:
                    cur += 1
                    q = deque([(i, j)])
                    lab[i, j] = cur
                    x1 = x2 = j
                    yy1 = yy2 = i
                    n = 0
                    while q:
                        a, b = q.popleft()
                        n += 1
                        if b < x1: x1 = b
                        if b > x2: x2 = b
                        if a < yy1: yy1 = a
                        if a > yy2: yy2 = a
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not lab[na, nb]:
                                lab[na, nb] = cur
                                q.append((na, nb))
                    ww, hh = x2 - x1 + 1, yy2 - yy1 + 1
                    if n < min_frac * (w * h):
                        continue
                    # 按钮特征：宽 > 高，宽度占面板 8% 以上
                    if ww < 0.08 * W or hh < 0.015 * H:
                        continue
                    if best is None or n > best[0]:
                        best = (n, x1, yy1 + y0, ww, hh)
        if not best:
            return None
        _n, x1, y1, ww, hh = best
        return {"x": round((x1 + ww / 2.0) / W, 4),
                "y": round((y1 + hh / 2.0) / H, 4),
                "w": round(ww / float(W), 4), "h": round(hh / float(H), 4)}
    except Exception as e:
        print("  [!] 按钮检测失败: %s" % e)
        return None


def wait_panel(page, timeout=25.0, verbose=True):
    """等挑战面板**真正可见**（不只是 iframe 存在 —— 隐藏时它在 y=-9999）。

    hCaptcha 在点完「发送验证码」后先挂一个隐藏 iframe，再异步决定要不要出题，
    实测从点击到面板可见常有 8~15s 延迟。太早开始 Agent 会误判"没有验证码"。
    面板可见后还要再等题图渲染完成，否则 VLM 看到的是一张白板。
    """
    t0 = time.time()
    st = None
    while time.time() - t0 < timeout:
        st = page_status(page)
        if st.get("panel") or st.get("has_token"):
            break
        time.sleep(0.4)
    else:
        if verbose:
            print("  [!] %.0fs 内面板未出现（可能被静默放行或还没触发）" % timeout)
        return st or page_status(page)

    if st.get("has_token"):
        if verbose:
            print("  [+] 已有 token（无需答题）")
        return st

    # 面板可见 → 等题图渲染
    t1 = time.time()
    while time.time() - t1 < 15:
        raw = _raw_screenshot(page)
        rect = panel_rect(page)
        if raw and rect:
            vp = viewport(page)
            c, _ = crop_panel(raw, rect, vp)
            if c is not None:
                ok, std = panel_ready(c)
                if ok:
                    if verbose:
                        print("  [+] 挑战面板已就绪（%.1fs，题图标准差 %.1f）"
                              % (time.time() - t0, std))
                    return page_status(page)
        time.sleep(0.5)
    if verbose:
        print("  [!] 面板可见但题图 15s 内未渲染完成，先照常开始")
    return page_status(page)


def crop_panel(img_bytes, rect, vp, long_side=1150):
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None, None
    W, H = im.size
    sx, sy = W / float(vp["w"]), H / float(vp["h"])
    box = (max(0, int(rect["x"] * sx)), max(0, int(rect["y"] * sy)),
           min(W, int((rect["x"] + rect["w"]) * sx)),
           min(H, int((rect["y"] + rect["h"]) * sy)))
    if box[2] - box[0] < 120 or box[3] - box[1] < 120:
        return None, None
    c = im.crop(box)
    ln = max(c.size)
    if ln < long_side:
        k = long_side / float(ln)
        c = c.resize((int(c.size[0] * k), int(c.size[1] * k)), Image.LANCZOS)
    return c, {"box": box, "rect": rect}


# ================================================================== 「同形配对」题型
_PAIR_TASK_RE = None


def is_pair_task(text):
    """题目是否属于"点击两个相同的图形"这一类。"""
    t = (text or "").strip()
    if not t:
        return False
    global _PAIR_TASK_RE
    if _PAIR_TASK_RE is None:
        import re
        _PAIR_TASK_RE = re.compile(
            r"(两个|2\s*个).{0,6}(相同|一样|同一)|相同(的)?(图形|形状|图案|物品)"
            r"|同形|identical|same\s+shape|matching\s+shape", re.I)
    return bool(_PAIR_TASK_RE.search(t))


def _dump(img_bytes, tag):
    try:
        uniq = "%s_%03d_t%d" % (
            time.strftime("%Y%m%d_%H%M%S"),
            int((time.time() % 1) * 1000),
            threading.get_ident() % 10000,
        )
        p = os.path.join(DEBUG_DIR, "agent_%s_%s.jpg" % (tag, uniq))
        with open(p, "wb") as f:
            f.write(img_bytes)
        return p
    except Exception:
        return None


def _annotate(img_bytes, points, tag):
    """把动作点画到截图上，便于人工复核 AI 判断是否准确。"""
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for kind, px, py in points:
            r = 13
            d.ellipse([px * W - r, py * H - r, px * W + r, py * H + r],
                      outline=(0, 220, 0), width=4)
            d.line([px * W - 20, py * H, px * W + 20, py * H], fill=(0, 220, 0), width=2)
            d.line([px * W, py * H - 20, px * W, py * H + 20], fill=(0, 220, 0), width=2)
        uniq = "%s_%03d_t%d" % (
            time.strftime("%Y%m%d_%H%M%S"),
            int((time.time() % 1) * 1000),
            threading.get_ident() % 10000,
        )
        p = os.path.join(DEBUG_DIR, "annot_%s_%s.jpg" % (tag, uniq))
        im.save(p, quality=85)
        return p
    except Exception:
        return None


# ================================================================== 主循环
class VisionCaptchaAgent(object):
    """视觉验证码 Agent：观察 → 决策 → 动作 → 再观察。"""

    def __init__(self, page, max_steps=12, verbose=True,
                 settle=(0.9, 1.8), settle_after_click=(1.0, 2.0),
                 wait_captcha=True, wait_timeout=25.0):
        self.page = page
        self.max_steps = int(max_steps)
        self.verbose = verbose
        self.settle = settle
        self.settle_after_click = settle_after_click
        self.wait_captcha = wait_captcha
        self.wait_timeout = float(wait_timeout)
        self.history = []
        self._pair_sig = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def observe(self, tag="step", want_panel=False):
        raw = _raw_screenshot(self.page)
        if not raw:
            return None
        jpeg, size = _djpeg(raw)
        vp = viewport(self.page)
        obs = {"jpeg": jpeg, "img_size": size, "vp": vp, "raw": raw,
               "panel": None, "rect": None}
        if want_panel:
            rect = panel_rect(self.page)
            if rect:
                c, info = crop_panel(raw, rect, vp)
                if c is not None:
                    obs["panel"] = c
                    obs["rect"] = rect
        return obs

    def _try_pair_solver(self, obs, plan, result, step):
        """两段式同形配对。成功点击后返回 True（调用方跳到下一步）。"""
        if time.time() < getattr(self, "_pair_skip_until", 0):
            return False
        try:
            import captcha_shapes as SH
        except Exception as e:
            self._log("        [!] 同形求解器不可用: %s" % e)
            return False

        rect = panel_rect(self.page)
        if not rect:
            self._log("        [!] 没有可见的挑战面板，无法裁剪")
            return False
        panel, _info = crop_panel(obs["raw"], rect, obs["vp"])
        if panel is None:
            return False

        res = SH.solve_pair_from_image(panel, ask_vlm, verbose=self.verbose)
        result["steps"].append({"step": step, "state": "challenge",
                                "pair_solver": res.get("reason"),
                                "pair": res.get("pair")})
        if not res.get("ok") or len(res.get("points") or []) != 2:
            self._log("        [!] 同形求解失败：%s" % res.get("reason"))
            self._pair_skip_until = time.time() + 8
            return False

        r = rect
        cxs = [r["x"] + x * r["w"] for x, y in res["points"]]
        cys = [r["y"] + y * r["h"] for x, y in res["points"]]
        # 面板内坐标落到标题/按钮区说明定位跑偏，放弃本次（交给通用流程兜底）
        if any(cy < r["y"] + 0.22 * r["h"] or cy > r["y"] + 0.86 * r["h"]
               for cy in cys):
            self._log("        [!] 配对点落在标题/按钮区，丢弃")
            self._pair_skip_until = time.time() + 8
            return False

        for i, (x, y) in enumerate(zip(cxs, cys)):
            self._log("        click  -> css(%.0f, %.0f)  [同形 #%d]" % (x, y, i + 1))
            if not human_click(self.page, x, y):
                self._log("        [!] 点击派发失败")
            time.sleep(random.uniform(0.30, 0.75))
        self._pair_skip_until = time.time() + 12
        return True

    def run(self, hint="", should_stop=None):
        result = {"ok": False, "reason": "", "steps": [], "final_state": None,
                  "model": None}
        last_state = None
        stuck = 0
        no_cap = 0

        def _abort():
            return bool(should_stop and should_stop())

        # 阶段 0：等待验证码控件注入。hCaptcha 静态资源常需 5~10s，
        # 过早截图会让 VLM 判成 "no_captcha" → agent 误以为任务完成。
        if self.wait_captcha:
            present, hits = has_captcha(self.page)
            if not present:
                wait_for_captcha(self.page, timeout=self.wait_timeout,
                                 verbose=self.verbose)

        for step in range(1, self.max_steps + 1):
            if _abort():
                result["reason"] = "外部信号要求停止（已放行）"
                return result
            time.sleep(random.uniform(*self.settle))
            obs = self.observe()
            if not obs:
                result["reason"] = "截图失败"
                self._log("  [-] 第 %d 步截图失败" % step)
                return result

            extra = ""
            if self.history:
                prev = self.history[-1]
                extra = ("上一步执行了 %s，但验证码仍在。请重新判断当前画面，"
                         "如果上次点击没命中目标，修正坐标；如果题型已变化，按新题型处理。"
                         % json.dumps(prev.get("actions"), ensure_ascii=False)[:300])

            plan, meta = ask_vlm(obs["jpeg"], hint=hint, extra=extra)
            if not plan:
                result["reason"] = "VLM 不可用: %s" % meta.get("error")
                self._log("  [-] 第 %d 步 VLM 失败: %s" % (step, meta.get("error")))
                result["steps"].append({"step": step, "error": meta.get("error")})
                return result

            state = str(plan.get("state") or "unknown").lower()
            actions = plan.get("actions") or []
            if not isinstance(actions, list):
                actions = []

            self._log("  [AI %ss] step=%d state=%s type=%s conf=%s | %s"
                      % (meta.get("elapsed"), step, state,
                         plan.get("captcha_type"), plan.get("confidence"),
                         str(plan.get("note", ""))[:70]))
            if plan.get("instruction"):
                self._log("        题目: %s" % str(plan.get("instruction"))[:110])

            result["model"] = meta.get("model")
            result["final_state"] = state

            if state == "solved":
                result["ok"] = True
                result["reason"] = "VLM 判定已通过"
                result["steps"].append({"step": step, "state": state, "plan": plan})
                self._log("  [+] 验证码已通过（step %d）" % step)
                return result
            if state == "no_captcha":
                present_now, _hits = has_captcha(self.page)
                if not present_now:
                    no_cap += 1
                    if no_cap >= 2:
                        result["ok"] = True
                        result["reason"] = "页面无验证码"
                        result["steps"].append({"step": step, "state": state, "plan": plan})
                        self._log("  [+] 页面无验证码（step %d，二次确认）" % step)
                        return result
                else:
                    no_cap += 1
                    if no_cap >= 3:
                        result["reason"] = "验证码控件存在但 AI 未识别，需人工介入"
                        result["steps"].append({"step": step, "state": state, "plan": plan})
                        self._log("  [!] 控件存在但 AI 连续 %d 次判为无验证码" % no_cap)
                        return result
                time.sleep(1.6)
                continue
            no_cap = 0
            if state == "blocked":
                result["reason"] = "被验证码站点拒绝: %s" % str(plan.get("note", ""))[:120]
                result["steps"].append({"step": step, "state": state, "plan": plan})
                return result
            if state == "loading":
                self._log("        题目加载中，等待…")
                time.sleep(1.6)
                continue

            # ---- 执行动作 ----
            pts = []
            for act in actions:
                t = (act.get("type") or "").lower()
                if t == "click":
                    pts.append(("click", float(act.get("x", 0)), float(act.get("y", 0))))
                elif t == "drag":
                    pts.append(("drag", float(act.get("x1", 0)), float(act.get("y1", 0))))

            if pts:
                _annotate(obs["jpeg"], pts, "s%d" % step)

            executed = 0
            for act in actions:
                if _abort():
                    result["reason"] = "外部信号要求停止（已放行）"
                    return result
                cmd = to_css(act, obs["vp"])
                if cmd[0] == "click":
                    self._log("        click  -> css(%.0f, %.0f)  [norm %.3f, %.3f]"
                              % (cmd[1], cmd[2], act.get("x", 0), act.get("y", 0)))
                    if human_click(self.page, cmd[1], cmd[2]):
                        executed += 1
                    time.sleep(random.uniform(0.25, 0.65))
                elif cmd[0] == "drag":
                    self._log("        drag   -> (%.0f,%.0f) → (%.0f,%.0f)"
                              % (cmd[1], cmd[2], cmd[3], cmd[4]))
                    if human_drag(self.page, cmd[1], cmd[2], cmd[3], cmd[4]):
                        executed += 1
                    time.sleep(random.uniform(0.4, 0.9))
                elif cmd[0] == "wait":
                    self._log("        wait")
                    time.sleep(1.2)
                elif cmd[0] == "done":
                    time.sleep(1.0)

            self.history.append({"step": step, "state": state, "actions": actions})
            result["steps"].append({"step": step, "state": state, "plan": plan,
                                    "executed": executed})

            if executed == 0 and not actions:
                stuck += 1
                self._log("        [!] AI 未给出可执行动作（连续 %d 次）" % stuck)
                if stuck >= 2:
                    result["reason"] = "AI 连续无动作，放弃"
                    return result
            else:
                stuck = 0

            # 状态未变化且动作与上一步完全一致 → 大概率是死循环，提前收敛
            if last_state == state and len(self.history) >= 2:
                if json.dumps(self.history[-1]["actions"], sort_keys=True) == \
                   json.dumps(self.history[-2]["actions"], sort_keys=True):
                    self._log("        [!] 动作与上一步完全相同，可能未生效")

            last_state = state
            if state == "checkbox" and executed:
                # 点完复选框后挑战面板需要 1~4s 展开；不等待会导致
                # 下一轮仍截到"复选框"，AI 反复点同一个位置（实测发生 3 次）。
                _log_ok = wait_challenge_active(self.page, timeout=9.0)
                if _log_ok:
                    self._log("        [+] 挑战面板已展开，等待题目渲染")
                    time.sleep(random.uniform(1.0, 1.8))
                else:
                    time.sleep(random.uniform(0.8, 1.4))
            elif state == "challenge":
                time.sleep(random.uniform(0.8, 1.6))
            time.sleep(random.uniform(*self.settle_after_click))

        result["reason"] = "达到最大步数 %d 仍未通过" % self.max_steps
        return result


def solve_captcha(page, max_steps=14, hint="", verbose=True, should_stop=None,
                  deadline=None, task_meta=None):
    """对外主入口：对给定 page 尝试自动通过验证码。

    默认走 v2「工具驱动」Agent（captcha_brain）：模型自己读题、自己决定
    调用 locate/sheet/zoom/click/drag/wait 哪个工具，代码里不写任何题型分支。
    设置环境变量 CAPTCHA_AGENT=legacy 可回退到 v1 的单段 prompt 循环。

    should_stop: 可选回调，返回 True 时立刻中止（用于「倒计时已生效」等外部信号）。
    """
    if os.environ.get("CAPTCHA_AGENT", "brain").lower() != "legacy":
        try:
            import captcha_brain
            return captcha_brain.solve(page, max_turns=max_steps,
                                       verbose=verbose, should_stop=should_stop,
                                       hint=hint, deadline=deadline,
                                       task_meta=task_meta)
        except Exception as e:
            if verbose:
                print("  [!] brain 模式不可用（%s），回退 legacy" % e)
    agent = VisionCaptchaAgent(page, max_steps=max_steps, verbose=verbose)
    return agent.run(hint=hint, should_stop=should_stop)


def probe_page(page):
    """只观察不动作：返回 VLM 对当前页面的判定。"""
    agent = VisionCaptchaAgent(page, max_steps=1, verbose=False)
    obs = agent.observe()
    if not obs:
        return {"ok": False, "reason": "截图失败"}
    plan, meta = ask_vlm(obs["jpeg"])
    return {"ok": bool(plan), "plan": plan, "meta": meta,
            "viewport": obs["vp"], "img_size": obs["img_size"]}


# ================================================================== HTTP 服务
def _pick_free_port(start=9520, end=9600):
    """挑一个未占用的本地端口（避免上次崩溃残留的 Chrome 占用调试端口）。"""
    import socket
    for p in range(start, end):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            try:
                s.close()
            except Exception:
                pass
    return start


def _connect(port=None, url=None):
    from DrissionPage import ChromiumPage
    if port:
        page = ChromiumPage("127.0.0.1:%d" % int(port))
    else:
        page = ChromiumPage()
    if url:
        page.get(url)
        time.sleep(2.0)
    return page


def serve(listen="127.0.0.1:8848"):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except Exception as e:
        print("[!] 缺少依赖，请先 pip install fastapi uvicorn pydantic")
        print("    ", e)
        return 2

    app = FastAPI(title="万能视觉打码 Agent", version="1.0")

    class SolveReq(BaseModel):
        port: int = 0
        url: str = ""
        max_steps: int = 12
        hint: str = ""

    @app.get("/health")
    def health():
        return {"ok": True, "model": VLM_MODEL, "endpoint": VLM_BASE}

    @app.post("/solve")
    def solve(req: SolveReq):
        try:
            page = _connect(port=req.port or None, url=req.url or None)
            return solve_captcha(page, max_steps=req.max_steps, hint=req.hint)
        except Exception as e:
            return {"ok": False, "reason": "%s: %s" % (type(e).__name__, e)}

    @app.post("/probe")
    def probe(req: SolveReq):
        try:
            page = _connect(port=req.port or None, url=req.url or None)
            return probe_page(page)
        except Exception as e:
            return {"ok": False, "reason": "%s: %s" % (type(e).__name__, e)}

    host, _, p = listen.partition(":")
    uvicorn.run(app, host=host or "127.0.0.1", port=int(p or 8848), log_level="info")
    return 0


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="万能视觉验证码 Agent")
    ap.add_argument("--url", default="https://accounts.hcaptcha.com/demo",
                    help="要处理的目标页面")
    ap.add_argument("--port", type=int, default=0, help="接管已开的 Chrome 调试端口")
    ap.add_argument("--rounds", type=int, default=1, help="连续测试轮数")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--probe", action="store_true", help="只观察不动作")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 接口")
    ap.add_argument("--listen", default="127.0.0.1:8848")
    args = ap.parse_args()

    if args.serve:
        return serve(args.listen)

    from DrissionPage import ChromiumPage, ChromiumOptions

    if args.port:
        page = _connect(port=args.port, url=args.url or None)
    else:
        profile = os.path.join(DEBUG_DIR, "_agent_profile")
        try:
            os.makedirs(profile, exist_ok=True)
        except Exception:
            pass
        co = ChromiumOptions()
        co.set_local_port(_pick_free_port())
        co.set_user_data_path(profile)
        for a in ("--disable-blink-features=AutomationControlled",
                  "--no-first-run", "--no-default-browser-check",
                  "--disable-infobars", "--window-size=1280,900"):
            try:
                co.set_argument(a)
            except Exception:
                pass
        if args.headless:
            co.headless()
        if args.proxy:
            co.set_proxy(args.proxy)
        page = ChromiumPage(co)
        page.get(args.url)
        time.sleep(2.5)

    print("=" * 78)
    print(" 万能视觉打码 Agent")
    print(" 模型 :", VLM_MODEL, "|", VLM_BASE.split("//")[-1].split("/")[0])
    print(" 页面 :", args.url)
    print("=" * 78)

    if args.probe:
        r = probe_page(page)
        print(json.dumps(r, ensure_ascii=False, indent=2)[:2500])
    else:
        ok_n = 0
        for i in range(1, args.rounds + 1):
            print("\n---------- 第 %d/%d 轮 ----------" % (i, args.rounds))
            r = solve_captcha(page, max_steps=args.max_steps)
            print("[结果] ok=%s  reason=%s  state=%s  步数=%d"
                  % (r.get("ok"), r.get("reason"), r.get("final_state"),
                     len(r.get("steps") or [])))
            if r.get("ok"):
                ok_n += 1
            if r.get("final_state") == "blocked":
                break
            if i < args.rounds:
                page.get(args.url)
                time.sleep(2.5)
        print("\n########## 汇总: %d/%d 轮通过 ##########" % (ok_n, args.rounds))

    if not args.keep_open:
        try:
            page.quit()
        except Exception:
            pass
    else:
        print("\n[*] 浏览器保持打开，Ctrl+C 退出…")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""captcha_brain.py — 工具驱动的验证码 Agent（v2）
================================================================================
为什么重写
--------------------------------------------------------------------------------
v1 是「代码判断题型 → 代码决定调用哪个专用解法」：
    if instruction 匹配 "两个相同" → 走同形配对流程
这条路的问题很明显 —— 题型是无限的，代码里的分支永远追不上；
而且一旦正则没命中，就会退化成通用 prompt 硬猜，表现极差。

v2 改成**模型自己编排工具**：
    模型每轮看到【全视口截图 + 验证码面板放大图】，
    自己读题目、自己决定先定位、还是先拼对照表、还是直接点、还是等加载。
    代码只负责“把工具执行好、把结果喂回去”。

工具清单（模型在 JSON 里自选，可一轮调用多个）
--------------------------------------------------------------------------------
    locate(target)      在面板里定位所有符合描述的对象，返回编号 + 坐标
    sheet(ids)          把已定位的对象逐个裁出、编号拼成对照表（图像）
    zoom(x,y,w,h)       把面板某个区域放大（用于看清细节/小字）
    click(x,y,space)    真实鼠标点击（space=panel 或 viewport）
    drag(x1,y1,x2,y2)   真实拖拽
    verify_pair(a,b)    只做二选一确认（是否同一形状）
    commit(...)         ★★ 通用原子动作：点选(ids/points)或拖拽(drag) → 点按钮 → 等回应
    solve_pair(ids)     ★ 配对题一步流：签名 → 代码比对 → 二次确认 → 提交
    submit()            精确点击底部按钮（颜色检测中心，不靠目测）
    wait(seconds)       等待（题目加载/动画）
    status()            查 token / 按钮文案
    finish(ok, note)    结束

配对题的两条快速路径（2026-11 新增）
--------------------------------------------------------------------------------
1) commit：把「点选/拖拽 + 提交」合并成一次调用，**适用所有题型**。
   实测一页从 2~3 轮往返（约 25~40s）压到 1 轮（约 10~20s）。
2) solve_pair：让模型给每块输出结构化签名（块数 n + 外框 box + 凸块方位 pro），
   由代码做分组比对挑出唯一候选，再把候选两块单独并排放大做二次确认。
   已知答案的样本上：签名 7.9s 得到唯一候选 [2,4]（正确），二次确认 5.5s 判 True。
   注意：二次确认必须用「候选对并排图」，拆成两张独立图片时模型会偏保守
   （同一对样本被判成 False，还慢到 55s）。

为什么这样更快更准
--------------------------------------------------------------------------------
· 模型擅长「就地判断」，不擅长「边定位边比形」：所以把定位和比较拆成两个
  独立工具，让模型自己按需调用。
· 比较任务交给对照表（同尺度、带编号、大图），比在小图上硬找可靠得多。
· 任何新题型（滑块、拼图、文字题、多页动物题）都不需要改代码，
  模型自己组合工具即可 —— 这就是“万能”。

对外接口（与 v1 保持一致，打号主程序无需改动）
--------------------------------------------------------------------------------
    from captcha_agent import solve_captcha
    solve_captcha(page, max_steps=12)
================================================================================
"""
import base64
from collections import deque
import io
import json
import math
import os
import random
import re
import sys
import threading
import time

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import captcha_agent as CA

DEBUG_DIR = CA.DEBUG_DIR
PROBE_FILE = os.path.join(HERE, "captcha_probe.jsonl")
_PROBE_LOCK = threading.Lock()

SYSTEM = """你是网页操作助手。用户在自己的浏览器里做图形核对题，你负责看图并告诉他点哪里、用工具替他点。

【每轮你会收到】
图①：验证码面板放大图（panel 坐标空间，左上边缘带 0.1~0.9 红色刻度尺；面板不可见时为全视口图）
图②（若存在）：工具返回的增强/对照图，或遇到横纹干扰题时系统自动附加的「去横纹清晰图」（panel 坐标系与图①完全一致）
文字：上一轮工具的执行结果

【坐标空间】
· space="viewport" → 坐标相对全视口，0~1 归一化
· space="panel"    → 坐标相对面板放大图，0~1 归一化。**处理题目内容时优先用这个**，更准。
· ⚠️ **看准刻度与区域再给坐标**（面板左侧与顶部标有 `0.1~0.9` 红色刻度尺）：
  - 顶部 `y=0.00~0.21` 是题干横幅，底部 `y=0.86~1.00` 是按钮栏，**所有图片内容都在 `y=0.21~0.85` 之间**！
  - 3×3 九宫格固定中心：`x = [0.18, 0.50, 0.82]`，`y = [0.32, 0.53, 0.75]`（**禁止用 `y=0.17` 或 `y=0.87`**，那会点到横幅或空白栏）。
- 【坐标系与区域识别铁律】
  - **九宫格题**（画面有明显 3×3 白色分割线，如“请点击包含…的每张图片”）→ **强烈推荐直接用 `commit(cells=[...])`**（按从左到右、从上到下编号 1~9：第1行 1,2,3 / 第2行 4,5,6 / 第3行 7,8,9，系统自动映射精准格心，**禁止自己心算 0.32/0.53/0.75 坐标**！仅非常规网格才用 `points`）。
  - **散落图形/物块配对题**（“点击两个相同的图形”/“两个相同”，背景是整块干扰纹理而非 3×3 九宫格）→ **严禁直接给 `commit(points=...)` 盲猜坐标**！必须先调用 `locate`（系统会自动裁出每块并做“去同心圆/去迷彩 + 黑线网格增强”双视图），再优先调用 `solve_pair`（或对照增强图调 `commit(ids=[a,b])`）。
  - **字母/图标拖拽配对题**（如“将字母拖动到适合它的位置”或拖拽补图）：
    1. 先找画面里**唯一带 `+ 移动`（或 `+ Move`）黑色胶囊标签的半透明悬浮方块**，其中心即唯一拖动起点 `(x1, y1)`；**禁止从静止底图元素开拖，也严禁 `x2≈x1` 且 `y2≈y1` 原地拖动**！
    2. 若背景有 6 个 2×3 排布的底图字母且被水平横纹遮挡：请先看**图②（去横纹清晰图）**认清悬浮块字母与 6 个底图字母，找到**与悬浮块完全相同的那个底图字母**（如悬浮块是 `A` 就拖去底图 `A`，是 `B` 就拖去底图 `B`），再对照红色刻度尺读准终点 `(x2, y2)`（例如右上常见 `y≈0.34~0.38`，右下常见 `y≈0.70~0.75`），直接调用 `commit(drag=[x1,y1,x2,y2], space="panel")`！
    3. 若是放射状/星形/花瓣等**图标拖拽**题：务必**逐个数清悬浮块与底图各候选的射线/花瓣/棱边数量及是否空心**（例如 6 根不能拖去 8 根，不要只挑最亮的白色底图）；若上轮某终点被判错且 `new_page=False`，禁止再拖去相近终点（换一个数量/特征真正匹配的候选）。
  - 【易错图块提醒】遇到“室温下会融化”类九宫格时：只有雪人、明确可见的透明冰块/杯中冰块、冰淇淋等会融化；**森林、雪山风景以及带方形底座的 3D 岩石/紫绿色山丘地形块都属于山石诱饵，不要选**。

【输出格式】严格输出单个 JSON，不要 Markdown，不要多余文字：
{"thought":"120 字内：先给每块一句速记（块数+轮廓+凸出位置），再给结论","tools":[{"tool":"...","...":...}]}

【工具】
1. {"tool":"locate","target":"要定位的对象描述"}
   在面板里找出所有符合描述的对象，返回编号和坐标（panel 空间）。
   适合：题目要求点某些图形/动物/物品时，用它先把候选都找出来。
2. {"tool":"sheet","ids":[1,2,3,4,5]}
   把 locate 找到的对象逐个裁出、编号、拼成一张对照表给你看。
   适合：需要**逐块对比**的题（找两个相同的、找唯一不同的）。
3. {"tool":"zoom","x":0.5,"y":0.5,"w":0.4,"h":0.3}
   把面板某块区域放大。适合看不清细节、或要读小字。
4. {"tool":"click","x":0.5,"y":0.5,"space":"panel"} （⚠️ 仅兜底用；点选/拖拽请直接用下面第 7 项 commit 一步完成并提交，禁止把 drag/click 和 submit 拆成两轮！）
5. {"tool":"drag","x1":0.2,"y1":0.5,"x2":0.7,"y2":0.5,"space":"panel"} （⚠️ 同上，补图/拖拽题必须直接用 {"tool":"commit","drag":[x1,y1,x2,y2],"space":"panel"}，不要单独调用 drag！）
6. {"tool":"wait","seconds":1.5}
7. {"tool":"commit","ids":[1,4],"verify":true}
   ⭐⭐ **九宫格/拖拽题默认动作**：一次调用完成「点选/拖拽 → 点底部按钮 → 等页面回应」（配对题请先 `locate`，再用 `solve_pair` 或 `commit(ids=...)`）。
   它会直接告诉你 passed=true（已通过）/ new_page=true（换到新题了）/ stage=verify（判成不同没点）。
   四种目标写法，可混用：
     · cells  : 九宫格 1~9 编号列表，如 [6,8,9]（1-3第一行、4-6第二行、7-9第三行，九宫格首选！）
     · ids    : locate 的编号列表，如 [1,4]、[1,2,3]（最准）
     · points : 直接给 panel 归一化坐标，如 [[0.30,0.42],[0.70,0.55]]（非九宫格或补充精调）
     · drag   : 一次拖拽 [x1,y1,x2,y2]（补图/拼图题用）
   verify 默认关闭；配对题实在拿不准时加 "verify":true（先二选一确认，判不同就不点）。
   **不要**再拆成 click→submit 两轮：每多一轮就多 10~20 秒，还多一次点漏的风险。
8. {"tool":"solve_pair","ids":[1,2,3,4,5]}
   ⭐ 配对题加强版（前提：先 locate）。系统内部：
     ① 每块单独出图，让模型给结构化签名：块数 n + 外框 box + 凸块方位 pro；
     ② 在代码里机械比对签名，挑出唯一候选对；
     ③ 再把候选两块单独并排放大做二次确认；
     ④ 确认通过后自动点击 + 提交，直接回报 passed / new_page。
   块数接近、肉眼容易混的配对题优先用它。
9. {"tool":"submit"}
   只点底部按钮、不做别的。已经用 commit 就不用它了（commit 内部已提交）。
10. {"tool":"verify_pair","a":1,"b":4}
   只做二选一确认，不点不提交。commit 内部已包含它，一般不单独用。
11. {"tool":"status"}  ⚠️ 一般不需要调用 —— 系统每轮结束都会自动检查 token。
12. {"tool":"finish","ok":true,"note":"通过/放弃原因"}

【通过判据 —— 必须记住】
· **面板消失不等于通过**！只有 hCaptcha 真的发 token 才算通过，
  用 status 看 has_token=true。
· 如果 status 显示 panel=false 且 has_token=false：说明这一轮挑战失败了，
  直接 finish(ok=false) 并说明原因，外层程序会重新触发或换号。
· 没有 token 时不要报 finish(ok=true)，否则会误判。
· 提交后如果画面直接换成新题 = 刚才那对被判错。**不要解释、不要 status，
  立刻按新题再来一遍**（这是免费多给的一次机会，抓紧时间）。

【比对类题目的铁律（找两个相同的 / 找唯一不同的）】
① 先 locate 把所有候选找出来（不要漏，数量通常 4~6）；
② locate 同轮就会附上对照表，直接看；
③ **在 thought 里逐格写速记再答题**：几块、整体轮廓（L/T/十字/梯形/锯齿）、
   凸出的小块在哪一侧。哪怕慢 5 秒也值得 —— 不数就选必错；
④ 由描述锁定 1~2 个候选对后：若块数相近、带旋转较难一眼分清，**优先调用 `solve_pair(ids=[1,2,...])`** 让系统做结构化签名+旋转对齐确认；十分确定时再直接 `commit(ids=[a,b], verify=true)`。严禁不数方块就盲目 `commit(ids=...)`。
⑤ 坐标一律用 locate 返回的编号（工具内部按编号取点），不要自己目测估算。
禁止"看起来差不多"这种判断 —— 相似但块数不同的绝不选。
最快的排除法：**先数块数**，块数不同立刻排除。

【节奏纪律 —— 面板是 120 秒「无操作」才关闭，不是总共 120 秒】
· 只要你还在持续操作，面板就不会关；真正会关的是「连续 100 秒什么都不做」。
  所以可以认真比对，但**单次思考不要超过 50 秒**。
· locate 会把对照表一起返回，通常不必再单独调 sheet；
  但如果两块拿不准，用 sheet(ids=[a,b]) 并排放大是**允许且鼓励**的（一次尝试最多两次）。
· 画面没变化就不要重复 locate（会浪费一次尝试的机会）。
· **一页 = 一轮**：① locate（对照表随本轮返回）→ ② commit(...) 一次做完动作+提交。
  拖拽补图题也一样：locate 不需要，直接 commit(drag=[...]) 一步到位。
  第三次往返 ≈ 25~40 秒的旧做法已经废弃，commit 一次 ≈ 10~20 秒。
· 凡是要"选目标再提交"的题，**一律用 commit**（配对/多选给 ids，网格给 points，补图给 drag）。
  配对题如果实在难分，用 commit(ids=[a,b], verify=true) 或 solve_pair 让系统机械比对。
· 禁止在同一页重复 locate；禁止连续 zoom 超过一次；拿不准就按最佳判断点，犹豫更亏。
· **一页只做一遍**：第 1 轮 locate+判断并点目标，第 2 轮点「下一步/检查」。做完就停。
· **被判错时不要乱点**。面板换题或关闭 = 这一轮失败，立刻 finish(ok=false) 结束，
  页面会重新出一道新题，下次再试即可。
  消息里会给出【距上次操作】的秒数；超过 90 秒不动面板就会关。别拖到那一步。


· 先读题目原文（面板顶部文字），再决定用什么工具。禁止不看清题目就点。
· 底部右侧按钮只有三种文案，含义完全不同：
  「跳过 / Skip」= 本题还没选中任何目标，**绝对不能点**（点=放弃）；
  「下一步 / Next」= 本页已选好，点它进入下一小题；
  「检查 / 验证 / Verify」= 最后一页，点它提交。
· 多页题（顶部有一排小点，实心=当前页）：每页选完点「下一步」，最后一页点「检查」。
· 已选中的目标身上会出现白色圆圈X标记 / 高亮：**不要再点它**，再点会取消选择。
  如果画面里已经出现了标记且按钮变成「下一步/检查」，本轮只点那个按钮。
· 点目标时要点在**目标图形内部**；不确定就把该区域 zoom 放大后再点。
· 不确定时优先 "wait" 或 "zoom"，不要乱点。
· 判断已完成（绿色对勾 / 面板消失 / 页面进入下一步）→ finish(ok=true)。"""


def parse_dsml(text):
    """解析 DeepSeek 原生工具调用格式（模型偶尔不用 JSON 而用这个）：

        <||DSML|| calls>
          <||DSML|| invoke name="verify_pair">
            <||DSML|| parameter name="a" string="true">3</||DSML|| parameter>
            ...
    """
    if not text or "DSML" not in text:
        return None
    import re
    tools = []
    for block in re.findall(r"invoke\s+name=\"([^\"]+)\"(.*?)(?:</\|\|DSML\|\|\s*invoke>|$)",
                            text, re.S):
        name, body = block
        call = {"tool": name.strip()}
        for pname, pval in re.findall(
                r"parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</\|\|DSML\|\|\s*parameter>",
                body, re.S):
            v = pval.strip()
            if re.match(r"^-?\d+$", v):
                v = int(v)
            elif re.match(r"^-?\d*\.\d+$", v):
                v = float(v)
            elif v.lower() in ("true", "false"):
                v = (v.lower() == "true")
            call[pname.strip()] = v
        tools.append(call)
    if not tools:
        return None
    thought = ""
    m = re.search(r"<\|\|DSML\|\|\s*thought>(.*?)</", text, re.S)
    if m:
        thought = m.group(1).strip()
    return {"thought": thought, "tools": tools}


def _jpeg(im, q=88):
    b = io.BytesIO()
    im.convert("RGB").save(b, format="JPEG", quality=q)
    return b.getvalue()


def _b64(j):
    return base64.b64encode(j).decode("utf-8")


def _annotate_panel_ruler(panel):
    """在发给 VLM 的 panel 副本边缘绘制 0.1~0.9 归一化刻度尺（不改动原图与归一化坐标系）。"""
    im = panel.convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(14, int(min(w, h) * 0.022)))
    except TypeError:
        font = ImageFont.load_default()
    color = (235, 45, 45)
    bg = (24, 24, 24)
    for k in range(1, 10):
        t = k / 10.0
        lbl = f".{k}"
        # 左侧 Y 轴刻度
        y = int(round(h * t))
        draw.line([(0, y), (14, y)], fill=color, width=3)
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(2, min(h - th - 4, y - th // 2))
        draw.rectangle([15, ty - 1, 19 + tw, ty + th + 2], fill=bg)
        draw.text((17, ty), lbl, fill=(255, 235, 120), font=font)
        # 顶部 X 轴刻度
        x = int(round(w * t))
        draw.line([(x, 0), (x, 12)], fill=color, width=3)
        tx = max(2, min(w - tw - 4, x - tw // 2))
        draw.rectangle([tx - 1, 13, tx + tw + 3, 15 + th], fill=bg)
        draw.text((tx + 1, 14), lbl, fill=(255, 235, 120), font=font)
    return im


def _has_horizontal_scanlines(panel):
    """检测面板题图区域是否存在明显水平周期条纹干扰（如新字母/图标拖拽题）。"""
    if panel is None:
        return False, 0.0, 0.0
    try:
        arr = np.asarray(panel.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        if h < 80 or w < 80:
            return False, 0.0, 0.0
        roi = arr[int(h * 0.22):int(h * 0.84), int(w * 0.06):int(w * 0.94)]
        gray = roi.mean(axis=2)
        v_diff = float(np.mean(np.abs(np.diff(gray, axis=0))))
        h_diff = float(np.mean(np.abs(np.diff(gray, axis=1))))
        detected = v_diff > 7.5 and v_diff > h_diff * 1.35
        return detected, round(v_diff, 2), round(h_diff, 2)
    except Exception:
        return False, 0.0, 0.0


def _clean_horizontal_scanlines(panel, k: int = 10):
    """仅对题图内容区 (y=0.21~0.85) 做垂直去横纹平滑 + 对比度增强，保持坐标系与红刻度尺一致。"""
    im = panel.convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    h, w, _ = arr.shape
    y1, y2 = int(h * 0.21), int(h * 0.85)
    if y2 - y1 > k + 4 and w > 20:
        roi = arr[y1:y2]
        cs = np.cumsum(np.pad(roi, ((k, 0), (0, 0), (0, 0)), mode="edge"), axis=0)
        box = np.clip((cs[k:] - cs[:-k]) / float(k), 0, 255).astype(np.uint8)
        clean_roi = ImageOps.autocontrast(Image.fromarray(box, mode="RGB"), cutoff=1)
        arr[y1:y2] = np.asarray(clean_roi, dtype=np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    return _annotate_panel_ruler(out)


def _read_challenge_prompt(page) -> str:
    """尽最大可能从 DrissionPage challenge iframe 读取题目文本；失败返回空字符串。"""
    if page is None:
        return ""
    try:
        frame = None
        if hasattr(page, "get_frame"):
            try:
                frame = page.get_frame('css:iframe[src*="hcaptcha.com"][src*="challenge"]', timeout=0.25)
            except Exception:
                frame = None
            if frame is None:
                try:
                    frame = page.get_frame('xpath://iframe[contains(@src,"challenge")]', timeout=0.2)
                except Exception:
                    frame = None
        if frame is not None and hasattr(frame, "run_js"):
            txt = frame.run_js(
                "(() => { const el = document.querySelector('.prompt-text span, .prompt-text, .prompt-padding h2, h2'); "
                "return el ? (el.innerText || el.textContent || '').trim() : ''; })()",
                as_expr=True,
            )
            if isinstance(txt, str) and txt.strip():
                return txt.strip()[:200]
    except Exception:
        pass
    return ""


def _classify_challenge_kind(prompt_text: str, thought: str, tools: list, has_scanlines: bool) -> str:
    """将题型归类以便探针按题型统计过题率。"""
    txt = f"{prompt_text or ''} {thought or ''}".lower()
    tool_names = {str((c or {}).get("tool") or "").lower() for c in (tools or []) if isinstance(c, dict)}
    has_drag = any(isinstance(c, dict) and (c.get("drag") or (c.get("tool") or "").lower() == "drag") for c in (tools or []))
    has_cells = any(isinstance(c, dict) and c.get("cells") for c in (tools or []))
    if has_cells or "九宫格" in txt or "旋转" in txt or "融化" in txt:
        return "grid_3x3"
    if has_scanlines and (has_drag or "字母" in txt or "拖动" in txt or "适合" in txt):
        return "drag_letter_scanline"
    if has_drag or "拖动" in txt or "拖拽" in txt or "拼图" in txt:
        return "drag_puzzle"
    if "locate" in tool_names or "solve_pair" in tool_names or "相同" in txt:
        return "pair_match"
    if any(isinstance(c, dict) and (c.get("points") or (c.get("tool") or "").lower() == "click") for c in (tools or [])):
        return "point_click"
    return "unknown"


def record_captcha_probe(event: dict) -> None:
    """线程安全地追加打码新题型探针记录至 producer/captcha_probe.jsonl（超 20MB 自动轮转）。"""
    try:
        rec = dict(event or {})
        rec.setdefault("ts", round(time.time(), 3))
        rec.setdefault("iso", time.strftime("%Y-%m-%d %H:%M:%S"))
        line = json.dumps(rec, ensure_ascii=False)
        with _PROBE_LOCK:
            if os.path.exists(PROBE_FILE) and os.path.getsize(PROBE_FILE) > 20 * 1024 * 1024:
                bak = PROBE_FILE + ".1"
                if os.path.exists(bak):
                    os.remove(bak)
                os.replace(PROBE_FILE, bak)
            with open(PROBE_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def get_captcha_probe_summary(limit: int = 40) -> dict:
    """汇总 producer/captcha_probe.jsonl 中各题型的求解表现、过题率与近期失败样本。"""
    if not os.path.exists(PROBE_FILE):
        return {"ok": True, "total_records": 0, "by_kind": {}, "recent_solves": []}
    rows = []
    with _PROBE_LOCK:
        with open(PROBE_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
    solves = [r for r in rows if r.get("event") == "solve_summary"]
    turns = [r for r in rows if r.get("event") == "turn"]
    by_kind = {}
    for r in solves:
        k = str(r.get("challenge_kind") or "unknown")
        st = by_kind.setdefault(k, {"solves": 0, "passed": 0, "failed": 0, "avg_elapsed_s": 0.0, "_sum_s": 0.0})
        st["solves"] += 1
        if r.get("final_ok"):
            st["passed"] += 1
        else:
            st["failed"] += 1
        st["_sum_s"] += float(r.get("total_elapsed_s") or 0.0)
    for st in by_kind.values():
        n = st["solves"]
        st["pass_rate"] = round(st["passed"] / n, 3) if n else 0.0
        st["avg_elapsed_s"] = round(st.pop("_sum_s") / n, 2) if n else 0.0
    return {
        "ok": True,
        "probe_file": PROBE_FILE,
        "total_turns": len(turns),
        "total_solves": len(solves),
        "by_kind": by_kind,
        "recent_solves": solves[-max(1, int(limit or 20)):],
        "recent_turns": turns[-max(1, int(limit or 20)):],
    }


# ---------------------------------------------------------------------------
# 抗识别图形题预处理（去同心圆莫尔纹 + 去迷彩均衡 + 黑帽缝线提取 + 旋转对齐）
# ---------------------------------------------------------------------------
def _gauss_blur_float(arr2d, sigma):
    """Gaussian blur on an arbitrary-range 2D float32 array via Pillow C filter."""
    lo = float(np.min(arr2d))
    hi = float(np.max(arr2d))
    span = hi - lo
    if span < 1e-6:
        return arr2d.astype(np.float32, copy=True)
    u8 = np.clip((arr2d - lo) / span * 255.0, 0, 255).astype(np.uint8)
    bl = np.asarray(
        Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma))),
        dtype=np.float32,
    )
    return bl / np.float32(255.0) * np.float32(span) + np.float32(lo)


def _gauss_blur_rgb(arr_rgb, sigma):
    im = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)


def _grey_closing_2d(arr2d, ksize=9):
    """Morphological closing (dilate then erode) to bridge narrow dark cube seams."""
    lo = float(np.min(arr2d))
    hi = float(np.max(arr2d))
    span = hi - lo
    if span < 1e-6:
        return arr2d.astype(np.float32, copy=True)
    u8 = np.clip((arr2d - lo) / span * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")
    cl = im.filter(ImageFilter.MaxFilter(ksize)).filter(ImageFilter.MinFilter(ksize))
    return np.asarray(cl, dtype=np.float32) / np.float32(255.0) * np.float32(span) + np.float32(lo)


def _sobel_mag(gray):
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy) * np.float32(0.25)


def _binary_morph(mask, dilate_iters=0, erode_iters=0):
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(max(0, int(dilate_iters))):
        im = im.filter(ImageFilter.MaxFilter(3))
    for _ in range(max(0, int(erode_iters))):
        im = im.filter(ImageFilter.MinFilter(3))
    return np.asarray(im) > 127


def _connected_components(mask):
    """8-connected component labeling in pure numpy/deque (no scipy required)."""
    h, w = mask.shape
    lbl = np.zeros((h, w), dtype=np.int32)
    comps = []
    cur = 0
    ys_all, xs_all = np.where(mask)
    for sy, sx in zip(ys_all.tolist(), xs_all.tolist()):
        if lbl[sy, sx] != 0:
            continue
        cur += 1
        q = deque([(sy, sx)])
        lbl[sy, sx] = cur
        pts_y, pts_x = [], []
        while q:
            y, x = q.popleft()
            pts_y.append(y)
            pts_x.append(x)
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx = x + dx
                    if 0 <= nx < w and mask[ny, nx] and lbl[ny, nx] == 0:
                        lbl[ny, nx] = cur
                        q.append((ny, nx))
        comps.append((cur, np.asarray(pts_y, dtype=np.int32), np.asarray(pts_x, dtype=np.int32)))
    return lbl, comps


def _joint_bilateral_rgb(arr_rgb, sigma_s=3.0, sigma_r=15.0):
    """Edge-preserving bilateral filter smoothing concentric rings while keeping cube seams crisp."""
    rad = int(np.ceil(sigma_s * 1.6))
    h, w, _ = arr_rgb.shape
    pad = np.pad(arr_rgb, ((rad, rad), (rad, rad), (0, 0)), mode="reflect")
    num = np.zeros_like(arr_rgb, dtype=np.float32)
    den = np.zeros((h, w, 1), dtype=np.float32)
    inv_2s2 = np.float32(1.0 / (2.0 * sigma_s * sigma_s))
    inv_2r2 = np.float32(1.0 / (2.0 * sigma_r * sigma_r))
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            ws = np.float32(np.exp(-(dx * dx + dy * dy) * inv_2s2))
            sh = pad[rad + dy:rad + dy + h, rad + dx:rad + dx + w]
            diff = sh - arr_rgb
            d2 = np.mean(diff * diff, axis=2, keepdims=True)
            wt = ws * np.exp(-d2 * inv_2r2)
            num += wt * sh
            den += wt
    return num / np.maximum(den, np.float32(1e-6))


def enhance_block_tile(tile_pil, work_side=210):
    """Suppress periodic ring/camo noise and accentuate 3D cube faces & dividing seams.

    Returns (enhanced_pil, fg_mask, seam_feat) at working resolution.
    """
    w0, h0 = tile_pil.size
    scale = float(work_side) / float(max(1, w0, h0))
    rw = max(32, int(round(w0 * scale)))
    rh = max(32, int(round(h0 * scale)))
    work = tile_pil.convert("RGB").resize((rw, rh), Image.BICUBIC)
    arr = np.asarray(work, dtype=np.float32)

    bf = _joint_bilateral_rgb(arr, sigma_s=3.0, sigma_r=15.0)
    bf = _joint_bilateral_rgb(bf, sigma_s=2.6, sigma_r=11.0)
    gray = 0.299 * bf[:, :, 0] + 0.587 * bf[:, :, 1] + 0.114 * bf[:, :, 2]

    sig_bg = max(14.0, min(rw, rh) * 0.10)
    smooth_bg_g = _gauss_blur_float(gray, sig_bg)
    rel = gray - smooth_bg_g

    # Clamp minimum stretch span so low-amplitude background texture stays calm gray
    p5, p97 = np.percentile(rel, [4, 97])
    span = max(38.0, float(p97 - p5))
    mid = 0.5 * (float(p5) + float(p97))
    lo = min(float(p5), mid - 0.5 * span)
    hi = max(float(p97), mid + 0.5 * span)
    norm = np.clip((rel - lo) / max(1e-5, hi - lo) * 235.0 + 15.0, 0, 255)

    # Black-hat seam extraction + crisp facet gradients
    closed = _grey_closing_2d(gray, ksize=9)
    seams = np.maximum(0.0, closed - gray)
    grad = _sobel_mag(gray)
    seam_raw = seams * 1.45 + grad * 0.28
    s_hi = max(22.0, float(np.percentile(seam_raw, 98.5)))
    seam_norm = np.clip(seam_raw / s_hi * 215.0, 0, 225)

    enhanced = np.clip(norm - seam_norm * 0.85, 0, 255).astype(np.uint8)
    enhanced_pil = Image.fromarray(enhanced).convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=1.4, percent=145, threshold=2)
    )

    # Center-anchored silhouette & edge-seam feature map for rotation alignment
    # 使用双向截断对比度（同时覆盖暗紫面与白亮顶面），避免暗阴影拉高阈值误砍顶面
    smooth_bg_rgb = _gauss_blur_rgb(bf, sig_bg)
    local_col_dist = np.sqrt(np.sum((bf - smooth_bg_rgb) ** 2, axis=2))
    lum_diff = np.abs(gray - smooth_bg_g)
    dark_shadow = np.maximum(0.0, smooth_bg_g - gray)
    fg_score = (np.minimum(lum_diff, 30.0) * 0.9
                + np.minimum(np.maximum(0.0, local_col_dist - 6.0), 30.0) * 1.1)
    fg_smooth = _gauss_blur_float(fg_score, 1.3)
    bg_vals = fg_smooth[fg_smooth <= np.percentile(fg_smooth, 55)]
    med_bg = float(np.median(bg_vals)) if bg_vals.size else float(np.median(fg_smooth))
    mad_bg = float(np.median(np.abs(bg_vals - med_bg))) + 1e-4 if bg_vals.size else 1.0
    peak_fg = float(np.percentile(fg_smooth, 94))
    thr = max(med_bg + 3.2 * mad_bg, med_bg + 0.34 * (peak_fg - med_bg))
    # 轻微开运算去噪点但不胀满凹槽（保持缺口与凸出清晰）
    m = _binary_morph(fg_smooth > thr, dilate_iters=0, erode_iters=1)
    m = _binary_morph(m, dilate_iters=1, erode_iters=0)
    lbl, comps = _connected_components(m)
    cy, cx = rh / 2.0, rw / 2.0
    best_mask = np.zeros_like(m)
    best_score = -1e9
    for cid, ys, xs in comps:
        if len(xs) < 60:
            continue
        d = float(np.hypot(xs.mean() - cx, ys.mean() - cy))
        contains = bool(lbl[int(cy), int(cx)] == cid)
        sc = (5000.0 if contains else 0.0) + len(xs) - d * 28.0
        if sc > best_score:
            best_score = sc
            best_mask = (lbl == cid)

    roi = _binary_morph(best_mask, dilate_iters=2, erode_iters=0)
    feat = (seams * 1.5 + grad * 0.35 + dark_shadow * 0.5) * roi
    if float(feat.max()) > 0:
        feat = feat / float(feat.max())
    return enhanced_pil, best_mask.astype(np.float32), feat.astype(np.float32)


def _center_pad_square(arr2d, size=64):
    ys, xs = np.where(arr2d > 0.05)
    if len(xs) < 10:
        return np.zeros((size, size), dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub = arr2d[y0:y1, x0:x1]
    im = Image.fromarray((np.clip(sub, 0, 1) * 255).astype(np.uint8), mode="L")
    k = (size * 0.72) / float(max(sub.shape))
    nw = max(4, int(im.size[0] * k))
    nh = max(4, int(im.size[1] * k))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def _best_rotation_align(maskA, featA, maskB, featB):
    """Search 0..355 deg (5 deg step) for the best in-plane rotation aligning B onto A."""
    mA = _center_pad_square(maskA, 64) > 0.35
    fA = _gauss_blur_float(_center_pad_square(featA, 64), 1.2)
    fA_norm = float(np.linalg.norm(fA)) + 1e-8
    im_mB = Image.fromarray((_center_pad_square(maskB, 64) * 255).astype(np.uint8), mode="L")
    im_fB = Image.fromarray((_center_pad_square(featB, 64) * 255).astype(np.uint8), mode="L")
    best = {"score": -1.0, "iou": 0.0, "ncc": 0.0, "angle": 0}
    for ang in range(0, 360, 5):
        rm = np.asarray(im_mB.rotate(ang, resample=Image.BILINEAR), dtype=np.float32) / 255.0 > 0.35
        iou = float(np.sum(mA & rm)) / (float(np.sum(mA | rm)) + 1e-8)
        rf = np.asarray(im_fB.rotate(ang, resample=Image.BILINEAR), dtype=np.float32) / 255.0
        rf_s = _gauss_blur_float(rf, 1.2)
        ncc = float(np.sum(fA * rf_s)) / (fA_norm * (float(np.linalg.norm(rf_s)) + 1e-8))
        sc = 0.55 * iou + 0.45 * ncc
        if sc > best["score"]:
            best = {"score": sc, "iou": iou, "ncc": ncc, "angle": ang}
    return best


def _has_3x3_grid(panel):
    """Return True if panel challenge region shows crisp white 3x3 grid dividers."""
    if panel is None:
        return False
    try:
        W, H = panel.size
        box = (int(W * 0.08), int(H * 0.21), int(W * 0.92), int(H * 0.85))
        sub = panel.crop(box).convert("L").resize((90, 90), Image.BILINEAR)
        arr = np.asarray(sub, dtype=np.float32)
        # 3x3 hCaptcha tiles are separated by bright white lines (~235..255) near 1/3 and 2/3
        v_hits = sum(
            max(float(np.mean(arr[:, c] > 235)) for c in range(center - 5, center + 6)) > 0.72
            for center in (30, 60)
        )
        h_hits = sum(
            max(float(np.mean(arr[r, :] > 235)) for r in range(center - 5, center + 6)) > 0.72
            for center in (30, 60)
        )
        return v_hits >= 2 and h_hits >= 2
    except Exception:
        return False


def _detect_canvas_objects_cv(panel):
    """Fallback CV salient-object detector for single-canvas scatter puzzles."""
    if panel is None:
        return []
    W, H = panel.size
    y0, y1 = int(H * 0.225), int(H * 0.825)
    x0, x1 = int(W * 0.16), int(W * 0.84)
    arr_b = np.asarray(panel.crop((x0, y0, x1, y1)), dtype=np.float32)
    chroma = np.max(arr_b, axis=2) - np.min(arr_b, axis=2)
    lum = np.mean(arr_b, axis=2)
    active = (chroma > 10) | (lum < 218)
    cx = np.where(np.mean(active, axis=0) > 0.65)[0]
    cy = np.where(np.mean(active, axis=1) > 0.65)[0]
    if len(cx) > 60 and len(cy) > 60:
        bx0, by0 = x0 + int(cx[0]) + 10, y0 + int(cy[0]) + 10
        bx1, by1 = x0 + int(cx[-1]) - 10, y0 + int(cy[-1]) - 10
    else:
        bx0, by0, bx1, by1 = int(W * 0.175), int(H * 0.235), int(W * 0.825), int(H * 0.815)
    if bx1 - bx0 < 80 or by1 - by0 < 80:
        return []

    arr = np.asarray(panel.crop((bx0, by0, bx1, by1)), dtype=np.float32)
    ch, cw, _ = arr.shape
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sig = max(14.0, min(ch, cw) * 0.05)
    bg_l = _gauss_blur_float(gray, sig)
    lum_diff = np.abs(bg_l - gray)
    dark_diff = np.maximum(0.0, bg_l - gray)
    bg_rgb = _gauss_blur_rgb(arr, sig)
    col_diff = np.sqrt(np.sum((arr - bg_rgb) ** 2, axis=2))
    grad = _sobel_mag(gray)
    e = dark_diff * 0.9 + lum_diff * 0.6 + np.maximum(0.0, col_diff - 8.0) * 1.0 + np.maximum(0.0, grad - 28.0) * 0.2
    se = _gauss_blur_float(e, max(4.0, min(ch, cw) * 0.012))
    margin = max(18, int(min(ch, cw) * 0.055))
    se[:margin, :] = 0
    se[-margin:, :] = 0
    se[:, :margin] = 0
    se[:, -margin:] = 0
    inner = se[margin:-margin, margin:-margin]
    if inner.size == 0:
        return []
    med = float(np.median(inner))
    mad = float(np.median(np.abs(inner - med))) + 1e-5
    thr = max(float(np.percentile(inner, 84)), med + 2.8 * mad)
    m = _binary_morph(se > thr, dilate_iters=3, erode_iters=1)
    lbl, comps = _connected_components(m)
    min_area = (min(ch, cw) * 0.042) ** 2
    max_area = (min(ch, cw) * 0.34) ** 2
    cand = []
    for _cid, ys, xs in comps:
        if not (min_area <= len(xs) <= max_area):
            continue
        ox0, ox1 = int(xs.min()), int(xs.max())
        oy0, oy1 = int(ys.min()), int(ys.max())
        pw, ph = ox1 - ox0, oy1 - oy0
        if pw < 28 or ph < 28 or not (0.28 <= pw / float(ph) <= 3.5):
            continue
        wts = se[ys, xs]
        ocx = float(np.average(xs, weights=wts))
        ocy = float(np.average(ys, weights=wts))
        cand.append((
            float(np.sum(wts)),
            (bx0 + ocx) / float(W),
            (by0 + ocy) / float(H),
        ))
    cand.sort(key=lambda t: t[0], reverse=True)
    cand = cand[:5]
    cand.sort(key=lambda t: (round(t[2], 1), t[1]))
    return [(x, y) for _, x, y in cand]


# ---------------------------------------------------------------------------
# 多图提问 + 结构化签名比对
# ---------------------------------------------------------------------------
# CA.ask_vlm 只支持单图。配对题要做两件单图做不到的事：
#   ① 一次把 5 块候选的原图都发过去，让模型给出可比对的"签名"；
#   ② 把候选两块各自一张图发过去做二次确认（拼图会被压到固定边长，单图分辨率更高）。
# 所以这里单独实现一个多图通道，其余逻辑（端点轮换、超时、JSON 解析）与 ask_vlm 一致。
def _ask_multi(images, prompt, effort=None, timeout=60, chain=None):
    """一次请求带多张图。返回 (plan_dict, meta)。"""
    content = [{"type": "text", "text": prompt}]
    for i, im in enumerate(images):
        content.append({"type": "text", "text": "图%d：" % (i + 1)})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + _b64(im)}})
    messages = [{"role": "user", "content": content}]
    # 面板是「闲置」超时：单次调用拖太久会连带把整轮拖死，所以给总预算
    return CA.call_vlm_endpoints(
        messages,
        chain=chain,
        effort=effort,
        timeout=timeout,
        ask_budget=85,
        extra_parsers=(parse_dsml,),
        include_raw=True,
    )


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "same", "是", "相同")
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _norm_sig(it):
    """把模型给的一块签名规整成 {id,n,box,pro}。任何解析不出的字段置空。"""
    if not isinstance(it, dict):
        return None
    try:
        i = int(it.get("id"))
    except Exception:
        return None
    n = it.get("n", it.get("blocks", it.get("count")))
    try:
        n = int(n)
    except Exception:
        n = None
    box = it.get("box") or it.get("bbox") or []
    if isinstance(box, str):
        box = re.findall(r"\d+", box)
    vals = []
    for v in (box if isinstance(box, (list, tuple)) else []):
        try:
            vals.append(int(round(float(v))))
        except Exception:
            pass
    pro = it.get("pro") or it.get("protrusions") or it.get("protrusion") or ""
    if isinstance(pro, (list, tuple)):
        toks = [str(v).strip().upper() for v in pro]
    else:
        toks = [t.strip().upper() for t in re.split(r"[,，;；\s]+", str(pro))]
    return {"id": i, "n": n, "box": vals[:2], "pro": [t for t in toks if t]}


def _sig_key(s, strict=True):
    """签名的可比对键。

    strict=True  : 块数 + 外框(宽高排序，抵消旋转) + 凸块方位与格数（完全一致）
    strict=False : 块数 + 外框(排序) + 只比凸块格数（忽略方位，兜底用）
    块数 n 是硬门槛 —— 实测模型数块数最稳，块数不同绝不可能是同一个图形。
    """
    n = s.get("n")
    if not isinstance(n, int) or n <= 0:
        return None
    box = tuple(sorted(v for v in (s.get("box") or []) if isinstance(v, int) and v > 0))
    if strict:
        pro = tuple(sorted(s.get("pro") or []))
    else:
        pro = tuple(sorted(re.sub(r"[^0-9]", "", t) or t for t in (s.get("pro") or [])))
    return (n, box, pro)


def _pair_candidates(sigs):
    """按签名分组，输出候选对列表（严格匹配排在前面）。"""
    out, seen = [], set()
    for strict in (True, False):
        groups = {}
        for s in sigs:
            if not isinstance(s, dict):
                continue
            k = _sig_key(s, strict)
            if k is None:
                continue
            groups.setdefault(k, []).append(s["id"])
        for ids in groups.values():
            if len(ids) < 2:
                continue
            ids = sorted(ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    key = (ids[i], ids[j])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"a": ids[i], "b": ids[j], "strict": strict})
    out.sort(key=lambda c: (not c["strict"], c["a"], c["b"]))
    return out


class Tools(object):
    """工具执行器：只做事，不做决策。"""

    def __init__(self, page, verbose=True):
        self.page = page
        self.verbose = verbose
        self.rect = None      # 面板矩形（CSS）
        self.panel = None     # 面板放大图（PIL）
        self.viewport = None
        self.objects = []     # locate 结果：[(x,y)] panel 归一化
        self._tile_cache = {}

    # ---------------------------------------------------------------- 坐标
    def _panel_to_css(self, x, y):
        r = self.rect
        return r["x"] + float(x) * r["w"], r["y"] + float(y) * r["h"]

    def _viewport_to_css(self, x, y):
        vp = self.viewport
        return float(x) * vp["w"], float(y) * vp["h"]

    def _resolve(self, x, y, space):
        if (space or "panel") == "viewport" or not self.rect:
            return self._viewport_to_css(x, y)
        return self._panel_to_css(x, y)

    def log(self, m):
        if self.verbose:
            print(m)

    # ---------------------------------------------------------------- 工具
    def click(self, x, y, space="panel"):
        cx, cy = self._resolve(x, y, space)
        vp = self.viewport or {"w": 1280, "h": 800}
        # 面板矩形偶尔会取到过期/错位的值，点出视口等于没点
        if not (0 <= cx <= vp["w"] - 2 and 0 <= cy <= vp["h"] - 2):
            self.log("        [!] 点击落在视口外 css(%.0f,%.0f)，已拒绝" % (cx, cy))
            return {"ok": False, "error": "坐标越界，先重新观察一次"}
        ok = CA.human_click(self.page, cx, cy)
        self.log("        click  [%s %.3f,%.3f] -> css(%.0f,%.0f) %s"
                 % (space, x, y, cx, cy, "ok" if ok else "FAIL"))
        return {"ok": bool(ok), "css": [round(cx), round(cy)]}

    def drag(self, x1, y1, x2, y2, space="panel"):
        a = self._resolve(x1, y1, space)
        b = self._resolve(x2, y2, space)
        ok = CA.human_drag(self.page, a[0], a[1], b[0], b[1])
        self.log("        drag   [%s] (%.3f,%.3f)->(%.3f,%.3f) %s"
                 % (space, x1, y1, x2, y2, "ok" if ok else "FAIL"))
        return {"ok": bool(ok)}

    def wait(self, seconds=1.5):
        s = max(0.2, min(8.0, float(seconds)))
        time.sleep(s)
        return {"waited": round(s, 1)}

    def locate(self, target):
        img = self.panel
        if img is None:
            return {"ok": False, "error": "当前没有面板放大图，无法定位"}
        prompt = ("这是一张题目面板截图。请把图形区域里所有独立的图形都找出来，"
                  "只输出它们的中心点。\n参考描述（别因为它漏掉候选）：%s\n"
                  "坐标相对本图，0~1 归一化，保留三位小数。只输出 JSON：\n"
                  '{"objects":[{"id":1,"x":0.000,"y":0.000}]}\n'
                  "不要判断哪个相同，不要给动作。" % target)
        plan, meta = CA.ask_vlm(_jpeg(img), hint=prompt, force=True,
                                raw_prompt=True, chain=CA.VLM_SUB)
        pts = []
        if isinstance(plan, dict):
            arr = plan.get("objects") or plan.get("shapes") or []
            for it in arr if isinstance(arr, list) else []:
                try:
                    x, y = float(it.get("x")), float(it.get("y"))
                except Exception:
                    continue
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    pts.append((x, y))
        cv_used = False
        if len(pts) < 2 and not _has_3x3_grid(img):
            cv_pts = _detect_canvas_objects_cv(img)
            if len(cv_pts) >= 2:
                pts = cv_pts
                cv_used = True
        self.objects = pts
        self._tile_cache = {}
        self.log("        locate %s -> %d 个%s (%ss)"
                 % (target[:18], len(pts), " [CV兜底]" if cv_used else "", meta.get("elapsed")))
        if not pts:
            return {"ok": False, "error": "没定位到对象", "elapsed": meta.get("elapsed")}
        out = {"ok": True, "count": len(pts),
               "objects": [{"id": i + 1, "x": round(x, 3), "y": round(y, 3)}
                           for i, (x, y) in enumerate(pts)]}
        # 同轮把对照表一起返回：少一轮往返 = 少 10~20s，验证码是有时间窗的
        sh = self.sheet()
        if sh.get("image"):
            out["image"] = sh["image"]
            out["note"] = "对照表已附在本轮图③，编号与上面 objects 的 id 一致"
        return out

    def _piece_bbox(self, x, y, self_idx=None):
        """按前景显著能量 + Voronoi 邻块隔离求单块 bbox，避免迷彩误漏或邻块串入。

        同时计入暗影面与亮顶面相对局部背景的亮度/色度偏离，并以背景底噪中位+MAD
        做自适应阈值（防止亮面或长条物块只截到一条阴影边）；再用净膨胀合并单个积木
        的全部分块，同时用 Voronoi 排除更靠近其它 locate 中心的区域。
        """
        img = self.panel
        W, H = img.size
        # 严格限制在题目正方形画布区域内（避开顶部题干、底部按钮与外围灰框）
        can_x0, can_y0 = int(W * 0.16), int(H * 0.22)
        can_x1, can_y1 = int(W * 0.84), int(H * 0.82)
        win = int(min(W, H) * 0.19)
        cx, cy = int(x * W), int(y * H)
        x0 = max(can_x0, cx - win)
        y0 = max(can_y0, cy - win)
        x1 = min(can_x1, cx + win)
        y1 = min(can_y1, cy + win)
        if x1 - x0 < 36 or y1 - y0 < 36:
            return None

        sub = np.asarray(img.crop((x0, y0, x1, y1))).astype(np.float32)
        h, w, _ = sub.shape
        gray = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
        sig = max(16.0, min(h, w) * 0.18)
        bg_l = _gauss_blur_float(gray, sig)
        lum_diff = np.abs(bg_l - gray)
        dark_diff = np.maximum(0.0, bg_l - gray)
        bg_rgb = _gauss_blur_rgb(sub, sig)
        col_diff = np.sqrt(np.sum((sub - bg_rgb) ** 2, axis=2))
        grad = _sobel_mag(gray)
        energy = (dark_diff * 0.9
                  + lum_diff * 0.65
                  + np.maximum(0.0, col_diff - 7.0) * 1.1
                  + np.maximum(0.0, grad - 26.0) * 0.22)
        se = _gauss_blur_float(energy, max(3.0, min(h, w) * 0.025))

        # Voronoi 屏蔽：明显更接近其它对象的像素归零，避免相邻积木粘连
        lx, ly = cx - x0, cy - y0
        yy, xx = np.ogrid[:h, :w]
        d_self = (xx - lx) ** 2 + (yy - ly) ** 2
        allowed = np.ones((h, w), dtype=bool)
        for oi, (ox_n, oy_n) in enumerate(self.objects):
            if self_idx is not None and oi == self_idx:
                continue
            ox, oy = int(ox_n * W) - x0, int(oy_n * H) - y0
            if (ox - lx) ** 2 + (oy - ly) ** 2 < 144:
                continue
            d_other = (xx - ox) ** 2 + (yy - oy) ** 2
            allowed &= (d_self <= d_other * 1.05)
        se_valid = se[allowed]
        if se_valid.size == 0:
            return None

        # 用较小的背景分位数(40%)与局部底噪估计阈值，保证亮顶面+暗侧面整体连通不被截断
        bg_floor = float(np.percentile(se_valid, 45))
        mad = float(np.median(np.abs(se_valid[se_valid <= np.percentile(se_valid, 70)] - bg_floor))) + 1e-5
        thr = max(bg_floor + 2.2 * mad, float(np.percentile(se_valid, 62)))
        m = _binary_morph((se > thr) & allowed, dilate_iters=4, erode_iters=1)
        lbl, comps = _connected_components(m)
        best_pts = None
        best_score = -1e9
        for _cid, ys, xs in comps:
            if len(xs) < 220:
                continue
            d = float(np.hypot(xs.mean() - lx, ys.mean() - ly))
            contains = bool(0 <= ly < h and 0 <= lx < w and lbl[int(ly), int(lx)] == _cid)
            sc = (1200.0 if contains else 0.0) - d + min(len(xs), 8000) * 0.03
            if sc > best_score:
                best_score = sc
                best_pts = (ys, xs)
        if best_pts is None:
            return None

        ys, xs = best_pts
        bx0, by0 = x0 + int(xs.min()), y0 + int(ys.min())
        bx1, by1 = x0 + int(xs.max()), y0 + int(ys.max())
        pw, ph = bx1 - bx0, by1 - by0
        if not (35 <= pw <= 0.30 * W and 35 <= ph <= 0.32 * H):
            return None
        if not (0.28 <= pw / float(max(1, ph)) <= 3.5):
            return None
        pad = int(0.18 * max(pw, ph)) + 8
        return (max(can_x0, bx0 - pad), max(can_y0, by0 - pad),
                min(can_x1, bx1 + pad), min(can_y1, by1 + pad))

    def _tile_data(self, k):
        """Return cached (raw_crop, enhanced_pil, fg_mask, seam_feat) for object index k."""
        key = (id(self.panel), int(k))
        if key in self._tile_cache:
            return self._tile_cache[key]
        W, H = self.panel.size
        x, y = self.objects[k]
        box = self._piece_bbox(x, y, self_idx=k)
        if box is None:
            cx, cy = int(x * W), int(y * H)
            hh = int(min(W, H) * 0.105)
            box = (max(int(W * 0.06), cx - hh), max(int(H * 0.20), cy - hh),
                   min(int(W * 0.94), cx + hh), min(int(H * 0.86), cy + hh))
        raw_crop = self.panel.crop(box)
        enh_pil, mask, feat = enhance_block_tile(raw_crop, work_side=210)
        val = (raw_crop, enh_pil, mask, feat)
        self._tile_cache[key] = val
        return val

    def _fit_slot(self, im, w, h):
        k = min(w / float(max(1, im.size[0])), h / float(max(1, im.size[1])))
        nw, nh = max(1, int(im.size[0] * k)), max(1, int(im.size[1] * k))
        return im.resize((nw, nh), Image.LANCZOS)

    def sheet(self, ids=None):
        if not self.objects:
            return {"ok": False, "error": "还没有 locate 结果，先调用 locate"}
        if self.panel is None:
            return {"ok": False, "error": "没有面板图"}
        idx = [int(i) - 1 for i in (ids or range(1, len(self.objects) + 1))
               if 0 < int(i) <= len(self.objects)]
        if not idx:
            idx = list(range(len(self.objects)))
        big = len(idx) <= 2
        card_w, card_h = (520, 270) if big else (460, 240)
        thumbs = []
        for k in idx:
            thumbs.append((k + 1, self._tile_image(k, size=card_w, height=card_h, dual=True)))
        cols = 2 if len(thumbs) > 1 else 1
        rows = (len(thumbs) + cols - 1) // cols
        head = 24
        sh = Image.new("RGB", (cols * (card_w + 8) + 8,
                               rows * (card_h + head + 8) + 8), (22, 22, 26))
        d = ImageDraw.Draw(sh)
        for n, (num, t) in enumerate(thumbs):
            r, c = divmod(n, cols)
            x0, y0 = 8 + c * (card_w + 8), 8 + r * (card_h + head + 8)
            d.text((x0 + 4, y0 + 4), "#%d  (LEFT=RAW | RIGHT=CLEAN+GRID)" % num, fill=(255, 215, 60))
            sh.paste(t, (x0, y0 + head))
            d.rectangle([x0, y0, x0 + card_w - 1, y0 + card_h + head - 1],
                        outline=(70, 70, 80), width=2)
        p = os.path.join(DEBUG_DIR, "tinysheet_%s.jpg"
                         % time.strftime("%H%M%S"))
        sh.save(p, quality=90)
        self.log("        sheet -> %s（%d 格%s）"
                 % (os.path.basename(p), len(thumbs), "，并排放大" if big else ""))
        out = {"ok": True, "count": len(thumbs), "image": _jpeg(sh),
               "note": "对照表已附在下一轮（左=原图，右=去环去迷彩+网格线增强图），编号与 locate 的 id 对应"}
        if len(thumbs) <= 6:
            out["tiles"] = [(num, _jpeg(t)) for num, t in thumbs]
            out["note"] = ("每块的「原图 | 去环增强」双视图已分别附上（图③起，按 #编号 标注），"
                           "重点数右侧增强图里的白色小方块数量与黑线网格再判断。")
        return out

    def _tile_image(self, k, size=440, height=None, dual=True):
        """按 locate 编号裁出单块积木：默认输出「原图 | 去环网格增强」双视图。"""
        raw_crop, enh_pil, _mask, _feat = self._tile_data(k)
        if not dual:
            t = self._fit_slot(raw_crop, size, height or size)
            canvas = Image.new("RGB", (size, height or size), (28, 28, 32))
            canvas.paste(t, ((canvas.size[0] - t.size[0]) // 2,
                             (canvas.size[1] - t.size[1]) // 2))
            return canvas
        h_out = height or max(220, int(size * 0.52))
        half_w = (size - 12) // 2
        slot_h = h_out - 8
        canvas = Image.new("RGB", (size, h_out), (28, 28, 32))
        t_raw = self._fit_slot(raw_crop, half_w, slot_h)
        t_enh = self._fit_slot(enh_pil, half_w, slot_h)
        canvas.paste(t_raw, (4 + (half_w - t_raw.size[0]) // 2,
                             4 + (slot_h - t_raw.size[1]) // 2))
        canvas.paste(t_enh, (8 + half_w + (half_w - t_enh.size[0]) // 2,
                             4 + (slot_h - t_enh.size[1]) // 2))
        return canvas

    def _pair_confirm_sheet(self, ia, ib):
        """Build a 2x2 comparison image:
          Row 1: Original A | Original B
          Row 2: Denoised+Seam A | Denoised+Seam B rotated by best_angle onto A
        """
        rawA, enhA, maskA, featA = self._tile_data(ia)
        rawB, enhB, maskB, featB = self._tile_data(ib)
        al = _best_rotation_align(maskA, featA, maskB, featB)
        ang = int(al.get("angle") or 0)
        enhB_rot = enhB.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=(128, 128, 128)) if ang else enhB

        cell_w, cell_h = 360, 260
        head = 24
        sheet = Image.new("RGB", (cell_w * 2 + 24, (cell_h + head) * 2 + 24), (22, 22, 26))
        d = ImageDraw.Draw(sheet)

        slots = [
            (0, 0, "A = #%d RAW" % (ia + 1), rawA, (255, 215, 60)),
            (1, 0, "B = #%d RAW" % (ib + 1), rawB, (255, 215, 60)),
            (0, 1, "A CLEAN+GRID", enhA, (120, 240, 255)),
            (1, 1, "B CLEAN+GRID (ROT %ddeg)" % ang, enhB_rot, (120, 240, 255)),
        ]
        for col, row, title, im_src, col_txt in slots:
            ox = 8 + col * (cell_w + 8)
            oy = 8 + row * (cell_h + head + 8)
            d.text((ox + 4, oy + 4), title, fill=col_txt)
            fitted = self._fit_slot(im_src, cell_w - 8, cell_h - 8)
            sheet.paste(fitted, (ox + (cell_w - fitted.size[0]) // 2,
                                 oy + head + (cell_h - fitted.size[1]) // 2))
            d.rectangle([ox, oy, ox + cell_w - 1, oy + cell_h + head - 1],
                        outline=(70, 75, 85), width=2)
        return sheet, al

    def _ask_pair_confirm(self, ia, ib):
        """Shared binary same-shape confirmation call for verify_pair and _confirm_pair."""
        sheet, al = self._pair_confirm_sheet(ia, ib)
        prompt = ("上图上下两行展示同一对候选图形 A（左）与 B（右）：\n"
                  "第一行是原图；第二行已去除背景同心圆噪点并锐化黑色分割线，且右侧 B 已自动旋转 %d° 对齐到 A 的朝向。\n"
                  "它们是同一个立体积木吗？请严格核对第二行白色小方块数量与黑线网格轮廓：\n"
                  "1) 先数清 A 和 B 各自包含多少个小方块（数量不同必为 false）；\n"
                  "2) 再核对主体长度、弯折拐角与凸出小块的相对位置（注意三维物体可能绕轴翻面呈左右镜像，但格数与连接拓扑必须一致）。\n"
                  '只输出 JSON：{"same":true,"why":"15字内"} 或 {"same":false,"why":"15字内"}'
                  % int(al.get("angle") or 0))
        plan, meta = CA.ask_vlm(_jpeg(sheet), hint=prompt, raw_prompt=True,
                                effort=os.environ.get("VLM_VERIFY_EFFORT", "low"),
                                chain=CA.VLM_SUB)
        same = _as_bool((plan or {}).get("same")) if isinstance(plan, dict) else None
        return same, (plan or {}).get("why", "") if isinstance(plan, dict) else "", meta, al

    def verify_pair(self, a, b):
        """把两块并排放大，问一个**二分问题**：是不是同一个图形。

        开放式的"从 5 块里挑相同的两块"很容易糊；强制二选一（是/否）要稳得多。
        子调用输出极短，延迟 3~6 秒，适合拿不准时确认。
        """
        if self.panel is None or not self.objects:
            return {"ok": False, "error": "缺面板或候选"}
        try:
            ia, ib = int(a) - 1, int(b) - 1
        except Exception:
            return {"ok": False, "error": "参数要是编号 a,b"}
        if not (0 <= ia < len(self.objects) and 0 <= ib < len(self.objects)) or ia == ib:
            return {"ok": False, "error": "编号越界或相同"}
        same, why, meta, al = self._ask_pair_confirm(ia, ib)
        self.log("        verify #%d vs #%d -> same=%s cv=%.2f@%d° (%ss)"
                 % (ia + 1, ib + 1, same, al.get("score", 0), al.get("angle", 0),
                    meta.get("elapsed")))
        return {"ok": same is not None, "same": same,
                "why": why, "elapsed": meta.get("elapsed"),
                "cv_score": round(float(al.get("score", 0)), 3)}

    def _confirm_pair(self, a, b):
        """二次确认：把候选两块做成原图+去环旋转对齐 2x2 对照图，只回答是/否。"""
        ia, ib = int(a) - 1, int(b) - 1
        same, _why, meta, al = self._ask_pair_confirm(ia, ib)
        self.log("        confirm #%d vs #%d -> same=%s cv=%.2f@%d° (%ss)"
                 % (ia + 1, ib + 1, same, al.get("score", 0), al.get("angle", 0),
                    meta.get("elapsed")))
        return same, meta

    def _uniform_cv_feats(self, idx):
        """Extract uniform-panel-scale masks & seam features centered by foreground centroid."""
        W, H = self.panel.size
        scale = 1150.0 / float(max(1, W, H))
        half_win = max(28, int(round(max(W, H) * 0.13)))
        raw_layers = {}
        for k in idx:
            if not (0 <= k < len(self.objects)):
                continue
            ox, oy = self.objects[k]
            cx, cy = int(ox * W), int(oy * H)
            x0, y0 = max(0, cx - half_win), max(0, cy - half_win)
            x1, y1 = min(W, cx + half_win), min(H, cy + half_win)
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            crop = self.panel.crop((x0, y0, x1, y1))
            rw = max(24, int(round(crop.size[0] * scale)))
            rh = max(24, int(round(crop.size[1] * scale)))
            arr = np.asarray(crop.convert("RGB").resize((rw, rh), Image.BICUBIC), dtype=np.float32)
            bf = _joint_bilateral_rgb(arr, sigma_s=2.4, sigma_r=14.0)
            gray = 0.299 * bf[:, :, 0] + 0.587 * bf[:, :, 1] + 0.114 * bf[:, :, 2]
            bg_l = _gauss_blur_float(gray, 24.0)
            bg_rgb = _gauss_blur_rgb(bf, 24.0)
            col_d = np.sqrt(np.sum((bf - bg_rgb) ** 2, axis=2))
            lum_d = np.abs(gray - bg_l)
            fg = np.minimum(lum_d, 30.0) * 0.9 + np.minimum(np.maximum(0.0, col_d - 6.0), 30.0) * 1.1
            fg_s = _gauss_blur_float(fg, 1.3)

            lx, ly = int((cx - x0) * scale), int((cy - y0) * scale)
            yy, xx = np.ogrid[:rh, :rw]
            d_self = (xx - lx) ** 2 + (yy - ly) ** 2
            allowed = np.ones((rh, rw), dtype=bool)
            for oi, (px, py) in enumerate(self.objects):
                if oi == k:
                    continue
                qx, qy = int((px * W - x0) * scale), int((py * H - y0) * scale)
                if (qx - lx) ** 2 + (qy - ly) ** 2 < 144:
                    continue
                allowed &= (d_self <= ((xx - qx) ** 2 + (yy - qy) ** 2) * 1.05)
            vals = fg_s[allowed]
            if vals.size == 0:
                continue
            bg_vals = vals[vals <= np.percentile(vals, 55)]
            med = float(np.median(bg_vals)) if bg_vals.size else float(np.median(vals))
            mad = float(np.median(np.abs(bg_vals - med))) + 1e-4 if bg_vals.size else 1.0
            peak = float(np.percentile(vals, 94))
            thr = max(med + 3.2 * mad, med + 0.34 * (peak - med))
            m = _binary_morph((fg_s > thr) & allowed, dilate_iters=0, erode_iters=1)
            m = _binary_morph(m, dilate_iters=1, erode_iters=0)
            lbl, comps = _connected_components(m)
            best_mask = np.zeros_like(m)
            best_sc = -1e9
            for cid, ys, xs in comps:
                if len(xs) < 60:
                    continue
                d = float(np.hypot(xs.mean() - lx, ys.mean() - ly))
                contains = bool(0 <= ly < rh and 0 <= lx < rw and lbl[ly, lx] == cid)
                sc = (5000.0 if contains else 0.0) + len(xs) - d * 28.0
                if sc > best_sc:
                    best_sc = sc
                    best_mask = (lbl == cid)
            closed = _grey_closing_2d(gray, ksize=7)
            seams = np.maximum(0.0, closed - gray)
            grad = _sobel_mag(gray)
            roi = _binary_morph(best_mask, dilate_iters=2, erode_iters=0)
            feat = (seams * 1.4 + grad * 0.35) * roi
            mx = float(feat.max())
            if mx > 1e-6:
                feat = feat / mx
            raw_layers[k] = (best_mask.astype(np.float32), feat.astype(np.float32))
        if len(raw_layers) < 2:
            return None

        canvas = max(120, int(round(max(max(v[0].shape) for v in raw_layers.values()) * 1.1)))
        centered = {}
        for k, (m, f) in raw_layers.items():
            ys, xs = np.where(m > 0.35)
            if len(xs) < 10:
                continue
            cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
            h, w = m.shape
            half = canvas // 2
            om = np.zeros((canvas, canvas), dtype=np.float32)
            of = np.zeros((canvas, canvas), dtype=np.float32)
            sx0, sx1 = max(0, cx - half), min(w, cx + half)
            sy0, sy1 = max(0, cy - half), min(h, cy + half)
            dx0, dy0 = sx0 - (cx - half), sy0 - (cy - half)
            om[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = m[sy0:sy1, sx0:sx1]
            of[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = f[sy0:sy1, sx0:sx1]
            cm = np.asarray(Image.fromarray((om * 255).astype(np.uint8), mode="L").resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
            cf = np.asarray(Image.fromarray((of * 255).astype(np.uint8), mode="L").resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
            centered[k] = (cm, cf)
        return centered if len(centered) >= 2 else None

    def _rank_pairs_cv(self, idx):
        """Compute rotation-aligned CV shape similarity for all pairs in idx."""
        centered = self._uniform_cv_feats(idx) if self.panel is not None else None
        if centered is not None and all(k in centered for k in idx):
            scores = {}
            for pos_i in range(len(idx)):
                ka = idx[pos_i]
                bmA = centered[ka][0] > 0.35
                bfA = _gauss_blur_float(centered[ka][1], 1.1)
                bfA_n = float(np.linalg.norm(bfA)) + 1e-8
                for pos_j in range(pos_i + 1, len(idx)):
                    kb = idx[pos_j]
                    im_mB = Image.fromarray((centered[kb][0] * 255).astype(np.uint8), mode="L")
                    im_fB = Image.fromarray((centered[kb][1] * 255).astype(np.uint8), mode="L")
                    top_angs = []
                    for ang in range(0, 360, 5):
                        rm = np.asarray(im_mB.rotate(ang, resample=Image.BILINEAR)) > 90
                        iou = float(np.sum(bmA & rm)) / (float(np.sum(bmA | rm)) + 1e-8)
                        top_angs.append((iou, ang))
                    top_angs.sort(reverse=True)
                    best_sc = -1.0
                    for _iou0, ang in top_angs[:10]:
                        rm0 = np.asarray(im_mB.rotate(ang, resample=Image.BILINEAR)) > 90
                        rf0 = _gauss_blur_float(
                            np.asarray(im_fB.rotate(ang, resample=Image.BILINEAR), dtype=np.float32) / 255.0, 1.1
                        )
                        for dy in (-2, 0, 2):
                            for dx in (-2, 0, 2):
                                rm = np.roll(np.roll(rm0, dy, axis=0), dx, axis=1)
                                rf = np.roll(np.roll(rf0, dy, axis=0), dx, axis=1)
                                iou = float(np.sum(bmA & rm)) / (float(np.sum(bmA | rm)) + 1e-8)
                                ncc = float(np.sum(bfA * rf)) / (bfA_n * (float(np.linalg.norm(rf)) + 1e-8))
                                sc = 0.60 * iou + 0.40 * ncc
                                if sc > best_sc:
                                    best_sc = sc
                    scores[(ka + 1, kb + 1)] = float(best_sc)
            return scores

        feats = {k: self._tile_data(k) for k in idx}
        scores = {}
        for pos_i in range(len(idx)):
            for pos_j in range(pos_i + 1, len(idx)):
                ka, kb = idx[pos_i], idx[pos_j]
                al = _best_rotation_align(feats[ka][2], feats[ka][3],
                                          feats[kb][2], feats[kb][3])
                scores[(ka + 1, kb + 1)] = float(al.get("score", 0.0))
        return scores

    def signatures(self, ids=None):
        """把所有候选拼成一张高清对照表（每格 左=原图 | 右=去环网格），一次上传给 VLM 输出结构化签名。"""
        if not self.objects or self.panel is None:
            return {"ok": False, "error": "还没有 locate 结果"}
        idx = [int(i) - 1 for i in (ids or range(1, len(self.objects) + 1))
               if 0 < int(i) <= len(self.objects)][:6]
        if not idx:
            idx = list(range(min(6, len(self.objects))))
        nums = [k + 1 for k in idx]
        # 合成单张高质量 2 列网格大图（相比传 5 张独立图：少 4 次上传 PoW，耗时从 >45s 降到 ~12s）
        card_w, card_h, head = 480, 250, 24
        cols = 2 if len(idx) > 1 else 1
        rows = (len(idx) + cols - 1) // cols
        comp = Image.new("RGB", (cols * (card_w + 8) + 8, rows * (card_h + head + 8) + 8), (22, 22, 26))
        draw = ImageDraw.Draw(comp)
        for pos, k in enumerate(idx):
            r, c = divmod(pos, cols)
            ox, oy = 8 + c * (card_w + 8), 8 + r * (card_h + head + 8)
            draw.text((ox + 4, oy + 4), "#%d  (LEFT=RAW | RIGHT=CLEAN+GRID)" % (k + 1), fill=(255, 215, 60))
            comp.paste(self._tile_image(k, size=card_w, height=card_h, dual=True), (ox, oy + head))
            draw.rectangle([ox, oy, ox + card_w - 1, oy + card_h + head - 1], outline=(70, 75, 85), width=2)
        prompt = ("这是一张编号图形拼盘，包含同一道图形题里的 %d 个候选图形（编号 %s）。\n"
                  "每格分为左右两半：左侧是原图，右侧是去除同心圆迷彩干扰后的高对比度网格图。\n"
                  "请重点数每格右侧增强图里的白色小方块与黑线分割线，逐个给出**结构化签名**（只描述，不选答案）：\n"
                  "- n：由多少个小方块组成（整数，务必数清右侧网格）\n"
                  "- box：外框占几格宽、几格高，写成 [宽, 高]\n"
                  "- pro：凸出或缺口，用「方位+格数」表示，方位取 N(上)/E(右)/S(下)/W(左)，"
                  "多个用逗号分隔，例如 \"N1,E2\"；没有明显凸出就写 \"\"\n"
                  "只输出 JSON：{\"sig\":[{\"id\":%d,\"n\":7,\"box\":[3,3],\"pro\":\"N1\"}]}\n"
                  "id 必须用格子左上角的 #编号，每块都要有。"
                  % (len(nums), "、".join("#%d" % n for n in nums), nums[0]))
        plan, meta = _ask_multi([_jpeg(comp, q=90)], prompt, timeout=60, chain=CA.VLM_SUB)
        if plan is None:
            self.log("        [!] signatures VLM 失败 (%ss): %s"
                     % (meta.get("elapsed"), meta.get("error") or "无有效JSON"))
        sigs = []
        arr = (plan or {}).get("sig") or (plan or {}).get("signatures") or []
        for it in arr if isinstance(arr, list) else []:
            s = _norm_sig(it)
            if s:
                sigs.append(s)
        got = set(s["id"] for s in sigs)
        for n in nums:                      # 模型漏掉的块标成 n=None，不参与比对
            if n not in got:
                sigs.append({"id": n, "n": None, "box": [], "pro": []})
        sigs.sort(key=lambda s: s["id"])
        self.log("        signature %d 块 (%ss) %s"
                 % (len(sigs), meta.get("elapsed"),
                    "; ".join("#%d n=%s box=%s pro=%s"
                              % (s["id"], s["n"],
                                 "x".join(str(v) for v in s["box"]) or "?",
                                 ",".join(s["pro"]) or "-") for s in sigs)))
        return {"ok": bool(sigs), "elapsed": meta.get("elapsed"),
                "signatures": [{"id": s["id"], "n": s["n"], "box": s["box"],
                                "pro": s["pro"]} for s in sigs]}

    def solve_pair(self, ids=None, confirm_limit=2):
        """配对题一步流：去环增强签名 → CV旋转对齐辅助排序 → 2x2对齐二次确认 → 提交。"""
        t0 = time.time()
        if not self.objects or self.panel is None:
            return {"ok": False, "error": "还没有 locate 结果，先调用 locate"}
        idx = [int(i) - 1 for i in (ids or range(1, len(self.objects) + 1))
               if 0 < int(i) <= len(self.objects)][:6]
        if not idx:
            idx = list(range(min(6, len(self.objects))))
        sg = self.signatures(ids)
        sigs = sg.get("signatures") if sg.get("ok") else []
        cands = _pair_candidates(sigs)
        cv_scores = self._rank_pairs_cv(idx) if len(idx) >= 2 else {}
        for c in cands:
            c["cv"] = cv_scores.get((min(c["a"], c["b"]), max(c["a"], c["b"])), 0.0)
        cands.sort(key=lambda c: (not c["strict"], -c.get("cv", 0.0), c["a"], c["b"]))

        # 如果结构化签名过于分散导致 cands 为空，则按相同块数 n + CV 旋转相似度补出候选
        if not cands and cv_scores:
            n_map = {s["id"]: s.get("n") for s in sigs if isinstance(s, dict)}
            extra = []
            for (pa, pb), sc in cv_scores.items():
                na, nb = n_map.get(pa), n_map.get(pb)
                same_n = bool(isinstance(na, int) and isinstance(nb, int) and na == nb)
                if isinstance(na, int) and isinstance(nb, int) and na != nb:
                    continue
                extra.append({"a": pa, "b": pb, "strict": False, "cv": sc, "same_n": same_n})
            if not extra:
                extra = [{"a": pa, "b": pb, "strict": False, "cv": sc, "same_n": False}
                         for (pa, pb), sc in cv_scores.items()]
            extra.sort(key=lambda c: (not c.get("same_n"), -c.get("cv", 0.0), c["a"], c["b"]))
            cands = extra[:3]

        self.log("        candidates -> %s"
                 % ("; ".join("#%d+#%d%s(%.2f)" % (c["a"], c["b"],
                                                   "" if c["strict"] else "*",
                                                   c.get("cv", 0.0))
                              for c in cands) or "无"))
        if not cands:
            return {"ok": False, "stage": "compare", "signatures": sigs,
                    "error": "签名与CV都没有形成候选对"}
        chosen, tried = None, []
        eff_limit = max(int(confirm_limit), 3) if (cands and not cands[0].get("strict")) else max(1, int(confirm_limit))
        for c in cands[:eff_limit]:
            same, meta = self._confirm_pair(c["a"], c["b"])
            tried.append({"pair": [c["a"], c["b"]], "same": same,
                          "elapsed": meta.get("elapsed"), "cv": round(c.get("cv", 0.0), 3)})
            if same:
                chosen = c
                break
        if chosen is None:
            return {"ok": False, "stage": "confirm", "signatures": sigs,
                    "candidates": cands, "tried": tried,
                    "error": "候选对都没通过二次确认"}
        # 二次确认已过 → verify=False，省掉一次 7~15s 的重复判断
        r = self.commit_pair(chosen["a"], chosen["b"], verify=False)
        r.update({"stage": "commit", "signatures": sigs,
                  "candidates": cands, "chosen": [chosen["a"], chosen["b"]],
                  "tried": tried, "elapsed": round(time.time() - t0, 1)})
        self.log("        solve_pair -> pair=#%d+#%d passed=%s (%ss)"
                 % (chosen["a"], chosen["b"], r.get("passed"), r.get("elapsed")))
        return r

    _GRID_3X3_CENTERS = {
        1: (0.18, 0.32), 2: (0.50, 0.32), 3: (0.82, 0.32),
        4: (0.18, 0.53), 5: (0.50, 0.53), 6: (0.82, 0.53),
        7: (0.18, 0.75), 8: (0.50, 0.75), 9: (0.82, 0.75),
    }

    def commit(self, ids=None, points=None, drag=None, cells=None, verify=False,
               submit_after=True):
        """**所有题型的统一原子动作**：点选（或拖拽）→ 点底部按钮 → 等页面回应。

        以前是「verify_pair / click / click / submit」多次往返，每轮都要夹一次
        8~15s 的模型调用；合并后一页只剩「动作 + 等 1~3 秒」。

        支持三种目标描述，可混用：
          ids    : locate 的编号列表，如 [1,4]（最准，坐标就是 locate 给的）
          points : 直接给 panel 归一化坐标，如 [[0.3,0.4],[0.7,0.5]]（网格题用）
          drag   : 一次拖拽 [x1,y1,x2,y2]（补图题用）
        verify=true 且恰好给两个 ids 时，先做一次二选一确认，判不同就不点。
        注意：verify 默认 **关闭** —— 视觉模型偶尔假阴性，一判 false 就整页白丢；
        拿不准时由模型自己显式打开。
        """
        if self.panel is None:
            return {"ok": False, "error": "当前没有面板图"}
        ids = list(ids) if isinstance(ids, (list, tuple)) else ([ids] if ids else [])
        targets, used_ids, used_cells = [], [], []
        for it in ids:
            try:
                k = int(it) - 1
            except Exception:
                continue
            if 0 <= k < len(self.objects):
                targets.append(tuple(self.objects[k]))
                used_ids.append(k + 1)
        cell_list = list(cells) if isinstance(cells, (list, tuple)) else ([cells] if cells else [])
        for c in cell_list:
            try:
                ci = int(c)
            except Exception:
                continue
            if ci in self._GRID_3X3_CENTERS and ci not in used_cells:
                targets.append(self._GRID_3X3_CENTERS[ci])
                used_cells.append(ci)
        for pt in (points or []):
            try:
                x, y = float(pt[0]), float(pt[1])
            except Exception:
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                targets.append((x, y))
        box = None
        if drag:
            try:
                box = [float(v) for v in drag[:4]]
            except Exception:
                box = None
        if not targets and not box:
            return {"ok": False, "error": "没有可执行的目标（给 cells / ids / points / drag）"}

        out = {"ok": True, "stage": "act", "ids": used_ids, "cells": used_cells,
               "targets": [[round(x, 3), round(y, 3)] for x, y in targets],
               "drag": box}

        # 恰好两个编号 + verify → 先二选一（配对题的防错闸）
        if verify and box is None and len(used_ids) == 2:
            v = self.verify_pair(used_ids[0], used_ids[1])
            out["verify"] = {"same": v.get("same"), "why": v.get("why"),
                             "elapsed": v.get("elapsed")}
            if v.get("same") is False:
                out.update({"ok": False, "stage": "verify",
                            "error": "二选一确认判成不同，没有点击"})
                return out

        if box is not None:
            r = self.drag(box[0], box[1], box[2], box[3], "panel")
            out["drag_ok"] = bool(r.get("ok"))
            time.sleep(random.uniform(0.2, 0.4))
        clicks = []
        for x, y in targets:
            clicks.append(self.click(x, y, "panel"))
            time.sleep(random.uniform(0.25, 0.45))
        out["clicks"] = clicks
        if clicks and not all(c.get("ok") for c in clicks):
            out.update({"ok": False, "stage": "click", "error": "有点击没生效"})
            return out
        if not submit_after:
            return out

        before_grid = None
        try:
            g = self.panel.convert("L").resize((64, 96), Image.BILINEAR)
            before_grid = np.asarray(g, dtype=np.int16)[22:80, 4:60]
        except Exception:
            before_grid = None

        time.sleep(0.25)
        sub = self.submit()
        out["submit"] = {"ok": sub.get("ok"), "button": sub.get("button")}
        if not sub.get("ok"):
            out.update({"ok": False, "error": "没点到提交按钮"})
            return out

        # 提交后直接等页面给答案：通过 / 换到新题 / 仍是原题（被判错）
        # hCaptcha 翻页通常在同一 challenge iframe 内直接切换题图，panel 不会隐藏，
        # 因此同时比对题目区像素变化（避免旧版 gone 判据固定空等 ~11 秒）。
        st, gone, content_changed = {}, False, False
        time.sleep(0.8)
        for step_idx in range(10):
            if step_idx > 0:
                time.sleep(0.5)
            st = CA.page_status(self.page)
            if st.get("passed"):
                break
            if not st.get("panel"):
                gone = True
                continue
            if gone:
                break
            if before_grid is not None and self.rect and self.viewport:
                try:
                    raw_now = CA._raw_screenshot(self.page)
                    pan_now, _ = CA.crop_panel(raw_now, self.rect, self.viewport) if raw_now else (None, None)
                    if pan_now is not None and CA.panel_ready(pan_now)[0]:
                        g_now = pan_now.convert("L").resize((64, 96), Image.BILINEAR)
                        arr_now = np.asarray(g_now, dtype=np.int16)[22:80, 4:60]
                        if float(np.abs(arr_now - before_grid).mean()) > 10.0:
                            content_changed = True
                            break
                except Exception:
                    pass
        out["passed"] = bool(st.get("passed"))
        out["new_page"] = bool((gone or content_changed) and st.get("panel"))
        out["status"] = {k: st.get(k) for k in
                         ("passed", "has_token", "panel", "btn", "countdown")}
        if not out["passed"]:
            out["ok"] = False
            out["stage"] = "submit"
            if not out["new_page"]:
                out["error"] = "提交后仍是原题（多半是这组目标没被判对）"
        self.log("        commit ids=%s points=%d drag=%s -> passed=%s new_page=%s"
                 % (used_ids, len(targets) - len(used_ids), box,
                    out["passed"], out["new_page"]))
        return out

    def commit_pair(self, a, b, verify=True):
        """兼容包装：两个编号的原子提交（solve_pair / 旧提示词都会用到）。"""
        return self.commit(ids=[a, b], verify=verify)

    def submit(self):
        """精确点击底部「跳过/下一个/检查」按钮。

        模型肉眼估按钮位置经常差 3~10px（实测点 0.86 完全落在按钮外），
        点了等于没点，面板就一直闲置到超时 —— 是一次尝试里最隐蔽的浪费。
        这里直接用颜色检测出的按钮矩形中心。
        """
        if self.panel is None:
            return {"ok": False, "error": "当前没有面板图"}
        box = CA.find_button(self.panel)
        if not box:
            return {"ok": False, "error": "没检测到按钮"}
        self.log("        submit -> 按钮 #%s @ (%.3f,%.3f)"
                 % ("?", box["x"], box["y"]))
        r = self.click(box["x"], box["y"], "panel")
        r["button"] = box
        return r

    def retrigger_send(self):
        """点一次「发送验证码」，让 hCaptcha 重新出题（失败的挑战不会自己重开）。"""
        for txt in ("发送验证码", "获取验证码", "重新获取", "Send code", "Send"):
            try:
                el = self.page.ele("text=%s" % txt, timeout=1)
            except Exception:
                el = None
            if el:
                try:
                    el.click()
                    self.log("        retrigger -> 点了「%s」" % txt)
                    return True
                except Exception as e:
                    self.log("        [!] 点「%s」失败: %s" % (txt, e))
        return False

    def status(self):
        st = CA.page_status(self.page)
        self.log("        status -> token=%s panel=%s btn=%s countdown=%s"
                 % (st.get("has_token"), st.get("panel"),
                    st.get("btn") or "-", st.get("countdown")))
        return st

    def zoom(self, x, y, w, h):
        if self.panel is None:
            return {"ok": False, "error": "没有面板图"}
        W, H = self.panel.size
        x0 = max(0, int((float(x) - float(w) / 2) * W))
        y0 = max(0, int((float(y) - float(h) / 2) * H))
        x1 = min(W, int((float(x) + float(w) / 2) * W))
        y1 = min(H, int((float(y) + float(h) / 2) * H))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return {"ok": False, "error": "区域太小"}
        c = self.panel.crop((x0, y0, x1, y1))
        k = max(1.0, 900.0 / max(c.size))
        if k > 1.0:
            c = c.resize((int(c.size[0] * k), int(c.size[1] * k)), Image.LANCZOS)
        self.log("        zoom   (%.2f,%.2f) %.2fx%.2f -> %s"
                 % (x, y, w, h, c.size))
        return {"ok": True, "image": _jpeg(c),
                "note": "区域放大图已附在下一轮图③"}


class CaptchaBrain(object):
    def __init__(self, page, max_turns=10, verbose=True, should_stop=None,
                 hint="", deadline=None, task_meta=None):
        self.page = page
        self.max_turns = int(max_turns)
        self.verbose = verbose
        self.should_stop = should_stop
        self.hint = hint
        self.task_meta = dict(task_meta) if isinstance(task_meta, dict) else {}
        self.solve_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        self.last_kind = "unknown"
        self.last_prompt_text = ""
        self.tools = Tools(page, verbose=verbose)
        # 整轮硬预算（绝对时间戳）。没有它，一次 solve 只要模型肯想就能跑 500s+，
        # 外层 wait_for_captcha_or_countdown 的 timeout 完全失效（实测 573s）。
        self.deadline = float(deadline) if deadline else None
        self.transcript = []
        # 「一次尝试」的状态：定位过没有、点过目标没有、放大过几次。
        # 目的：一次尝试里只允许定位一次，之后必须落地点击。
        # 实测不加这个护栏，模型会 locate→sheet→zoom→locate 反复循环，
        # 单轮 47s、一轮 113s 都没点出一下，直接把 120s 时间窗烧光。
        self.attempt = {"located": False, "clicked": False, "zoomed": 0, "blind_blocked": 0}
        self.solved = False
        self.reason = ""
        self.final_state = None
        self.steps = []

    def log(self, m):
        if self.verbose:
            print(m)

    def _panel_sig(self, panel):
        if panel is None:
            return None
        try:
            import hashlib
            g = panel.convert("L").resize((12, 12), Image.BILINEAR)
            return hashlib.md5(g.tobytes()).hexdigest()
        except Exception:
            return None

    def _reset_attempt(self):
        self.attempt = {"located": False, "clicked": False, "zoomed": 0, "blind_blocked": 0}

    def _guard(self, name, call):
        """返回拒绝理由（str）或 None。只约束节奏，不涉及题型。"""
        a = self.attempt
        if self.tools.panel is None and name == "click" and (call.get("space") or "panel") == "viewport":
            if getattr(self, "_checkbox_clicks", 0) >= 2:
                return ("复选框已经点击过多次，禁止继续盲点复选框！请 wait(3) 等待面板展开，若仍无面板则 finish(ok=false)。")
        if name == "locate" and a["located"] and not a["clicked"]:
            return ("已经定位过了，不要再 locate。请立刻调 solve_pair 或从上一轮的 objects 里选两个点 commit(ids=[a,b])。")
        if (name == "commit"
                and not call.get("ids") and not call.get("drag") and not call.get("cells")
                and isinstance(call.get("points"), list)
                and 1 <= len(call.get("points")) <= 3
                and not a["located"] and not self.tools.objects
                and int(a.get("blind_blocked", 0)) < 1
                and getattr(self, "turn", 0) < 6
                and self.tools.panel is not None
                and not _has_3x3_grid(self.tools.panel)):
            a["blind_blocked"] = int(a.get("blind_blocked", 0)) + 1
            return ('这是非九宫格的散落图形题，严禁直接目测猜 points！请立刻先调用 '
                    '{"tool":"locate","target":"所有图形"}，系统会自动去除背景同心圆噪点并锐化方块网格。')
        # 注意：这里**不能**再硬拦 click —— 换题后状态若没复位会把模型锁死，
        # 实测出现连续 10 轮只调 submit、一个目标都没点，整段白跑。
        # 只在反馈里提示，不做拒绝。
        if name in ("sheet", "zoom") and a["zoomed"] >= 2 and not a["clicked"]:
            return ("本轮已经放大过两次了，别再放大。立刻 click 两个目标。")
        if name in ("commit", "drag"):
            drag_vals = call.get("drag") if name == "commit" else [call.get("x1"), call.get("y1"), call.get("x2"), call.get("y2")]
            if isinstance(drag_vals, (list, tuple)) and len(drag_vals) >= 4:
                try:
                    dx = float(drag_vals[2]) - float(drag_vals[0])
                    dy = float(drag_vals[3]) - float(drag_vals[1])
                    if math.hypot(dx, dy) < 0.04:
                        return ("拖拽起点与终点几乎重合（距离 < 0.04）！禁止原地拖拽：请先找唯一带「+ 移动」标签的悬浮块作为起点 (x1,y1)，再拖到背景中匹配的另一个目标字母中心 (x2,y2)。")
                except (TypeError, ValueError):
                    pass
        # 空转护栏：跑了 6 轮还一个目标都没点过，只允许动手（commit/click/drag）
        if (name in ("locate", "sheet", "zoom", "status")
                and getattr(self, "turn", 0) >= 6
                and not a["clicked"] and self.tools.objects):
            return ("已经观察够久了，禁止再观察。立刻用 commit(ids=[...]) 或 "
                    "commit(points=[[x,y],...]) 把你判断的目标点掉并提交。")
        return None

    def _note_tool(self, name, call):
        a = self.attempt
        if name == "locate":
            a["located"] = True
        elif name in ("sheet", "zoom"):
            a["zoomed"] += 1
        elif name == "click":
            try:
                y = float(call.get("y", 0.5))
            except Exception:
                y = 0.5
            if y < 0.85:
                a["clicked"] = True
            else:
                # 点了底部按钮 = 一次尝试结束，进入下一题
                self._reset_attempt()
            if self.tools.panel is None and (call.get("space") or "panel") == "viewport":
                self._checkbox_clicks = getattr(self, "_checkbox_clicks", 0) + 1

    def _abort(self):
        return bool(self.should_stop and self.should_stop())

    # ---------------------------------------------------------------- 观察
    def _observe(self):
        raw = CA._raw_screenshot(self.page)
        if not raw:
            return None
        vp = CA.viewport(self.page)
        jpeg, _size = CA._djpeg(raw)
        rect = CA.panel_rect(self.page)
        panel = None
        if rect:
            panel, _info = CA.crop_panel(raw, rect, vp)
            # 题图没渲染完就等它（否则模型看到白板，locate 得到 0 个）
            for _ in range(10):
                if panel is None:
                    break
                ok, _std = CA.panel_ready(panel)
                if ok:
                    break
                time.sleep(0.5)
                raw = CA._raw_screenshot(self.page) or raw
                rect = CA.panel_rect(self.page) or rect
                panel, _info = CA.crop_panel(raw, rect, vp)
        self.tools.rect = rect
        self.tools.panel = panel
        self.tools.viewport = vp
        return {"jpeg": jpeg, "panel": panel, "rect": rect, "vp": vp}

    # ---------------------------------------------------------------- 提问
    def _ask(self, obs, feedback, extra_images):
        content = []
        text = SYSTEM
        if self.hint:
            text += "\n\n【补充信息】%s" % self.hint
        left = int(max(0, 120 - (time.time() - getattr(self, "t_panel", time.time()))))
        text += "\n\n【剩余时间】约 %ds（到 0 秒面板自动关闭，一切作废）" % left
        if left < 40:
            text += ("\n【紧急】没时间了！立刻 commit(你最有把握的目标组)，"
                     "不要再 locate、不要 zoom、不要再分析。")
        text += "\n\n【上一轮结果】\n%s" % (feedback or "（第一轮，还没执行任何工具）")
        if self.transcript:
            text += "\n\n【更早的动作】\n" + "\n".join(self.transcript[-8:])
        content.append({"type": "text", "text": text})
        if obs["panel"] is not None:
            # 面板可见时以面板大图为主（题目细节都在里面），遇到横纹干扰题自动附带去横纹清晰图
            content.append({"type": "text", "text": "图① 验证码面板放大图（panel 坐标空间，左上边缘带 0.1~0.9 红色刻度尺）："})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"
                                          + _b64(_jpeg(_annotate_panel_ruler(obs["panel"])))}})
            has_scan, _, _ = _has_horizontal_scanlines(obs["panel"])
            if has_scan and not any(
                isinstance(item, tuple) and "去横纹" in str(item[0]) for item in (extra_images or [])
            ):
                extra_images = [
                    ("去横纹清晰图（已消除水平扫描线，panel 坐标系与图①完全一致，请对照本图看清底图 2×3 字母并匹配「+ 移动」块）",
                     _jpeg(_clean_horizontal_scanlines(obs["panel"]))),
                    *list(extra_images or []),
                ]
            n = 1
        else:
            content.append({"type": "text", "text": "图① 全视口截图（viewport 坐标空间）："})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + _b64(obs["jpeg"])}})
            n = 1
        for item in extra_images:
            label, im = item if isinstance(item, tuple) else ("", item)
            n += 1
            name = "①②③④⑤⑥⑦⑧⑨"[min(n, 9) - 1]
            content.append({"type": "text",
                            "text": "图%s %s：" % (name, label or "工具产出的图像")})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + _b64(im)}})

        # 单次提问的总预算：整条兜底链加起来也不许超过 90 秒（从拿到 VLM 槽位后开始计时）。
        return CA.call_vlm_endpoints(
            [{"role": "user", "content": content}],
            timeout=CA.VLM_TIMEOUT,
            deadline=self.deadline,
            ask_budget=float(os.environ.get("VLM_ASK_BUDGET", "90")),
            extra_parsers=(parse_dsml,),
            include_raw=True,
            return_on_parse_fail=True,
        )

    # ---------------------------------------------------------------- 主循环
    def run(self):
        t0 = time.time()
        if self.verbose:
            print("  [brain] 工具驱动模式启动")

        present = CA.wait_panel(self.page, timeout=18, verbose=self.verbose)
        if not present.get("panel") and not present.get("has_token"):
            # 上一轮被判错后面板会关闭且不会自己重开 —— 必须再点一次「发送验证码」，
            # 否则这一轮会空转（历史现象：第 2/3 轮一进来就"没有验证码"直接放弃）。
            if self.tools.retrigger_send():
                present = CA.wait_panel(self.page, timeout=40, verbose=self.verbose)
        self.t_panel = time.time()
        self.t_last_action = time.time()
        if present.get("has_token"):
            self.solved = True
            self.reason = "hCaptcha 静默放行（直接给了 token）"
            self.final_state = "solved"
            self.log("  [+] 无需答题，hCaptcha 直接放行")
            return {"ok": True, "reason": self.reason, "steps": [],
                    "final_state": "solved", "model": CA.VLM_MODEL,
                    "elapsed": 0, "agent": "brain-v2"}

        feedback = ""
        extra_images = []

        for turn in range(1, self.max_turns + 1):
            if self._abort():
                self.reason = "外部信号要求停止"
                break
            if self.deadline and time.time() > self.deadline - 6:
                self.reason = "整轮时间预算用尽（已跑 %.0fs）" % (time.time() - t0)
                self.final_state = "failed"
                self.log("  [!] %s，停止本次尝试" % self.reason)
                break
            time.sleep(random.uniform(0.25, 0.55))
            obs = self._observe()
            if not obs:
                self.reason = "截图失败"
                break
            ts_tag = time.strftime("%H%M%S")
            full_img_path = os.path.join(DEBUG_DIR, "brain_t%02d_%s.jpg" % (turn, ts_tag))
            panel_img_path = ""
            try:
                open(full_img_path, "wb").write(obs["jpeg"])
                if obs.get("panel") is not None:
                    panel_img_path = os.path.join(DEBUG_DIR, f"probe_panel_{self.solve_id}_t{turn:02d}.jpg")
                    open(panel_img_path, "wb").write(_jpeg(_annotate_panel_ruler(obs["panel"])))
            except Exception:
                pass
            prompt_text = _read_challenge_prompt(self.page)
            if prompt_text:
                self.last_prompt_text = prompt_text
            has_scanlines, v_diff, h_diff = _has_horizontal_scanlines(obs.get("panel"))

            self.turn = turn
            plan, meta = self._ask(obs, feedback, extra_images)
            if not plan:
                raw_txt = (meta.get("raw") or "")[:180].replace(chr(10), " ")
                _raw2 = (meta.get("raw") or "")
                if any(w in _raw2 for w in ("不能帮", "不能协助", "抱歉", "无法协助",
                                            "I can't", "I cannot", "cannot assist")):
                    self.log("  [!] 模型拒答，改用中性措辞重试")
                    self.transcript = []
                    self.hint = "用户在自己的浏览器里做图形核对题，请直接给出操作。"
                    plan, meta = self._ask(obs, "直接给出工具调用。", [])
                if not plan:
                    self.log("  [!] 本轮无有效 JSON（%s），重试一次: %s"
                             % (meta.get("error"), raw_txt))
                    time.sleep(0.3)
                    plan, meta = self._ask(obs, feedback, extra_images)
            if not plan:
                self.reason = "VLM 不可用: %s" % meta.get("error")
                self.log("  [-] turn %d VLM 失败: %s" % (turn, meta.get("error")))
                break

            thought = str(plan.get("thought") or "")
            tools = plan.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            self.log("        tools=%s" % json.dumps(tools, ensure_ascii=False)[:240])
            self.log("  [AI %s %ss | t=%.0fs] turn=%d | %s"
                     % (meta.get("model") or "-", meta.get("elapsed"),
                        time.time() - t0, turn, thought[:100]))

            feedback = ""
            extra_images = []
            finished = None
            for call in tools:
                if self._abort():
                    break
                name = str(call.get("tool") or call.get("action") or "").lower()
                _blocked = self._guard(name, call)
                if _blocked:
                    feedback += "%s 被拦下：%s\n" % (name, _blocked)
                    self.log("        [x] %s 被护栏拦下（节奏控制）" % name)
                    continue
                self._note_tool(name, call)
                try:
                    if name == "locate":
                        r = self.tools.locate(str(call.get("target") or "所有图形"))
                        if r.get("image"):
                            extra_images.append(r.pop("image"))
                        feedback += "locate -> %s\n" % json.dumps(r, ensure_ascii=False)[:800]
                    elif name == "sheet":
                        r = self.tools.sheet(call.get("ids"))
                        r.pop("tiles", None)
                        if r.get("image"):
                            extra_images.append(("图形对照表（含 #编号：左=原图 | 右=去环网格）", r.pop("image")))
                        feedback += "sheet -> %s\n" % json.dumps(r, ensure_ascii=False)[:300]
                    elif name == "zoom":
                        r = self.tools.zoom(call.get("x", 0.5), call.get("y", 0.5),
                                            call.get("w", 0.4), call.get("h", 0.3))
                        if r.get("image"):
                            extra_images.append(r.pop("image"))
                        feedback += "zoom -> %s\n" % json.dumps(r, ensure_ascii=False)[:200]
                    elif name == "click":
                        r = self.tools.click(call.get("x", 0.5), call.get("y", 0.5),
                                             call.get("space", "panel"))
                        feedback += "click -> %s\n" % json.dumps(r, ensure_ascii=False)
                        time.sleep(random.uniform(0.18, 0.40))
                    elif name == "verify_pair":
                        r = self.tools.verify_pair(call.get("a"), call.get("b"))
                        feedback += "verify_pair -> %s\n" % json.dumps(r, ensure_ascii=False)[:200]
                    elif name in ("commit", "commit_pair"):
                        if name == "commit_pair" or (
                                call.get("a") is not None and call.get("b") is not None
                                and not call.get("ids")):
                            r = self.tools.commit_pair(call.get("a"), call.get("b"),
                                                       bool(call.get("verify", True)))
                        else:
                            r = self.tools.commit(
                                ids=call.get("ids"),
                                points=call.get("points"),
                                drag=call.get("drag"),
                                cells=call.get("cells"),
                                verify=bool(call.get("verify", True)),
                                submit_after=bool(call.get("submit_after", True)))
                        feedback += "commit -> %s\n" % json.dumps(r, ensure_ascii=False)[:600]
                        self._reset_attempt()
                        time.sleep(random.uniform(0.3, 0.6))
                    elif name == "solve_pair":
                        r = self.tools.solve_pair(call.get("ids"))
                        brief = {k: r.get(k) for k in
                                 ("ok", "stage", "passed", "new_page", "chosen",
                                  "candidates", "tried", "error")}
                        feedback += "solve_pair -> %s\n" % json.dumps(brief, ensure_ascii=False)[:600]
                        feedback += "  每块签名: %s\n" % json.dumps(
                            r.get("signatures"), ensure_ascii=False)[:400]
                        self._reset_attempt()
                        time.sleep(random.uniform(0.3, 0.6))
                    elif name == "submit":
                        r = self.tools.submit()
                        feedback += "submit -> %s\n" % json.dumps(r, ensure_ascii=False)[:200]
                        self.attempt = {"located": False, "clicked": False, "zoomed": 0, "blind_blocked": 0}
                        time.sleep(random.uniform(0.4, 0.8))
                    elif name == "drag":
                        r = self.tools.drag(call.get("x1", 0), call.get("y1", 0),
                                            call.get("x2", 1), call.get("y2", 0),
                                            call.get("space", "panel"))
                        feedback += "drag -> %s\n" % json.dumps(r, ensure_ascii=False)
                        time.sleep(random.uniform(0.20, 0.45))
                    elif name == "wait":
                        r = self.tools.wait(call.get("seconds", 1.5))
                        feedback += "wait -> %s\n" % json.dumps(r, ensure_ascii=False)
                    elif name == "status":
                        r = self.tools.status()
                        feedback += "status -> %s\n" % json.dumps(r, ensure_ascii=False)
                    elif name == "finish":
                        finished = call
                        break
                    else:
                        feedback += ("未知工具 %s（可用：locate/sheet/zoom/click/drag/"
                                     "commit/solve_pair/submit/wait/status/finish）\n" % name)
                except Exception as e:
                    feedback += "%s 执行异常: %s\n" % (name, e)

            # 换页检测：hCaptcha 一换题，旧坐标和“已选中”状态全部失效。
            # 不复位会导致反馈里一直说“已选中，直接提交”，模型就一个都不点。
            _sig = self._panel_sig(obs.get("panel"))
            if _sig and _sig != getattr(self, "_last_sig", None):
                _clicked_target = any(
                    (c.get("tool") or "").lower() == "click"
                    and float(c.get("y", 0) or 0) < 0.85 for c in tools)
                if not _clicked_target:
                    self._reset_attempt()
                self._last_sig = _sig

            self.transcript.append("turn%d: %s | tools=%s"
                                   % (turn, thought[:60],
                                      json.dumps(tools, ensure_ascii=False)[:180]))
            self.steps.append({"turn": turn, "thought": thought,
                               "tools": tools, "elapsed": meta.get("elapsed")})
            turn_kind = _classify_challenge_kind(self.last_prompt_text, thought, tools, has_scanlines)
            if turn_kind != "unknown":
                self.last_kind = turn_kind
            record_captcha_probe({
                "event": "turn",
                "solve_id": self.solve_id,
                "task_id": self.task_meta.get("task_id", ""),
                "email": self.task_meta.get("email", ""),
                "scenario": self.task_meta.get("scenario", ""),
                "worker_id": self.task_meta.get("worker_id", ""),
                "turn": turn,
                "challenge_kind": turn_kind,
                "prompt_text": self.last_prompt_text,
                "has_scanlines": has_scanlines,
                "scanline_v_diff": v_diff,
                "scanline_h_diff": h_diff,
                "tools": tools,
                "thought": thought[:240],
                "feedback": feedback[:240],
                "vlm_elapsed_s": meta.get("elapsed"),
                "panel_img": panel_img_path,
                "full_img": full_img_path,
            })

            # ---- 硬判据：以 hCaptcha token 为准，不看"面板是否消失" ----
            st = CA.page_status(self.page)
            if st.get("passed"):
                self.solved = True
                self.final_state = "solved"
                self.reason = ("hCaptcha 已发放 token" if st.get("has_token")
                               else "发码倒计时已生效（验证码已通过）")
                self.log("  [+] %s —— 通过" % self.reason)
                break
            if st.get("panel"):
                self.seen_panel = True
                self.gone_turns = 0
            elif getattr(self, "seen_panel", False):
                self.gone_turns = getattr(self, "gone_turns", 0) + 1
                if self.gone_turns >= 2:
                    # 失败的挑战不会自动重开 —— 主动点一次「发送验证码」拿新题
                    if getattr(self, "retriggers", 0) < 2 and self.tools.retrigger_send():
                        self.retriggers = getattr(self, "retriggers", 0) + 1
                        self.t_panel = time.time()
                        self.gone_turns = 0
                        st2 = CA.wait_panel(self.page, timeout=40, verbose=self.verbose)
                        if st2.get("panel"):
                            self.seen_panel = True
                            feedback = ("上一轮被判错（面板关闭且没有 token）。"
                                        "已重新触发验证码，这是全新题目 —— 立刻按节奏再来一遍："
                                        "locate → 选出两个相同的 → 点提交。")
                            self.log("  [*] 已重新触发验证码（第 %d 次），继续"
                                     % self.retriggers)
                            continue
                    self.final_state = "failed"
                    self.reason = "挑战面板消失但没拿到 token，本轮判定失败"
                    self.log("  [!] 面板连续 %d 轮消失且无 token —— 失败"
                             % self.gone_turns)
                    break

            if finished is not None:
                want_ok = bool(finished.get("ok"))
                st = CA.page_status(self.page)
                if (not want_ok and not st.get("panel") and turn <= 3
                        and CA.has_captcha(self.page)[0]):
                    feedback = ("验证码 iframe 还在，面板可能仍在异步加载。"
                                "请先用 wait(2) 等待，再观察一轮，不要现在就结束。")
                    self.log("  [!] 模型过早收工，已驳回（面板可能未加载完）")
                    finished = None
                    time.sleep(1.0)
                    continue
                if want_ok and not st.get("passed"):
                    # 硬判据：没有 token 一律不算通过，把真相告诉模型让它继续
                    feedback = ("finish(ok=true) 被拒绝：hCaptcha 还没有 token"
                                "（panel=%s, countdown=%s, btn=%s）。"
                                "面板消失不等于通过。请继续处理，或 finish(ok=false)。"
                                % (st.get("panel"), st.get("countdown"),
                                   st.get("btn") or "-"))
                    self.log("  [!] 模型想判通过，但没有 token，已驳回")
                    finished = None
                    time.sleep(0.8)
                    continue
                self.solved = want_ok
                self.reason = str(finished.get("note") or ("完成" if want_ok else "放弃"))
                self.final_state = "solved" if want_ok else "failed"
                self.log("  [%s] %s" % ("+" if want_ok else "!", self.reason[:120]))
                break

            if self.attempt.get("clicked") and not any(
                    (c.get("tool") or "") == "submit" for c in tools):
                feedback += ("\n【提示】这两个目标已经选中了。如果题目只需选两块，"
                             "下一轮直接 submit 即可；如需改选，先点掉已选的再重新 locate。\n")
            if not tools:
                feedback = "你这一轮没有给 tools。请给出至少一个工具调用。"
                continue

            self.t_last_action = time.time()
            time.sleep(random.uniform(0.35, 0.7))

        if not self.reason:
            self.reason = "达到最大轮数 %d" % self.max_turns

        total_elapsed = round(time.time() - t0, 1)
        record_captcha_probe({
            "event": "solve_summary",
            "solve_id": self.solve_id,
            "task_id": self.task_meta.get("task_id", ""),
            "email": self.task_meta.get("email", ""),
            "scenario": self.task_meta.get("scenario", ""),
            "worker_id": self.task_meta.get("worker_id", ""),
            "challenge_kind": self.last_kind,
            "prompt_text": self.last_prompt_text,
            "turns": len(self.steps),
            "final_ok": bool(self.solved),
            "final_state": self.final_state,
            "final_reason": self.reason,
            "total_elapsed_s": total_elapsed,
            "model": CA.VLM_MODEL,
        })

        return {"ok": self.solved, "reason": self.reason,
                "steps": self.steps, "final_state": self.final_state,
                "model": CA.VLM_MODEL, "elapsed": total_elapsed,
                "agent": "brain-v2"}


def solve(page, max_turns=10, verbose=True, should_stop=None, hint="",
          deadline=None, task_meta=None):
    return CaptchaBrain(page, max_turns=max_turns, verbose=verbose,
                        should_stop=should_stop, hint=hint,
                        deadline=deadline, task_meta=task_meta).run()

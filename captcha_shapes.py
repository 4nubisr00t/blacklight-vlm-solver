# -*- coding: utf-8 -*-
"""captcha_shapes.py — 「点击两个相同的图形」专用求解器（两段式）

为什么单独做
--------------------------------------------------------------------------------
实测（probe_A_full / probe_B_crop 两张标注图）：
  · VLM 在**全视口**上定位图形中心是准的（两次点击都落在积木几何中心）；
  · 但让它"边定位边判断哪两块同形"时，判断很不稳定：
    同一张图两次给出不同的配对，还出现过坐标飘到空白背景上。
  · 裁剪放大后再问，反而出现系统性 y 偏移（点击落在图形下方 ~140px）。

结论：难的不是"看清"，是"在一堆 3D 积木里比对同形"。人解题时也是先把每块
单独拎出来并排看 —— 那就把这件难事拆成两个各自简单的子任务：

  第 1 段（定位）：问 VLM「图里所有独立图形的中心点」→ 只要坐标，不要判断。
  第 2 段（配对）：把每个图形裁成小图、编号、拼成一张九宫格 → 问
                  「哪两个格子里的图形完全相同」→ 输出极短 {"pair":[i,j]}。

第 2 段输入是小图（~960x640）、输出是一个短 JSON，延迟远低于长推理；
而且并排同尺度的大图，正好是视觉模型最擅长的比较任务。

对外接口
--------------------------------------------------------------------------------
    from captcha_shapes import solve_pair_from_image, find_shapes, build_sheet
    res = solve_pair_from_image(pil_image, ask=ask_vlm_function, verbose=True)
    res["points"]   -> [(x, y), (x, y)]   归一化坐标（相对传入图）
    res["pair"]    -> (i, j)
    res["ok"]      -> bool
"""
import io
import json
import os
import sys
import time

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEBUG_DIR = os.environ.get("CAPTCHA_DEBUG_DIR", os.path.join(HERE, "captcha_debug"))
try:
    os.makedirs(DEBUG_DIR, exist_ok=True)
except Exception:
    DEBUG_DIR = HERE

LOCATE_PROMPT = """这是一张验证码题目面板（hCaptcha）。请只做一件事：**定位**。

把题目图片区域里**每一个独立的积木图形**找出来，给出它的中心点。
- 坐标是相对**本图**左上角的归一化值，0.0~1.0，保留三位小数。
- 只报「图形」的中心，不要报文字、按钮、背景纹理。
- 不要判断哪个相同，不要给任何动作。
- 图形数量通常 3~8 个；每个图形是由若干小方块拼成的立体块。

只输出一个 JSON，禁止任何解释：
{"shapes":[{"id":1,"x":0.000,"y":0.000},{"id":2,"x":0.000,"y":0.000}]}"""

PAIR_PROMPT = """这张图是把一道验证码题目里的图形**逐个裁出来、编号后拼成的对照表**。
每个格子里是一个图形，格子左上角有它的编号。

题目要求：「点击两个相同的图形」。

请找出**完全相同**的两个图形：
- 同一个三维积木从不同角度看也算相同；但**块数不同、轮廓不同就不是同一个**。
- 仔细比较：组成的小方块数量、整体的长宽比例、凸出的分支在哪一侧、
  有没有某个角缺一块、是直条还是拐角、分支的长度。
- 必须找到一对；如果确实找不到一样的，才输出空数组。

只输出一个 JSON，禁止任何解释：
{"pair":[编号a,编号b],"reason":"为什么这两个相同（15字内）"}"""


def _to_jpeg(im, quality=90):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _parse_shapes(plan):
    """兼容 {"shapes":[...]} 和直接给数组/单个坐标的退化输出。"""
    if not isinstance(plan, dict):
        return []
    arr = plan.get("shapes")
    if not isinstance(arr, list):
        arr = []
        for k in ("points", "items", "objects", "result"):
            if isinstance(plan.get(k), list):
                arr = plan[k]
                break
    out = []
    for it in arr:
        try:
            x = float(it.get("x"))
            y = float(it.get("y"))
        except Exception:
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append((x, y))
    return out


def find_shapes(img, ask, verbose=True):
    """第 1 段：让 VLM 只定位，返回归一化中心点列表。"""
    pts = []
    for attempt in range(2):
        plan, meta = ask(_to_jpeg(img), hint=LOCATE_PROMPT)
        if verbose:
            print("    [定位] %s -> %s" % (meta.get("elapsed"), meta.get("error") or ""))
        pts = _parse_shapes(plan or {})
        if verbose:
            print("    [定位] 找到 %d 个图形: %s"
                  % (len(pts), ", ".join("(%.3f,%.3f)" % p for p in pts)))
        if len(pts) >= 3:
            return pts
    return pts


def build_sheet(img, pts, tile=150, scale=2, labels=True):
    """把每个图形裁出来，编号拼成一张对照表。返回 (sheet_img, meta)。"""
    W, H = img.size
    half = int(tile * 0.5)
    thumbs = []
    for i, (nx, ny) in enumerate(pts):
        cx, cy = int(nx * W), int(ny * H)
        box = (max(0, cx - half), max(0, cy - half),
               min(W, cx + half), min(H, cy + half))
        t = img.crop(box).resize((tile, tile), Image.LANCZOS)
        thumbs.append((i + 1, t))

    cols = 3 if len(thumbs) <= 6 else 4
    rows = (len(thumbs) + cols - 1) // cols
    pad, head = 8, 22
    cw, ch = tile * scale, tile * scale
    sheet = Image.new("RGB", (cols * (cw + pad) + pad,
                              rows * (ch + head + pad) + pad), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for k, (num, t) in enumerate(thumbs):
        r, c = divmod(k, cols)
        x0 = pad + c * (cw + pad)
        y0 = pad + r * (ch + head + pad)
        if labels:
            d.text((x0 + 4, y0 + 3), "#%d" % num, fill=(255, 220, 60))
        sheet.paste(t.resize((cw, ch), Image.LANCZOS), (x0, y0 + head))
        d.rectangle([x0, y0, x0 + cw - 1, y0 + ch + head - 1],
                    outline=(70, 70, 80), width=2)
    return sheet, {"cols": cols, "rows": rows, "n": len(thumbs)}


def _parse_pair(plan, n):
    if not isinstance(plan, dict):
        return None
    pair = plan.get("pair")
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    try:
        a, b = int(pair[0]), int(pair[1])
    except Exception:
        return None
    if a == b or not (1 <= a <= n) or not (1 <= b <= n):
        return None
    return (a, b)


def solve_pair_from_image(img, ask, verbose=True, votes=1):
    """两段式求解：定位 → 拼表 → 配同形。返回 points（相对传入图的归一化坐标）。"""
    t0 = time.time()
    res = {"ok": False, "points": [], "pair": None, "n": 0, "reason": ""}

    pts = find_shapes(img, ask, verbose=verbose)
    if len(pts) < 2:
        res["reason"] = "定位到的图形不足 2 个"
        return res
    res["n"] = len(pts)

    sheet, meta = build_sheet(img, pts)
    sp = os.path.join(DEBUG_DIR, "shapes_sheet_%s.jpg" % time.strftime("%H%M%S"))
    try:
        sheet.save(sp, quality=88)
        if verbose:
            print("    [对照表] %dx%d 共 %d 格 -> %s"
                  % (sheet.size[0], sheet.size[1], meta["n"], sp))
    except Exception:
        pass

    pair = None
    for v in range(max(1, votes)):
        plan, m2 = ask(_to_jpeg(sheet), hint=PAIR_PROMPT)
        if verbose:
            print("    [配对] 第 %d 票 %s -> %s"
                  % (v + 1, m2.get("elapsed"), json.dumps(plan, ensure_ascii=False)[:160]))
        p = _parse_pair(plan, len(pts))
        if p:
            pair = p
            break
    if not pair:
        res["reason"] = "配对失败"
        return res

    res["pair"] = pair
    res["points"] = [pts[pair[0] - 1], pts[pair[1] - 1]]
    res["ok"] = True
    res["reason"] = "第 %s 与第 %s 个图形同形" % pair
    if verbose:
        print("    [+] 选 #%d %s 与 #%d %s  （%.1fs）"
              % (pair[0], "(%.3f,%.3f)" % res["points"][0],
                 pair[1], "(%.3f,%.3f)" % res["points"][1], time.time() - t0))
    return res

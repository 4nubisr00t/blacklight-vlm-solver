# -*- coding: utf-8 -*-
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "producer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PRODUCER) not in sys.path:
    sys.path.insert(0, str(PRODUCER))


def _make_concentric_panel_with_blocks():
    """Synthesize an hCaptcha-like 600x600 panel with concentric ring camo and 5 blocks."""
    W, H = 600, 600
    arr = np.full((H, W, 3), 245, dtype=np.float32)
    # Teal header & light gray body
    arr[:120, :, :] = [10, 128, 130]
    arr[120:510, :, :] = [238, 238, 238]

    # Textured square canvas inside [100..500, 135..495]
    x0, y0, x1, y1 = 100, 135, 500, 495
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r1 = np.hypot(xx - 220, yy - 240)
    r2 = np.hypot(xx - 380, yy - 360)
    rings = 10.0 * np.sin(r1 * 0.45) + 8.0 * np.sin(r2 * 0.42)
    bg_patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    bg_patch[:, :, 0] = 148 + rings + 14 * np.sin(xx * 0.025)
    bg_patch[:, :, 1] = 158 + rings + 12 * np.cos(yy * 0.025)
    bg_patch[:, :, 2] = 195 + rings
    arr[y0:y1, x0:x1] = np.clip(bg_patch, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    dr = ImageDraw.Draw(img)

    def draw_block(cells, ox, oy, cell=18):
        for cx, cy in cells:
            px = ox + cx * cell
            py = oy + cy * cell
            # Shadow side face + pale top face + seam border
            dr.rectangle([px + 2, py + 4, px + cell + 2, py + cell + 4], fill=(45, 42, 65))
            dr.rectangle([px, py, px + cell, py + cell], fill=(205, 200, 222), outline=(65, 62, 85), width=2)

    # Identical L-tetromino pair (#1 upright, #4 rotated 90 deg) + 3 decoys
    draw_block([(0, 0), (0, 1), (0, 2), (1, 2)], 165, 195)
    draw_block([(0, 0), (1, 0), (2, 0), (1, 1)], 315, 195)
    draw_block([(0, 0), (1, 0), (0, 1), (1, 1)], 240, 290)
    draw_block([(0, 0), (1, 0), (2, 0), (0, 1)], 175, 385)
    draw_block([(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)], 335, 385)
    return img


def _make_3x3_grid_panel():
    img = Image.new("RGB", (600, 600), (242, 242, 242))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 600, 120], fill=(10, 128, 130))
    # 3x3 colored tiles with crisp white grid dividers inside y=[0.22..0.82]
    x0, y0, size = 120, 135, 360
    cell = size // 3
    for r in range(3):
        for c in range(3):
            cx0 = x0 + c * cell + 6
            cy0 = y0 + r * cell + 6
            dr.rectangle([cx0, cy0, cx0 + cell - 12, cy0 + cell - 12], fill=(80 + r * 30, 120, 160))
    for k in (1, 2):
        gx = x0 + k * cell
        gy = y0 + k * cell
        dr.rectangle([gx - 4, y0, gx + 4, y0 + size], fill=(255, 255, 255))
        dr.rectangle([x0, gy - 4, x0 + size, gy + 4], fill=(255, 255, 255))
    return img


def test_has_3x3_grid_distinguishes_scatter_vs_grid():
    import captcha_brain as CB

    scatter = _make_concentric_panel_with_blocks()
    grid = _make_3x3_grid_panel()
    assert CB._has_3x3_grid(scatter) is False
    assert CB._has_3x3_grid(grid) is True


def test_enhance_and_detect_objects_on_concentric_camo():
    import captcha_brain as CB

    panel = _make_concentric_panel_with_blocks()
    objs = CB._detect_canvas_objects_cv(panel)
    assert len(objs) == 5
    for x, y in objs:
        assert 0.16 <= x <= 0.84
        assert 0.22 <= y <= 0.82

    tools = CB.Tools(page=None, verbose=False)
    tools.panel = panel
    tools.objects = objs
    sh = tools.sheet()
    assert sh["ok"] is True
    assert sh["count"] == 5
    assert len(sh["tiles"]) == 5

    # Verify _piece_bbox keeps each piece bounded and non-overlapping with panel margins
    box0 = tools._piece_bbox(objs[0][0], objs[0][1], self_idx=0)
    assert box0 is not None
    assert box0[2] - box0[0] < 0.30 * panel.size[0]
    assert box0[3] - box0[1] < 0.32 * panel.size[1]

    # Verify 2x2 rotation-aligned confirmation sheet renders cleanly
    conf_im, al = tools._pair_confirm_sheet(0, 3)
    assert conf_im.size[0] > 600 and conf_im.size[1] > 400
    assert 0 <= al["angle"] < 360


def test_guard_blocks_blind_points_on_scatter_but_allows_3x3_grid():
    import captcha_brain as CB

    brain = CB.CaptchaBrain(page=None, verbose=False)
    brain.tools.panel = _make_concentric_panel_with_blocks()
    msg = brain._guard("commit", {"tool": "commit", "points": [[0.3, 0.4], [0.6, 0.5]]})
    assert msg is not None and "locate" in msg

    # Second blind attempt is not deadlocked forever, and 3x3 grid is allowed immediately
    brain_grid = CB.CaptchaBrain(page=None, verbose=False)
    brain_grid.tools.panel = _make_3x3_grid_panel()
    assert brain_grid._guard("commit", {"tool": "commit", "points": [[0.18, 0.32], [0.50, 0.53]]}) is None


def test_solve_pair_cv_fallback_and_confirm(monkeypatch):
    import captcha_brain as CB

    panel = _make_concentric_panel_with_blocks()
    tools = CB.Tools(page=None, verbose=False)
    tools.panel = panel
    tools.objects = CB._detect_canvas_objects_cv(panel)

    # Simulate VLM signatures giving matching cube count n=4 without exact matching box string
    def fake_ask_multi(images, prompt, effort=None, timeout=60, chain=None):
        return {
            "sig": [
                {"id": i + 1, "n": 4 if i in (0, 1, 2, 3) else 5, "box": [2, 2], "pro": ""}
                for i in range(len(tools.objects))
            ]
        }, {"elapsed": 0.1, "model": "mock"}

    seen_confirm = []

    def fake_ask_vlm(jpeg_bytes, hint="", raw_prompt=False, effort=None, chain=None, force=False):
        seen_confirm.append(hint)
        return {"same": True, "why": "same cubes"}, {"elapsed": 0.1}

    monkeypatch.setattr(CB, "_ask_multi", fake_ask_multi)
    monkeypatch.setattr(CB.CA, "ask_vlm", fake_ask_vlm)
    monkeypatch.setattr(tools, "commit_pair", lambda a, b, verify=False: {
        "ok": True, "passed": True, "new_page": False, "ids": [a, b]
    })

    res = tools.solve_pair()
    assert res["ok"] is True
    assert res["passed"] is True
    assert len(res["chosen"]) == 2
    assert len(seen_confirm) == 1
    assert "去除背景同心圆噪点" in seen_confirm[0]


def test_signatures_sends_single_composite_sheet_and_guard_limits_checkbox_clicks(monkeypatch):
    import captcha_brain as CB

    panel = _make_concentric_panel_with_blocks()
    tools = CB.Tools(page=None, verbose=False)
    tools.panel = panel
    tools.objects = CB._detect_canvas_objects_cv(panel)

    captured = {}

    def fake_ask_multi(images, prompt, effort=None, timeout=60, chain=None):
        captured["count"] = len(images)
        captured["timeout"] = timeout
        return {
            "sig": [
                {"id": i + 1, "n": 4 if i in (0, 3) else 5, "box": [2, 3], "pro": ""}
                for i in range(len(tools.objects))
            ]
        }, {"elapsed": 0.2, "model": "mock"}

    monkeypatch.setattr(CB, "_ask_multi", fake_ask_multi)
    sg = tools.signatures()
    assert sg["ok"] is True
    assert captured["count"] == 1
    assert captured["timeout"] >= 60

    # Verify uniform-scale CV ranking produces valid normalized pair scores
    scores = tools._rank_pairs_cv(list(range(len(tools.objects))))
    assert len(scores) == 10
    assert all(0.0 <= v <= 1.0 for v in scores.values())

    # Verify checkbox blind viewport click guard caps at 2 clicks when panel is None
    brain = CB.CaptchaBrain(page=None, verbose=False)
    brain.tools.panel = None
    for _ in range(2):
        call = {"tool": "click", "x": 0.4, "y": 0.5, "space": "viewport"}
        assert brain._guard("click", call) is None
        brain._note_tool("click", call)
    assert brain._guard("click", {"tool": "click", "x": 0.4, "y": 0.5, "space": "viewport"}) is not None

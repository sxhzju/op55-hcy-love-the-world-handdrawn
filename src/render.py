"""Hand-drawn paper style lyric video renderer.

Reads an SRT file, draws one frame at a time with Pillow (strokes supersampled
2x for smooth anti-aliasing) and pipes raw RGB frames into ffmpeg, which muxes
the untouched original audio starting at t=0.

Usage:
    uv run python src/render.py --srt subtitles.srt --audio original.wav --out op55-hd.mp4
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = ROOT / "assets" / "fonts" / "PaperHand-Subset.ttf"
PAPER_PATH = ROOT / "assets" / "paper.png"

W, H, FPS = 1080, 1440, 30
SS = 2  # stroke supersampling factor
BOIL_FRAMES = 4  # redraw jitter every N frames ("line boil" of hand animation)

INK = (46, 40, 36)
INK_SOFT = (92, 84, 76)
PALETTE = {
    "red": (226, 94, 82),
    "pink": (240, 150, 160),
    "yellow": (246, 196, 72),
    "blue": (98, 160, 214),
    "sky": (150, 196, 232),
    "green": (118, 178, 112),
    "orange": (240, 146, 70),
    "purple": (160, 128, 200),
    "gray": (128, 130, 146),
    "light": (206, 210, 220),
    "skin": (250, 214, 180),
    "brown": (196, 150, 110),
    "tan": (238, 205, 160),
}

TITLE = "好想爱这个世界啊"
SUBTITLE = "华晨宇 · 补全结尾版"

# Lyric stage: illustrations live in this box (output pixel coords).
STAGE = (90, 300, 990, 1010)
TEXT_BOX = (80, 1060, 1000, 1330)


# --------------------------------------------------------------------------- SRT
@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str
    motif: str = ""


def _ts(s: str) -> float:
    h, m, rest = s.strip().replace(".", ",").split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.split("\n")]
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        a, b = lines[ti].split("-->")
        text = "\n".join(ln.strip() for ln in lines[ti + 1 :] if ln.strip())
        idx = int(lines[0]) if ti > 0 and lines[0].strip().isdigit() else len(cues) + 1
        cues.append(Cue(idx, _ts(a), _ts(b.split()[0]), text))
    return cues


# ------------------------------------------------------------------ motif choice
# Each cue is split into clauses at punctuation; every clause gets an
# illustration chosen by meaning.  A keyword may map to several motifs so a
# repeated phrase ("想过离开" ... "曾想过离开") gets a fresh picture.
KEYWORDS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("离开", "告别", "走了"), ("leave", "door")),
    (("阴霾", "败给", "乌云"), ("gloom",)),
    (("拨开", "推开", "照亮"), ("part",)),
    (("坚持", "现在", "熬", "撑"), ("sprout",)),
    (("旁白", "议论", "流言", "声音"), ("bubbles",)),
    (("伤害", "伤", "痛", "姿态"), ("mended",)),
    (("爱", "心", "喜欢", "温柔", "拥抱"), ("heart",)),
    (("世界", "地球", "人间"), ("globe",)),
    (("阳光", "太阳", "光", "温暖", "晴"), ("sun",)),
    (("夜", "月", "梦", "睡", "晚"), ("moon",)),
    (("星", "宇宙", "银河"), ("stars",)),
    (("雨", "泪", "哭"), ("rain",)),
    (("飞", "鸟", "翅膀", "自由", "天空", "远方"), ("birds",)),
    (("花", "开", "春", "生命"), ("flower",)),
    (("窗", "房间", "门", "家"), ("window",)),
    (("海", "浪", "河", "水"), ("sea",)),
    (("云", "风", "雾", "飘"), ("cloud",)),
    (("手", "牵", "握"), ("hands",)),
    (("歌", "唱", "听", "音乐"), ("notes",)),
    (("路", "走", "回", "去", "一直"), ("road",)),
]
FALLBACK = ["globe", "heart", "birds", "sun", "flower", "stars", "cloud", "notes"]
CLAUSE_RE = re.compile(r".+?(?:[，。、；！？,.!?…\s]+|$)")
MIN_SCENE = 2.0  # seconds; shorter clauses are merged into one picture


@dataclass
class Scene:
    start: float
    end: float
    motif: str
    cue: Cue
    span: tuple[int, int]  # character range of the clause inside cue.text


def clauses(text: str) -> list[tuple[int, int]]:
    out = []
    for m in CLAUSE_RE.finditer(text):
        if m.group().strip():
            out.append((m.start(), m.end()))
    return out or [(0, len(text))]


def _weight(text: str) -> int:
    return max(1, len(re.sub(r"[，。、；！？,.!?…\s]", "", text)))


def pick_motif(text: str, used: list[str]) -> str:
    # KEYWORDS is ordered by priority: specific story motifs before generic ones
    hits = [motifs for words, motifs in KEYWORDS if any(w in text for w in words)]
    for motifs in hits:
        for m in motifs:
            if m not in used:
                return m
    for motifs in hits:
        if motifs[0] != (used[-1] if used else ""):
            return motifs[0]
    return next(m for m in FALLBACK if not used or m != used[-1])


def build_scenes(cues: list[Cue]) -> list[Scene]:
    scenes: list[Scene] = []
    used: list[str] = []
    for c in cues:
        spans = clauses(c.text)
        dur = c.end - c.start
        # merge clauses until each one has enough screen time
        while len(spans) > 1:
            total = sum(_weight(c.text[a:b]) for a, b in spans)
            shortest = min(range(len(spans)), key=lambda i: _weight(c.text[spans[i][0]:spans[i][1]]))
            if dur * _weight(c.text[spans[shortest][0]:spans[shortest][1]]) / total >= MIN_SCENE:
                break
            j = shortest + 1 if shortest + 1 < len(spans) else shortest - 1
            a, b = sorted((shortest, j))
            spans[a:b + 1] = [(spans[a][0], spans[b][1])]
        total = sum(_weight(c.text[a:b]) for a, b in spans)
        t = c.start
        for i, (a, b) in enumerate(spans):
            end = c.end if i == len(spans) - 1 else t + dur * _weight(c.text[a:b]) / total
            motif = pick_motif(c.text[a:b], used)
            used.append(motif)
            scenes.append(Scene(t, end, motif, c, (a, b)))
            t = end
        c.motif = scenes[-1].motif
    return scenes


# ---------------------------------------------------------------- stroke engine
@dataclass
class Stroke:
    pts: list[tuple[float, float]]
    width: float = 5.0
    color: tuple[int, int, int] = INK
    closed: bool = False
    tag: str = ""


@dataclass
class Fill:
    pts: list[tuple[float, float]]
    color: tuple[int, int, int]
    alpha: int = 150
    tag: str = ""


@dataclass
class Doodle:
    strokes: list[Stroke] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    name: str = ""
    center: tuple[float, float] = (0.0, 0.0)


def _noise(seed: int, n: int, amp: float) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    return rng.normal(0.0, amp, size=(n, 2))


def resample(pts: list[tuple[float, float]], step: float = 9.0) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = seg.sum()
    if total == 0:
        return p
    n = max(2, int(total / step) + 1)
    d = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, total, n)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], axis=1)


def wobble(p: np.ndarray, seed: int, amp: float) -> np.ndarray:
    """Low-frequency hand tremor: smoothed noise along the stroke."""
    if len(p) < 3:
        return p + _noise(seed, len(p), amp * 0.5)
    nz = _noise(seed, len(p) + 4, amp)
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    nz = np.stack([np.convolve(nz[:, 0], k, "valid"), np.convolve(nz[:, 1], k, "valid")], 1)
    return p + nz[: len(p)] * 1.6


def partial(p: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 1 or len(p) < 2:
        return p
    if frac <= 0:
        return p[:0]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    cut = d[-1] * frac
    k = int(np.searchsorted(d, cut))
    if k <= 0:
        return p[:1]
    a, b = p[k - 1], p[k]
    r = (cut - d[k - 1]) / max(seg[k - 1], 1e-6)
    return np.vstack([p[:k], a + (b - a) * r])


class Canvas:
    """2x supersampled RGBA stroke layer."""

    def __init__(self) -> None:
        self.img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        self.fill_img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.fd = ImageDraw.Draw(self.fill_img)

    def stroke(self, s: Stroke, seed: int, frac: float = 1.0, amp: float = 2.2, alpha: int = 255) -> None:
        pts = list(s.pts)
        if s.closed:
            pts = pts + [pts[0]]
        base = resample(pts)
        if len(base) < 2 or frac <= 0:
            return
        for pas, (w_mul, a_mul, off) in enumerate(((1.0, 1.0, 0.0), (0.45, 0.55, 1.4))):
            p = wobble(base, seed * 7 + pas * 101, amp)
            if off:
                p = p + _noise(seed * 13 + 5, 1, off)
            p = partial(p, frac)
            if len(p) < 2:
                continue
            xy = [(float(x) * SS, float(y) * SS) for x, y in p]
            col = (*s.color, int(alpha * a_mul))
            self.d.line(xy, fill=col, width=max(1, int(round(s.width * w_mul * SS))), joint="curve")
            r = s.width * w_mul * SS / 2
            for (x, y) in (xy[0], xy[-1]):
                self.d.ellipse((x - r, y - r, x + r, y + r), fill=col)

    def fill(self, f: Fill, seed: int, alpha_mul: float) -> None:
        if alpha_mul <= 0:
            return
        p = wobble(resample(f.pts + [f.pts[0]], 14), seed, 3.0) + np.array([6.0, 5.0])
        xy = [(float(x) * SS, float(y) * SS) for x, y in p]
        self.fd.polygon(xy, fill=(*f.color, int(f.alpha * alpha_mul)))

    def result(self) -> Image.Image:
        fills = self.fill_img.filter(ImageFilter.GaussianBlur(2 * SS))
        out = Image.alpha_composite(fills, self.img)
        return out.reduce(SS)


# ------------------------------------------------------------------ doodle kit
def circle(cx, cy, r, n=48, start=0.0, sweep=2 * math.pi, squash=1.0):
    return [(cx + r * math.cos(start + sweep * i / n), cy + r * squash * math.sin(start + sweep * i / n)) for i in range(n + 1)]


def heart_pts(cx, cy, s):
    pts = []
    for i in range(61):
        t = math.pi * 2 * i / 60
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * s, cy + y * s))
    return pts


def star_pts(cx, cy, r, rot=-math.pi / 2):
    pts = []
    for i in range(11):
        rr = r if i % 2 == 0 else r * 0.45
        a = rot + math.pi * i / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def cloud_pts(cx, cy, s):
    """Outline of a union of circles above a flat base, traced by polar rays."""
    bumps = [(-0.85, 0.05, 0.42), (-0.4, -0.22, 0.5), (0.15, -0.32, 0.55), (0.7, -0.08, 0.45), (1.0, 0.12, 0.32)]
    base_y = 0.36
    pts = []
    for i in range(120):
        th = 2 * math.pi * i / 120
        dx, dy = math.cos(th), math.sin(th)
        best = 0.0
        for bx, by, br in bumps:  # farthest ray/circle intersection
            b = dx * bx + dy * by
            c = bx * bx + by * by - br * br
            disc = b * b - c
            if disc >= 0:
                best = max(best, b + math.sqrt(disc))
        if dy > 0:  # clip by flat bottom
            best = min(best, base_y / dy) if best > 0 else base_y / dy
        if best <= 0:
            continue
        pts.append((cx + best * dx * s, cy + best * dy * s))
    return pts


def bird(cx, cy, s):
    return [(cx - s, cy - 0.3 * s), (cx - 0.5 * s, cy - 0.55 * s), (cx, cy), (cx + 0.5 * s, cy - 0.55 * s), (cx + s, cy - 0.3 * s)]


def figure(x, fy, h=1.0, walk=False, tag=""):
    """Stick figure standing on y=fy (feet)."""
    hy = fy - 200 * h
    out = [Stroke(circle(x, hy, 28 * h, 24), 5, tag=tag),
           Stroke([(x, hy + 28 * h), (x + 2, fy - 80 * h)], 5, tag=tag)]
    spread = 42 if walk else 30
    out += [Stroke([(x + 2, fy - 80 * h), (x - spread * h, fy)], 5, tag=tag),
            Stroke([(x + 2, fy - 80 * h), (x + spread * h, fy)], 5, tag=tag),
            Stroke([(x, fy - 150 * h), (x - 48 * h, fy - 96 * h)], 5, tag=tag),
            Stroke([(x, fy - 150 * h), (x + 48 * h, fy - 96 * h)], 5, tag=tag)]
    return out


def hand(x, y, s, ang):
    """Open palm (fingers pointing along `ang`), as point lists."""
    def r(px, py):
        ca, sa = math.cos(ang), math.sin(ang)
        return (x + (px * ca - py * sa) * s, y + (px * sa + py * ca) * s)
    palm = [r(-40, -32), r(10, -36), r(26, -10), r(26, 22), r(0, 36), r(-40, 30)]
    fingers = [[r(18, yy), r(70, yy * 1.15)] for yy in (-28, -9, 10, 28)]
    thumb = [r(-10, 34), r(20, 62)]
    return palm, fingers + [thumb]


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def rot_rect(cx, cy, w, h, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def build_doodle(name: str) -> Doodle:
    cx = (STAGE[0] + STAGE[2]) / 2
    cy = (STAGE[1] + STAGE[3]) / 2 + 10
    d = Doodle(name=name, center=(cx, cy))
    S, F = d.strokes.append, d.fills.append
    if name == "heart":
        F(Fill(heart_pts(cx, cy, 15), PALETTE["red"], 170))
        S(Stroke(heart_pts(cx, cy, 15), 6))
        F(Fill(heart_pts(cx - 250, cy - 170, 4.5), PALETTE["pink"], 160))
        S(Stroke(heart_pts(cx - 250, cy - 170, 4.5), 4))
        F(Fill(heart_pts(cx + 260, cy + 160, 5.5), PALETTE["pink"], 160))
        S(Stroke(heart_pts(cx + 260, cy + 160, 5.5), 4))
        for a in (-0.9, -0.5, 0.5, 0.9):
            x0, y0 = cx + 300 * math.sin(a), cy - 40 - 300 * math.cos(a) * 0.6
            S(Stroke([(x0, y0), (x0 + 30 * math.sin(a), y0 - 30 * math.cos(a))], 4))
    elif name == "globe":
        r = 230
        F(Fill(circle(cx, cy, r), PALETTE["sky"], 150))
        S(Stroke(circle(cx, cy, r, 64), 6))
        S(Stroke(circle(cx, cy, r, 40, squash=0.32), 3.5, INK_SOFT))
        for k in (0.35, 0.75):
            S(Stroke([(cx + r * k * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, 30)], 3.5, INK_SOFT))
            S(Stroke([(cx - r * k * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, 30)], 3.5, INK_SOFT))
        land = [(cx - 120, cy - 120), (cx - 40, cy - 160), (cx + 20, cy - 110), (cx - 10, cy - 40), (cx - 90, cy - 20), (cx - 150, cy - 60)]
        F(Fill(land, PALETTE["green"], 190))
        S(Stroke(land, 4, closed=True))
        land2 = [(cx + 60, cy + 40), (cx + 160, cy + 20), (cx + 190, cy + 100), (cx + 110, cy + 170), (cx + 50, cy + 120)]
        F(Fill(land2, PALETTE["green"], 190))
        S(Stroke(land2, 4, closed=True))
        S(Stroke(circle(cx, cy, r + 70, 60, -2.6, 1.6, squash=0.35), 4, PALETTE["orange"]))
    elif name == "sun":
        F(Fill(circle(cx, cy, 140), PALETTE["yellow"], 200))
        S(Stroke(circle(cx, cy, 140, 50), 6))
        for i in range(12):
            a = i * math.pi / 6 + 0.1
            S(Stroke([(cx + 185 * math.cos(a), cy + 185 * math.sin(a)), (cx + 270 * math.cos(a), cy + 270 * math.sin(a))], 5, PALETTE["orange"], tag="ray"))
        S(Stroke(circle(cx - 45, cy - 20, 14, 16), 4))
        S(Stroke(circle(cx + 45, cy - 20, 14, 16), 4))
        S(Stroke(circle(cx, cy + 20, 60, 20, 0.4, 2.3), 5))
    elif name == "moon":
        moon = circle(cx, cy, 190, 40, -1.9, 3.8) + circle(cx + 90, cy - 30, 160, 40, 1.75, -3.5)
        F(Fill(moon, PALETTE["yellow"], 180))
        S(Stroke(moon, 6))
        for (x, y, r) in ((cx + 230, cy - 200, 26), (cx - 260, cy + 180, 20), (cx + 260, cy + 190, 16)):
            F(Fill(star_pts(x, y, r), PALETTE["yellow"], 200))
            S(Stroke(star_pts(x, y, r), 3.5))
        S(Stroke([(cx - 330, cy - 240), (cx - 300, cy - 260), (cx - 260, cy - 240)], 3.5, INK_SOFT))
        S(Stroke([(cx + 120, cy + 230), (cx + 150, cy + 210), (cx + 190, cy + 230)], 3.5, INK_SOFT))
    elif name == "stars":
        for (x, y, r, col) in ((cx, cy - 20, 150, "yellow"), (cx - 260, cy - 200, 55, "orange"), (cx + 270, cy - 170, 65, "yellow"), (cx - 230, cy + 220, 50, "pink"), (cx + 240, cy + 210, 45, "orange")):
            F(Fill(star_pts(x, y, r), PALETTE[col], 200, tag=f"star{int(x)}_{int(y)}"))
            S(Stroke(star_pts(x, y, r), 5 if r > 100 else 4, tag=f"star{int(x)}_{int(y)}"))
        S(Stroke([(cx - 380, cy + 40), (cx - 200, cy + 20), (cx - 160, cy + 80)], 3, INK_SOFT))
        S(Stroke([(cx + 160, cy + 60), (cx + 260, cy + 40), (cx + 380, cy + 80)], 3, INK_SOFT))
    elif name == "rain":
        F(Fill(cloud_pts(cx, cy - 120, 170), PALETTE["sky"], 170))
        S(Stroke(cloud_pts(cx, cy - 120, 170), 6, closed=True))
        for i in range(9):
            x = cx - 200 + i * 50
            y = cy + 30 + (i % 3) * 50
            S(Stroke([(x, y), (x - 12, y + 55)], 4.5, PALETTE["blue"], tag=f"rain{i}"))
        drop = [(cx + 300, cy + 140)] + circle(cx + 300, cy + 210, 38, 20, -0.3, math.pi + 0.6)[0:] + [(cx + 300, cy + 140)]
        F(Fill(drop, PALETTE["blue"], 180))
        S(Stroke(drop, 4))
    elif name == "birds":
        S(Stroke(cloud_pts(cx - 220, cy + 180, 85), 4, INK_SOFT, closed=True))
        S(Stroke(cloud_pts(cx + 250, cy - 210, 70), 4, INK_SOFT, closed=True))
        for i, (x, y, s) in enumerate(((cx, cy - 30, 110), (cx - 230, cy - 160, 60), (cx + 210, cy + 40, 70), (cx - 60, cy + 170, 45), (cx + 330, cy - 80, 40))):
            S(Stroke(bird(x, y, s), 5.5 if s > 80 else 4.5, tag=f"bird{i}"))
        S(Stroke([(cx - 400, cy + 290), (cx - 150, cy + 250), (cx + 120, cy + 300), (cx + 400, cy + 260)], 4, PALETTE["blue"]))
    elif name == "flower":
        fy = cy - 60
        for i in range(6):
            a = i * math.pi / 3
            p = circle(cx + 85 * math.cos(a), fy + 85 * math.sin(a), 62, 26)
            F(Fill(p, PALETTE["pink"], 180, tag="petal"))
            S(Stroke(p, 4.5, tag="petal"))
        F(Fill(circle(cx, fy, 55), PALETTE["yellow"], 220, tag="petal"))
        S(Stroke(circle(cx, fy, 55, 30), 5, tag="petal"))
        S(Stroke([(cx, fy + 60), (cx + 10, cy + 150), (cx - 5, cy + 300)], 6, PALETTE["green"]))
        leaf = [(cx + 5, cy + 200), (cx + 90, cy + 140), (cx + 150, cy + 170), (cx + 70, cy + 220)]
        F(Fill(leaf, PALETTE["green"], 180))
        S(Stroke(leaf, 4, closed=True))
        S(Stroke([(cx - 400, cy + 310), (cx + 400, cy + 300)], 4, INK_SOFT))
    elif name == "window":
        x0, y0, x1, y1 = cx - 220, cy - 260, cx + 220, cy + 220
        F(Fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], PALETTE["sky"], 140))
        S(Stroke([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 6, closed=True))
        S(Stroke([(cx, y0), (cx, y1)], 5))
        S(Stroke([(x0, cy - 20), (x1, cy - 20)], 5))
        S(Stroke([(x0 - 40, y1 + 10), (x1 + 40, y1 + 10)], 7))
        S(Stroke([(x0 - 60, y0 - 20), (x0 - 20, cy + 40), (x0 - 70, y1 + 5)], 4.5, PALETTE["red"]))
        S(Stroke([(x1 + 60, y0 - 20), (x1 + 20, cy + 40), (x1 + 70, y1 + 5)], 4.5, PALETTE["red"]))
        F(Fill(circle(cx + 110, cy - 150, 50), PALETTE["yellow"], 200))
        S(Stroke(circle(cx + 110, cy - 150, 50, 24), 4))
        S(Stroke(bird(cx - 100, cy + 80, 50), 4))
    elif name == "sea":
        F(Fill(circle(cx + 220, cy - 150, 80), PALETTE["orange"], 200))
        S(Stroke(circle(cx + 220, cy - 150, 80, 30), 5))
        for k in range(4):
            y = cy + k * 75
            pts = [(cx - 420 + i * 20, y + 18 * math.sin(i * 0.7 + k)) for i in range(43)]
            S(Stroke(pts, 5, PALETTE["blue"] if k % 2 == 0 else INK_SOFT))
        boat = [(cx - 220, cy - 30), (cx - 40, cy - 30), (cx - 80, cy + 20), (cx - 190, cy + 20)]
        F(Fill(boat, PALETTE["red"], 190))
        S(Stroke(boat, 5, closed=True))
        S(Stroke([(cx - 130, cy - 30), (cx - 130, cy - 200)], 5))
        sail = [(cx - 125, cy - 195), (cx - 40, cy - 60), (cx - 125, cy - 60)]
        F(Fill(sail, PALETTE["yellow"], 170))
        S(Stroke(sail, 4, closed=True))
    elif name == "cloud":
        for (x, y, s, c) in ((cx - 60, cy - 60, 190, "sky"), (cx + 210, cy + 170, 100, "purple"), (cx - 250, cy + 200, 80, "sky")):
            F(Fill(cloud_pts(x, y, s), PALETTE[c], 150))
            S(Stroke(cloud_pts(x, y, s), 5.5 if s > 150 else 4.5, closed=True))
        for k in range(3):
            y = cy - 240 + k * 40
            S(Stroke([(cx + 120, y), (cx + 260, y - 10), (cx + 330, y + 15), (cx + 300, y + 40)], 3.5, INK_SOFT))
    elif name == "hands":
        F(Fill(heart_pts(cx, cy - 160, 5), PALETTE["red"], 190))
        S(Stroke(heart_pts(cx, cy - 160, 5), 4))
        # two stick figures holding hands
        for sx in (-1, 1):
            hx = cx + sx * 140
            F(Fill(circle(hx, cy - 20, 55), PALETTE["yellow"] if sx < 0 else PALETTE["pink"], 160))
            S(Stroke(circle(hx, cy - 20, 55, 28), 5))
            S(Stroke([(hx, cy + 35), (hx, cy + 190)], 5))
            S(Stroke([(hx, cy + 190), (hx - 45, cy + 290)], 5))
            S(Stroke([(hx, cy + 190), (hx + 45, cy + 290)], 5))
            S(Stroke([(hx, cy + 80), (hx - sx * 120, cy + 140)], 5))
            S(Stroke([(hx, cy + 80), (hx + sx * 90, cy + 150)], 5))
        S(Stroke([(cx - 420, cy + 300), (cx + 420, cy + 295)], 4, INK_SOFT))
    elif name == "notes":
        for (x, y, s, c) in ((cx - 120, cy, 1.0, "purple"), (cx + 150, cy - 90, 0.8, "blue"), (cx - 300, cy - 200, 0.55, "orange"), (cx + 300, cy + 170, 0.6, "red")):
            head = circle(x, y + 130 * s, 42 * s, 22, squash=0.75)
            F(Fill(head, PALETTE[c], 200))
            S(Stroke(head, 4.5))
            S(Stroke([(x + 40 * s, y + 125 * s), (x + 40 * s, y - 120 * s)], 5))
            S(Stroke([(x + 40 * s, y - 120 * s), (x + 110 * s, y - 70 * s), (x + 95 * s, y - 20 * s)], 5))
        S(Stroke([(cx - 420 + i * 20, cy + 280 + 15 * math.sin(i * 0.5)) for i in range(43)], 3.5, INK_SOFT))
    elif name == "road":
        S(Stroke([(cx - 60, cy - 160), (cx - 380, cy + 300)], 5))
        S(Stroke([(cx + 60, cy - 160), (cx + 380, cy + 300)], 5))
        for k in range(4):
            y = cy - 120 + k * 100
            S(Stroke([(cx, y), (cx, y + 40 + k * 10)], 6, PALETTE["yellow"]))
        hill = [(cx - 430, cy - 140), (cx - 200, cy - 230), (cx, cy - 160), (cx + 220, cy - 250), (cx + 430, cy - 150)]
        S(Stroke(hill, 5, PALETTE["green"]))
        F(Fill(circle(cx, cy - 250, 60), PALETTE["orange"], 200))
        S(Stroke(circle(cx, cy - 250, 60, 26, math.pi, math.pi), 5))
    elif name == "leave":
        hz = cy - 110
        S(Stroke([(cx - 430, hz), (cx + 430, hz + 4)], 4, INK_SOFT))
        F(Fill(circle(cx + 230, hz, 70, 30, math.pi, math.pi), PALETTE["orange"], 200))
        S(Stroke(circle(cx + 230, hz, 70, 30, math.pi, math.pi), 5))
        road = [(cx - 35, hz), (cx + 35, hz), (cx + 320, cy + 330), (cx - 320, cy + 330)]
        F(Fill(road, PALETTE["tan"], 150))
        S(Stroke([(cx - 35, hz), (cx - 320, cy + 330)], 5))
        S(Stroke([(cx + 35, hz), (cx + 320, cy + 330)], 5))
        for k in range(3):
            y = hz + 40 + k * 95
            S(Stroke([(cx, y), (cx, y + 30 + k * 12)], 5, PALETTE["yellow"]))
        for st in figure(cx + 70, cy + 180, 0.85, walk=True, tag="walker"):
            S(st)
        bag = rect(cx + 115, cy + 60, cx + 175, cy + 110)
        F(Fill(bag, PALETTE["red"], 200, tag="walker"))
        S(Stroke(bag, 4, closed=True, tag="walker"))
        S(Stroke(bird(cx - 230, cy - 230, 45), 4))
        S(Stroke(bird(cx - 150, cy - 280, 30), 3.5))
    elif name == "gloom":
        F(Fill(circle(cx + 80, cy - 110, 115), PALETTE["yellow"], 200))
        S(Stroke(circle(cx + 80, cy - 110, 115, 30, -1.75, 1.5), 5))
        F(Fill(cloud_pts(cx - 30, cy - 40, 220), PALETTE["gray"], 225, tag="gloomA"))
        S(Stroke(cloud_pts(cx - 30, cy - 40, 220), 6, closed=True, tag="gloomA"))
        F(Fill(cloud_pts(cx + 300, cy - 215, 100), PALETTE["gray"], 210, tag="gloomB"))
        S(Stroke(cloud_pts(cx + 300, cy - 215, 100), 5, closed=True, tag="gloomB"))
        for k in range(6):  # pencil hatching on the big cloud
            x = cx - 200 + k * 60
            S(Stroke([(x, cy + 20), (x + 45, cy - 40)], 2.5, INK_SOFT, tag="gloomA"))
        for i in range(9):
            x = cx - 220 + i * 50
            y = cy + 110 + (i % 3) * 40
            S(Stroke([(x, y), (x - 12, y + 50)], 4.5, PALETTE["blue"], tag=f"rain{i}"))
        S(Stroke([(cx - 430, cy + 310), (cx + 430, cy + 305)], 4, INK_SOFT))
        for st in figure(cx - 280, cy + 308, 0.55):
            S(st)
        S(Stroke(circle(cx - 280, cy + 308 - 110, 70, 20, math.pi * 1.05, math.pi * 0.9), 4, PALETTE["blue"]))
    elif name == "part":
        F(Fill(circle(cx, cy - 40, 125), PALETTE["yellow"], 215))
        S(Stroke(circle(cx, cy - 40, 125, 48), 6))
        d.center = (cx, cy - 40)
        for i in range(12):
            a = i * math.pi / 6 + 0.1
            S(Stroke([(cx + 165 * math.cos(a), cy - 40 + 165 * math.sin(a)), (cx + 225 * math.cos(a), cy - 40 + 225 * math.sin(a))], 5, PALETTE["orange"], tag="ray"))
        S(Stroke(circle(cx - 42, cy - 62, 12, 14), 4))
        S(Stroke(circle(cx + 42, cy - 62, 12, 14), 4))
        S(Stroke(circle(cx, cy - 25, 55, 20, 0.4, 2.3), 5))
        for side, tag in ((-1, "pushL"), (1, "pushR")):
            ccx = cx + side * 265
            F(Fill(cloud_pts(ccx, cy - 50, 125), PALETTE["light"], 220, tag=tag))
            S(Stroke(cloud_pts(ccx, cy - 50, 125), 5, closed=True, tag=tag))
            palm, fingers = hand(cx + side * 150, cy + 120, 1.1, math.pi if side < 0 else 0.0)
            F(Fill(palm, PALETTE["skin"], 230, tag=tag))
            S(Stroke(palm, 4.5, closed=True, tag=tag))
            for fp in fingers:
                S(Stroke(fp, 4.5, tag=tag))
            S(Stroke([(cx + side * 190, cy + 150), (cx + side * 120, cy + 290)], 5, tag=tag))
        for (x, y) in ((cx - 120, cy - 230), (cx + 140, cy - 220), (cx + 30, cy + 250)):
            S(Stroke([(x - 18, y), (x + 18, y)], 3.5, PALETTE["orange"]))
            S(Stroke([(x, y - 18), (x, y + 18)], 3.5, PALETTE["orange"]))
    elif name == "door":
        fl = cy + 260
        S(Stroke([(cx - 430, fl), (cx + 430, fl + 3)], 4, INK_SOFT))
        F(Fill(rect(cx - 120, cy - 250, cx + 120, fl), PALETTE["yellow"], 150))
        F(Fill([(cx - 120, fl), (cx + 120, fl), (cx + 300, cy + 335), (cx - 230, cy + 335)], PALETTE["yellow"], 95))
        S(Stroke(rect(cx - 120, cy - 250, cx + 120, fl), 6, closed=True))
        leaf = [(cx - 120, cy - 250), (cx - 35, cy - 215), (cx - 35, fl - 25), (cx - 120, fl)]
        F(Fill(leaf, PALETTE["brown"], 220))
        S(Stroke(leaf, 5, closed=True))
        S(Stroke(circle(cx - 55, cy + 20, 9, 12), 4))
        for st in figure(cx + 45, fl - 5, 0.85, tag="hesitate"):
            S(st)
        for (x, y) in ((cx - 300, cy + 300), (cx - 240, cy + 280), (cx - 330, cy + 250)):
            S(Stroke(circle(x, y, 14, 12, squash=0.6), 3.5, INK_SOFT))
        S(Stroke([(cx + 220, cy - 200), (cx + 250, cy - 230), (cx + 280, cy - 200)], 3.5, INK_SOFT))
    elif name == "sprout":
        gy = cy + 170
        F(Fill(rect(cx - 430, gy, cx + 430, cy + 340), PALETTE["tan"], 150))
        S(Stroke([(cx - 430, gy), (cx - 120, gy + 6), (cx + 120, gy - 4), (cx + 430, gy + 3)], 5))
        S(Stroke([(cx - 10, gy), (cx + 25, gy + 45), (cx - 15, gy + 85), (cx + 30, gy + 130), (cx + 5, gy + 165)], 4, INK))
        for (x, y, r) in ((cx - 260, gy + 70, 34), (cx + 230, gy + 90, 26), (cx - 120, gy + 130, 20)):
            F(Fill(circle(x, y, r, 18, squash=0.6), PALETTE["gray"], 160))
            S(Stroke(circle(x, y, r, 18, squash=0.6), 3.5))
        S(Stroke([(cx, gy), (cx - 12, cy + 40), (cx + 8, cy - 80), (cx, cy - 150)], 7, PALETTE["green"], tag="sprout"))
        for side in (-1, 1):
            leaf = [(cx, cy - 20 + side * 30), (cx + side * 80, cy - 70 + side * 20), (cx + side * 150, cy - 40 + side * 30), (cx + side * 80, cy + side * 25)]
            F(Fill(leaf, PALETTE["green"], 200, tag="sprout"))
            S(Stroke(leaf, 4, closed=True, tag="sprout"))
        bud = [(cx, cy - 150), (cx - 40, cy - 190), (cx - 25, cy - 250), (cx, cy - 225), (cx + 25, cy - 250), (cx + 40, cy - 190)]
        F(Fill(bud, PALETTE["pink"], 210, tag="sprout"))
        S(Stroke(bud, 4.5, closed=True, tag="sprout"))
        F(Fill(circle(cx + 300, cy - 230, 55), PALETTE["yellow"], 200))
        S(Stroke(circle(cx + 300, cy - 230, 55, 26), 4.5))
        for i in range(7):
            a = math.pi * (0.55 + i * 0.16)
            S(Stroke([(cx + 300 + 75 * math.cos(a), cy - 230 + 75 * math.sin(a)), (cx + 300 + 105 * math.cos(a), cy - 230 + 105 * math.sin(a))], 4, PALETTE["orange"]))
    elif name == "bubbles":
        for i, (x, y, rx, ry) in enumerate(((cx - 260, cy - 220, 120, 70), (cx + 40, cy - 260, 135, 72), (cx + 290, cy - 150, 105, 64), (cx - 310, cy - 40, 95, 58))):
            tag = f"bub{i}"
            body = circle(x, y, rx, 40, squash=ry / rx)
            tail = [(x + rx * 0.2, y + ry * 0.85), (x + rx * 0.45, y + ry * 1.6), (x + rx * 0.55, y + ry * 0.7)]
            F(Fill(body, PALETTE["light"], 200, tag=tag))
            S(Stroke(body, 4.5, tag=tag))
            S(Stroke(tail, 4.5, tag=tag))
            for k in range(2):
                yy = y - 18 + k * 32
                S(Stroke([(x - rx * 0.6 + j * rx * 0.2, yy + (10 if j % 2 else -10)) for j in range(7)], 3, INK_SOFT, tag=tag))
        canopy = circle(cx, cy + 90, 170, 40, math.pi, math.pi)
        F(Fill(canopy, PALETTE["red"], 215))
        S(Stroke(canopy, 6))
        S(Stroke([(cx - 170 + k * 85, cy + 90) for k in range(5)], 4))
        S(Stroke([(cx, cy - 80), (cx, cy + 190), (cx + 20, cy + 215), (cx + 40, cy + 195)], 5))
        for st in figure(cx - 35, cy + 335, 0.62):
            S(st)
    elif name == "mended":
        F(Fill(heart_pts(cx, cy - 10, 14.5), PALETTE["red"], 185))
        S(Stroke(heart_pts(cx, cy - 10, 14.5), 6))
        S(Stroke([(cx, cy - 120), (cx - 30, cy - 50), (cx + 25, cy + 10), (cx - 20, cy + 80), (cx + 5, cy + 150)], 4.5))
        band = rot_rect(cx, cy, 300, 92, -0.55)
        F(Fill(band, PALETTE["tan"], 240))
        S(Stroke(band, 4.5, closed=True))
        pad = rot_rect(cx, cy, 100, 70, -0.55)
        F(Fill(pad, PALETTE["brown"], 150))
        S(Stroke(pad, 3, INK_SOFT, closed=True))
        for (ox, oy) in ((-95, 0), (95, 0), (-115, -20), (115, 20)):
            ca, sa = math.cos(-0.55), math.sin(-0.55)
            x, y = cx + ox * ca - oy * sa, cy + ox * sa + oy * ca
            S(Stroke(circle(x, y, 4, 8), 3, INK_SOFT))
        for (x, y, sc) in ((cx - 300, cy - 210, 3.5), (cx + 310, cy + 190, 4.0)):
            F(Fill(heart_pts(x, y, sc), PALETTE["pink"], 170))
            S(Stroke(heart_pts(x, y, sc), 3.5))
        for a in (-0.9, -0.5, 0.5, 0.9):
            x0, y0 = cx + 300 * math.sin(a), cy - 60 - 300 * math.cos(a) * 0.6
            S(Stroke([(x0, y0), (x0 + 30 * math.sin(a), y0 - 30 * math.cos(a))], 4))
    else:  # paper plane
        plane = [(cx - 200, cy + 40), (cx + 220, cy - 120), (cx - 40, cy + 120)]
        F(Fill(plane, PALETTE["sky"], 160))
        S(Stroke(plane, 6, closed=True))
        S(Stroke([(cx - 40, cy + 120), (cx - 20, cy + 40), (cx + 220, cy - 120)], 4))
        S(Stroke([(cx - 230, cy + 60), (cx - 330, cy + 120), (cx - 300, cy + 200), (cx - 400, cy + 250)], 3.5, INK_SOFT))
    return d


def _xf(pts, cx, cy, ang=0.0, sc=1.0, dx=0.0, dy=0.0):
    ca, sa = math.cos(ang), math.sin(ang)
    out = []
    for x, y in pts:
        x, y = (x - cx) * sc, (y - cy) * sc
        out.append((cx + x * ca - y * sa + dx, cy + x * sa + y * ca + dy))
    return out


def idle_motion(d: Doodle, tag: str, pts, t: float):
    """Per-motif idle animation, applied once a doodle sits on stage."""
    cx, cy = d.center
    n = d.name
    if tag.startswith("rain"):
        k = int(tag[4:])
        ph = (t * 1.6 + k * 0.37) % 1.0
        pts = _xf(pts, cx, cy, dy=ph * 70 - 20)
    elif tag.startswith("bird"):
        k = int(tag[4:])
        pts = _xf(pts, cx, cy, dx=22 * math.sin(t * 1.3 + k), dy=12 * math.sin(t * 2.1 + k * 1.7))
    elif tag.startswith("star"):
        sx, sy = (float(v) for v in tag[4:].split("_"))
        pts = _xf(pts, sx, sy, ang=0.12 * math.sin(t * 2 + sx), sc=1 + 0.08 * math.sin(t * 4 + sy))
    elif tag == "ray":
        pts = _xf(pts, cx, cy, ang=t * 0.35)
    elif tag == "walker":
        pts = _xf(pts, cx, cy, dy=-min(t, 5.0) * 14 - abs(math.sin(t * 5)) * 6)
    elif tag in ("pushL", "pushR"):
        side = -1 if tag == "pushL" else 1
        pts = _xf(pts, cx, cy, dx=side * 70 * ease_out(t / 3.0))
    elif tag in ("gloomA", "gloomB"):
        pts = _xf(pts, cx, cy, dx=(14 if tag == "gloomA" else -10) * math.sin(t * 0.7))
    elif tag == "hesitate":
        pts = _xf(pts, cx, cy, dx=8 * math.sin(t * 1.4))
    elif tag == "sprout":
        pts = _xf(pts, cx, cy + 170, ang=0.035 * math.sin(t * 1.6))
    elif tag.startswith("bub"):
        k = int(tag[3:])
        pts = _xf(pts, cx, cy, dx=6 * math.sin(t * 1.1 + k), dy=10 * math.sin(t * 1.7 + k * 2))
    elif tag == "petal":
        pts = _xf(pts, cx, cy - 60, ang=0.06 * math.sin(t * 1.5))
    if n in ("heart", "mended"):
        beat = max(0.0, math.sin(t * 2 * math.pi * 1.1)) ** 6
        pts = _xf(pts, cx, cy, sc=1 + 0.06 * beat)
    elif n == "globe":
        pts = _xf(pts, cx, cy, ang=0.05 * math.sin(t * 0.9))
    elif n == "cloud":
        pts = _xf(pts, cx, cy, dx=18 * math.sin(t * 0.8))
    elif n not in ("rain", "sun", "leave", "gloom", "part", "door", "sprout", "bubbles"):
        pts = _xf(pts, cx, cy, dy=7 * math.sin(t * 1.8))
    return pts


def draw_doodle(cv: Canvas, d: Doodle, prog: float, seed: int, alpha: float = 1.0, t: float = 0.0) -> None:
    """prog 0..1: strokes draw on sequentially, fills wash in afterwards."""
    n = len(d.strokes)
    if n == 0:
        return
    span = 0.8  # fraction of the reveal spent on outlines
    per = span / n
    for i, s in enumerate(d.strokes):
        frac = (prog - i * per * 0.75) / (per * 1.8)
        moved = Stroke(idle_motion(d, s.tag, s.pts, t), s.width, s.color, s.closed, s.tag)
        cv.stroke(moved, seed + i * 17, max(0.0, min(1.0, frac)), alpha=int(255 * alpha))
    fill_a = max(0.0, min(1.0, (prog - 0.45) / 0.45)) * alpha
    for j, f in enumerate(d.fills):
        cv.fill(Fill(idle_motion(d, f.tag, f.pts, t), f.color, f.alpha, f.tag), seed + 900 + j, fill_a)


# --------------------------------------------------------------------- paper
def make_paper(seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[:] = (247, 240, 225)
    # low-frequency blotches
    low = rng.normal(0, 1, (H // 24 + 1, W // 24 + 1)).astype(np.float32)
    low_img = Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    low_img = low_img.filter(ImageFilter.GaussianBlur(30))
    lowf = np.asarray(low_img, dtype=np.float32) / 255.0 - 0.5
    base += lowf[..., None] * np.array([14, 16, 22], dtype=np.float32)
    # fine grain
    grain = rng.normal(0, 1, (H, W)).astype(np.float32)
    grain_img = Image.fromarray(np.clip(grain * 40 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    base += (np.asarray(grain_img, dtype=np.float32)[..., None] - 128) * 0.12
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    base *= (1 - 0.13 * np.clip(r - 0.45, 0, 1) ** 1.6)[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    # paper fibers
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(900):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, math.pi)
        ln = rng.uniform(6, 26)
        tone = int(rng.uniform(150, 215))
        d.line([(x, y), (x + ln * math.cos(a), y + ln * math.sin(a))], fill=(tone, tone - 10, tone - 25, int(rng.uniform(18, 45))), width=1)
    # faint ruled lines in the lyric area (sketchbook feel)
    for y in range(TEXT_BOX[1] + 20, TEXT_BOX[3] + 40, 64):
        d.line([(60, y), (W - 60, y)], fill=(140, 170, 205, 34), width=2)
    d.line([(118, 0), (118, H)], fill=(220, 120, 120, 26), width=2)
    return img


# ---------------------------------------------------------------------- text
class TextRenderer:
    def __init__(self, font_path: Path) -> None:
        self.font_path = str(font_path)
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._cache: dict[tuple, Image.Image] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def layout(self, text: str, max_w: int, max_h: int, start: int = 80, min_size: int = 34):
        """Return (size, lines, line_height): the largest size whose wrapped text
        fits max_w x max_h.  Lines break at clause boundaries (punctuation or
        spaces) when possible, splitting inside a clause only as a last resort."""
        paras = [p.strip() for p in text.split("\n") if p.strip()] or [text]
        for allow_chars in (False, True):
            for size in range(start, min_size - 1, -2):
                f = self.font(size)
                fits = lambda l: f.getlength(l) * 1.06 <= max_w
                lh = int(size * 1.42)
                max_lines = max(1, max_h // lh)
                lines: list[str] = []
                for para in paras:
                    lines += self._wrap(para, f, fits, max_lines, allow_chars) or [None]
                if None not in lines and len(lines) <= max_lines:
                    return size, lines, lh
        return min_size, paras, int(min_size * 1.42)

    @staticmethod
    def _wrap(para: str, f, fits, max_lines: int, allow_chars: bool) -> list[str] | None:
        if fits(para):
            return [para]
        words = [m.group() for m in CLAUSE_RE.finditer(para) if m.group()]
        best = None
        for k in range(2, min(max_lines, len(words)) + 1):
            # contiguous grouping of clauses into k lines minimising the widest line
            for cuts in itertools.combinations(range(1, len(words)), k - 1):
                bounds = (0, *cuts, len(words))
                out = ["".join(words[bounds[i]:bounds[i + 1]]).strip() for i in range(k)]
                widest = max(f.getlength(l) for l in out)
                if all(fits(l) for l in out) and (best is None or widest < best[0]):
                    best = (widest, out)
            if best:
                return best[1]
        if not allow_chars:
            return None
        chars = para.replace(" ", "")
        for k in range(2, max_lines + 1):
            n = math.ceil(len(chars) / k)
            out = [chars[i : i + n] for i in range(0, len(chars), n)]
            if all(fits(l) for l in out):
                return out
        return None

    def glyph(self, ch: str, size: int, color, rot: float) -> Image.Image:
        key = (ch, size, color, round(rot, 1))
        if key not in self._cache:
            f = self.font(size)
            pad = size // 3
            im = Image.new("RGBA", (int(size * 1.2) + pad * 2, int(size * 1.4) + pad * 2), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=(*color, 255))
            if rot:
                im = im.rotate(rot, resample=Image.BICUBIC, expand=False)
            self._cache[key] = im
        return self._cache[key]


# ---------------------------------------------------------------- composition
def ease_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def text_layout(tr: TextRenderer, cue: Cue):
    """Glyph positions for a cue: list of (char index in cue.text, x, y, advance), size, line height."""
    x0, y0, x1, y1 = TEXT_BOX
    size, lines, lh = tr.layout(cue.text, x1 - x0 - 40, y1 - y0 - 20)
    f = tr.font(size)
    top = y0 + (y1 - y0 - len(lines) * lh) / 2
    pos, p = [], 0
    for li, line in enumerate(lines):
        x = (x0 + x1) / 2 - f.getlength(line) / 2
        y = top + li * lh
        for ch in line:
            while p < len(cue.text) and cue.text[p] != ch:
                p += 1
            pos.append((p, x, y, f.getlength(ch), ch))
            x += f.getlength(ch)
            p += 1
    return pos, size, lh


def render_text_block(frame: Image.Image, tr: TextRenderer, cue: Cue, t: float, seed: int) -> None:
    pos, size, lh = text_layout(tr, cue)
    dur = max(0.2, cue.end - cue.start)
    reveal_time = min(1.0, dur * 0.4)
    total = len(pos)
    shown = total * min(1.0, (t - cue.start) / reveal_time + 0.08)
    fade_out = max(0.0, min(1.0, (cue.end - t) / 0.18))
    rng = np.random.default_rng(cue.index * 991)
    pad = size // 3
    for k, (_, x, y, cw, ch) in enumerate(pos):
        jy = rng.normal(0, size * 0.03)
        rot = float(rng.normal(0, 2.2))
        if ch.isspace():
            continue
        a = max(0.0, min(1.0, shown - k)) * fade_out
        if a <= 0:
            continue
        g = tr.glyph(ch, size, INK, rot)
        if a < 1:
            g = g.copy()
            g.putalpha(g.getchannel("A").point(lambda v, a=a: int(v * a)))
        frame.alpha_composite(g, (int(x - pad), int(y + jy - pad - size * 0.08)))


UNDERLINE_COLORS = [PALETTE["yellow"], PALETTE["pink"], PALETTE["sky"], PALETTE["green"]]


def render_underline(cv: Canvas, tr: TextRenderer, sc: Scene, k: int, t: float, seed: int) -> None:
    """Highlighter stroke under the clause the current illustration belongs to."""
    pos, size, lh = text_layout(tr, sc.cue)
    a, b = sc.span
    rows: dict[float, list[float]] = {}
    for p, x, y, cw, ch in pos:
        if a <= p < b and not ch.isspace():
            rows.setdefault(y, []).extend([x, x + cw])
    prog = ease_out((t - sc.start - 0.15) / 0.7)
    fade = max(0.0, min(1.0, (sc.cue.end - t) / 0.18))
    col = UNDERLINE_COLORS[k % len(UNDERLINE_COLORS)]
    n = len(rows)
    for i, (y, xs) in enumerate(sorted(rows.items())):
        xa, xb = min(xs) - 8, max(xs) + 10
        yb = y + size * 1.12
        part = max(0.0, min(1.0, prog * n - i))
        cv.stroke(Stroke([(xa, yb), ((xa + xb) / 2, yb + 4), (xb, yb - 3)], 12, col),
                  seed + k * 31 + i, part, amp=1.6, alpha=int(150 * fade))


def stage_frame(cv: Canvas, seed: int, t: float) -> None:
    """Sketchbook frame around the illustration stage, drawn on during the intro."""
    x0, y0, x1, y1 = STAGE
    prog = ease_out(t / 1.2)
    box = [(x0, y0), (x1, y0 + 4), (x1 - 3, y1), (x0 + 2, y1 - 3)]
    cv.stroke(Stroke(box, 4, INK_SOFT, closed=True), seed + 3, prog, amp=1.8)
    # tape corners
    for (x, y, a) in ((x0 + 10, y0 + 6, -0.6), (x1 - 10, y0 + 8, 0.6)):
        dx, dy = 55 * math.cos(a), 55 * math.sin(a)
        nx, ny = -18 * math.sin(a), 18 * math.cos(a)
        tape = [(x - dx + nx, y - dy + ny), (x + dx + nx, y + dy + ny), (x + dx - nx, y + dy - ny), (x - dx - nx, y - dy - ny)]
        cv.fill(Fill(tape, (232, 214, 170), 170), seed + 77, prog)


def title_block(frame: Image.Image, cv: Canvas, tr: TextRenderer, seed: int, t: float) -> None:
    f = tr.font(78)
    tw = f.getlength(TITLE)
    x = (W - tw) / 2
    rng = np.random.default_rng(42)
    a = ease_out(t / 0.8)
    for ch in TITLE:
        rot = float(rng.normal(0, 2.5))
        g = tr.glyph(ch, 78, INK, rot)
        if a < 1:
            g = g.copy()
            g.putalpha(g.getchannel("A").point(lambda v: int(v * a)))
        frame.alpha_composite(g, (int(x - 26), int(88 + rng.normal(0, 3) - 26)))
        x += f.getlength(ch)
    f2 = tr.font(36)
    sw = f2.getlength(SUBTITLE)
    x = (W - sw) / 2
    for ch in SUBTITLE:
        g = tr.glyph(ch, 36, INK_SOFT, 0.0)
        if a < 1:
            g = g.copy()
            g.putalpha(g.getchannel("A").point(lambda v: int(v * a)))
        frame.alpha_composite(g, (int(x - 12), int(205 - 12)))
        x += f2.getlength(ch)
    ul = ease_out((t - 0.3) / 0.9)
    cv.stroke(Stroke([((W - tw) / 2 - 20, 190), (W / 2, 196), ((W + tw) / 2 + 20, 186)], 6, PALETTE["red"]), seed + 9, ul, amp=1.5)
    # little doodles flanking the title
    cv.stroke(Stroke(star_pts(110, 150, 30), 3.5, PALETTE["orange"]), seed + 11, ul)
    cv.stroke(Stroke(heart_pts(W - 110, 150, 2.4), 3.5, PALETTE["red"]), seed + 12, ul)


def progress_line(cv: Canvas, seed: int, t: float, total: float) -> None:
    y = 1385
    frac = max(0.0, min(1.0, t / total))
    xa, xb = 110, W - 110
    cv.stroke(Stroke([(xa, y), (xb, y)], 2.5, (190, 180, 165)), seed + 21, 1.0, amp=1.2, alpha=200)
    if frac > 0:
        x = xa + (xb - xa) * frac
        cv.stroke(Stroke([(xa, y), (x, y)], 4.5, INK_SOFT), seed + 22, 1.0, amp=1.2)
        plane = [(x - 26, y - 22), (x + 22, y - 4), (x - 18, y + 8)]
        cv.fill(Fill(plane, PALETTE["sky"], 200), seed + 23, 1.0)
        cv.stroke(Stroke(plane, 3, INK, closed=True), seed + 24, 1.0, amp=1.0)


def render_frame(n: int, scenes: list[Scene], doodles: dict[str, Doodle], paper: Image.Image, tr: TextRenderer, total: float) -> Image.Image:
    t = n / FPS
    seed = (n // BOIL_FRAMES) * 7919 + 1
    frame = paper.convert("RGBA")
    cv = Canvas()
    stage_frame(cv, seed, t)
    progress_line(cv, seed, t, total)

    # Illustration on stage: the latest scene that has started; the first one
    # is already sketched in during the short lead-in before the first cue.
    k = 0
    for i, sc in enumerate(scenes):
        if sc.start <= t:
            k = i
    sc = scenes[k]
    t0 = 0.0 if k == 0 else sc.start
    local = t - t0
    prog = ease_out(local / min(1.6, max(0.6, (sc.end - t0) * 0.6)))
    if k > 0 and local < 0.4:  # previous picture fades out underneath
        prev = scenes[k - 1]
        draw_doodle(cv, doodles[prev.motif], 1.0, seed + 5, alpha=1 - local / 0.4, t=t - (0.0 if k == 1 else prev.start))
    draw_doodle(cv, doodles[sc.motif], prog, seed, t=local)

    title_block(frame, cv, tr, seed, t)
    active = sc.cue if sc.cue.start <= t < sc.cue.end else None
    if active is not None:
        render_underline(cv, tr, sc, k, t, seed)
    frame.alpha_composite(cv.result())
    if active is not None:
        render_text_block(frame, tr, active, t, seed)
    return frame.convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", default=str(ROOT / "subtitles.srt"))
    ap.add_argument("--audio", default=str(ROOT / "original.wav"))
    ap.add_argument("--out", default=str(ROOT / "op55-hd.mp4"))
    ap.add_argument("--still", type=float, nargs="*", help="render PNG stills at these times instead of the video")
    args = ap.parse_args()

    cues = parse_srt(Path(args.srt))
    if not cues:
        print("no cues in srt", file=sys.stderr)
        return 1
    scenes = build_scenes(cues)
    for c in cues:
        print(f"cue {c.index:>3} {c.start:7.3f}-{c.end:7.3f} {c.text}")
    for sc in scenes:
        print(f"  scene {sc.start:7.3f}-{sc.end:7.3f} [{sc.motif:8}] {sc.cue.text[sc.span[0]:sc.span[1]]}")

    audio_dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.audio]).decode().strip())
    total = max(audio_dur, max(c.end for c in cues)) + 0.5
    nframes = int(math.ceil(total * FPS))
    print(f"audio {audio_dur:.3f}s, video {total:.3f}s, {nframes} frames")

    paper = Image.open(PAPER_PATH).convert("RGB") if PAPER_PATH.exists() else make_paper()
    tr = TextRenderer(FONT_PATH)
    doodles = {name: build_doodle(name) for name in {sc.motif for sc in scenes}}

    if args.still is not None:
        out_dir = ROOT / "build"
        out_dir.mkdir(exist_ok=True)
        for s in args.still:
            p = out_dir / f"still_{s:06.2f}.png"
            render_frame(int(round(s * FPS)), scenes, doodles, paper, tr, total).save(p)
            print("wrote", p)
        return 0

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", args.audio,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "320k", "-af", "apad", "-t", f"{nframes / FPS:.3f}",
        args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for n in range(nframes):
        proc.stdin.write(render_frame(n, scenes, doodles, paper, tr, total).tobytes())
        if n % 60 == 0:
            print(f"frame {n}/{nframes}", flush=True)
    proc.stdin.close()
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())

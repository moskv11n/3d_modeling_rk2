from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw

@dataclass(frozen=True)
class RenderConfig:
    width: int = 900
    height: int = 700
    margin: int = 60
    bg: Tuple[int, int, int] = (10, 12, 20)

    face_colors: Tuple[Tuple[int, int, int], ...] = (
        (200, 90, 120),
        (90, 200, 180),
        (220, 200, 90),
    )

    edge_color: Tuple[int, int, int] = (245, 245, 245)
    edge_width: int = 2

    axis_len: float = 120.0
    axis_width: int = 3
    axis_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 120, 120),
        (120, 255, 160),
        (120, 170, 255),
    )
    axis_label_color: Tuple[int, int, int] = (250, 250, 250)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]], dtype=float)


def fit_to_screen(xy, w, h, margin, bounds_from_xy=None):
    src = xy if bounds_from_xy is None else bounds_from_xy
    xmin, ymin = src.min(axis=0)
    xmax, ymax = src.max(axis=0)
    span_x = max(1e-9, xmax - xmin)
    span_y = max(1e-9, ymax - ymin)
    scale = min((w - 2 * margin) / span_x, (h - 2 * margin) / span_y)
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    sx, sy = w * 0.5, h * 0.5
    screen = np.empty_like(xy)
    screen[:, 0] = (xy[:, 0] - cx) * scale + sx
    screen[:, 1] = -(xy[:, 1] - cy) * scale + sy
    return screen, scale, (cx, cy), (sx, sy)


def rasterize_triangle_zbuf(img, zbuf, p0, p1, p2, z0, z1, z2, color):
    W, H = img.size
    pix = img.load()
    minx = max(int(min(p0[0], p1[0], p2[0])), 0)
    maxx = min(int(max(p0[0], p1[0], p2[0])), W - 1)
    miny = max(int(min(p0[1], p1[1], p2[1])), 0)
    maxy = min(int(max(p0[1], p1[1], p2[1])), H - 1)

    def edge(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    area = edge(p0, p1, p2)
    if abs(area) < 1e-9:
        return

    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            p = np.array([x + 0.5, y + 0.5])
            w0 = edge(p1, p2, p) / area
            w1 = edge(p2, p0, p) / area
            w2 = edge(p0, p1, p) / area
            if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > zbuf[y, x]:
                    zbuf[y, x] = z
                    pix[x, y] = color


def draw_line(draw, a, b, color, width):
    draw.line([a, b], fill=color, width=width)


def main():
    cfg = RenderConfig()

    V = np.array([
        [0.0,  0.0,  0.0],
        [45.0, 0.0,  0.0],
        [45.0, 45.0, 0.0],
        [0.0,  45.0, 0.0],
        [0.0,  0.0,  90.0],
    ], dtype=float)

    R = rot_x(10.0) @ rot_y(135.0)
    Vr = (R @ V.T).T

    xy = Vr[:, :2]
    z = Vr[:, 2]

    L = cfg.axis_len
    axes3d = np.array([
        [0.0, 0.0, 0.0],
        [L, 0.0, 0.0],
        [0.0, L, 0.0],
        [0.0, 0.0, L],
    ], dtype=float)

    axes_rot = (R @ axes3d.T).T
    axes_xy = axes_rot[:, :2]

    all_xy = np.vstack([xy, axes_xy])
    screen_xy, scale, (cx, cy), (sx, sy) = fit_to_screen(
        xy, cfg.width, cfg.height, cfg.margin, bounds_from_xy=all_xy
    )

    axes_screen = np.empty_like(axes_xy)
    axes_screen[:, 0] = (axes_xy[:, 0] - cx) * scale + sx
    axes_screen[:, 1] = -(axes_xy[:, 1] - cy) * scale + sy

    img = Image.new("RGB", (cfg.width, cfg.height), cfg.bg)
    zbuf = np.full((cfg.height, cfg.width), -1e18, dtype=float)

    filled_tris = [
        ((0, 1, 2), 0),
        ((0, 2, 3), 0),
        ((1, 2, 4), 1),
        ((2, 3, 4), 2),
    ]

    for (i0, i1, i2), ci in filled_tris:
        rasterize_triangle_zbuf(
            img, zbuf,
            screen_xy[i0], screen_xy[i1], screen_xy[i2],
            z[i0], z[i1], z[i2],
            cfg.face_colors[ci]
        )

    draw = ImageDraw.Draw(img)

    o = tuple(axes_screen[0])
    for idx, end_i in enumerate([1, 2, 3]):
        draw_line(draw, o, tuple(axes_screen[end_i]), cfg.axis_colors[idx], cfg.axis_width)

    try:
        draw.text((axes_screen[1][0] + 4, axes_screen[1][1] + 4), "X", fill=cfg.axis_label_color)
        draw.text((axes_screen[2][0] + 4, axes_screen[2][1] + 4), "Y", fill=cfg.axis_label_color)
        draw.text((axes_screen[3][0] + 4, axes_screen[3][1] + 4), "Z", fill=cfg.axis_label_color)
    except Exception:
        pass

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (0,1),(1,4),(4,0),
        (3,0),(0,4),(4,3),
        (2,4)
    ]

    for a, b in edges:
        draw_line(draw, tuple(screen_xy[a]), tuple(screen_xy[b]), cfg.edge_color, cfg.edge_width)

    img.save("zbuffer.png")

if __name__ == "__main__":
    main()

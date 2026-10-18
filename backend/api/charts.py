"""Tiny dependency-free SVG charts for the server-rendered dashboard.

Plotly is listed in the blueprint, but a research dashboard should render without
a CDN or a JS build step. These functions emit plain inline SVG, so the page is
self-contained and works offline.
"""

from __future__ import annotations

import html


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def bar_chart(items: list[dict], *, width: int = 640, max_value: float | None = None,
              label_width: int = 190) -> str:
    """Horizontal bars with 95% CI whiskers.

    items: [{"label": str, "value": float|None, "lo": float|None, "hi": float|None,
             "highlight": bool}]
    """
    items = [i for i in items if i.get("value") is not None]
    if not items:
        return '<p class="muted">No data.</p>'
    top = max_value or max(1.0, max(i["value"] for i in items) * 1.15)
    row_h, pad_top, pad_bottom = 30, 12, 26
    height = pad_top + pad_bottom + row_h * len(items)
    plot_w = width - label_width - 60
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="bar chart">']
    # gridlines
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_width + tick * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" '
                     f'y2="{height - pad_bottom}" stroke="#222a35" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" fill="#5c6b7f" '
                     f'font-size="10" text-anchor="middle">{tick * top:.2f}</text>')
    for idx, item in enumerate(items):
        y = pad_top + idx * row_h + row_h / 2
        v = float(item["value"])
        bar_w = max(1.0, v / top * plot_w)
        colour = "#7cc4ff" if item.get("highlight") else "#3d6b8f"
        parts.append(f'<text x="{label_width - 10}" y="{y + 4}" fill="#c9d4e3" '
                     f'font-size="11" text-anchor="end">{_esc(item["label"])}</text>')
        parts.append(f'<rect x="{label_width}" y="{y - 8}" width="{bar_w:.1f}" '
                     f'height="16" rx="3" fill="{colour}" opacity="0.85"/>')
        if item.get("lo") is not None and item.get("hi") is not None:
            lo = min(max(float(item["lo"]), 0.0), top)
            hi = min(max(float(item["hi"]), 0.0), top)
            x1 = label_width + lo / top * plot_w
            x2 = label_width + hi / top * plot_w
            parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" '
                         f'stroke="#e8eef7" stroke-width="1.4"/>')
            for x in (x1, x2):
                parts.append(f'<line x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" '
                             f'y2="{y + 5}" stroke="#e8eef7" stroke-width="1.4"/>')
        parts.append(f'<text x="{label_width + plot_w + 8:.1f}" y="{y + 4}" '
                     f'fill="#8fa2b8" font-size="10">{_fmt(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def calibration_chart(bins: list[dict], *, size: int = 300) -> str:
    """Reliability diagram: predicted probability vs observed frequency."""
    points = [(b["mean_pred"], b["frac_pos"], b["n"]) for b in bins
              if b.get("mean_pred") is not None and b.get("n")]
    if not points:
        return '<p class="muted">No calibration data.</p>'
    pad = 34
    plot = size - pad * 1.5
    parts = [f'<svg viewBox="0 0 {size} {size}" width="100%" '
             f'role="img" aria-label="calibration plot">']
    parts.append(f'<rect x="{pad}" y="{pad / 2}" width="{plot}" height="{plot}" '
                 f'fill="#0e141c" stroke="#222a35"/>')
    for i in range(5):
        t = i / 4
        x = pad + t * plot
        y = pad / 2 + plot - t * plot
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad + plot}" y2="{y:.1f}" '
                     f'stroke="#1a222d"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{pad / 2}" x2="{x:.1f}" '
                     f'y2="{pad / 2 + plot}" stroke="#1a222d"/>')
        parts.append(f'<text x="{pad - 6}" y="{y + 3:.1f}" fill="#5c6b7f" font-size="9" '
                     f'text-anchor="end">{t:.2f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{pad / 2 + plot + 14}" fill="#5c6b7f" '
                     f'font-size="9" text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<line x1="{pad}" y1="{pad / 2 + plot}" x2="{pad + plot}" y2="{pad / 2}" '
                 f'stroke="#3d6b8f" stroke-dasharray="4 4"/>')
    for pred, obs, n in points:
        x = pad + pred * plot
        y = pad / 2 + plot - obs * plot
        r = 3 + min(6, (n ** 0.5) / 3)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#7cc4ff" '
                     f'opacity="0.85"/>')
    parts.append(f'<text x="{pad + plot / 2}" y="{size - 4}" fill="#5c6b7f" font-size="9" '
                 f'text-anchor="middle">predicted probability</text>')
    parts.append("</svg>")
    return "".join(parts)

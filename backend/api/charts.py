"""Inline SVG charts.

No JS, no CDN: the dashboard renders offline and prints sensibly. Colours are
duplicated in static/site.css — keep the two in sync.
"""

from __future__ import annotations

import html

INK = "#e6edf6"
MUTED = "#8496ab"
GRID = "#1b2533"
BAR = "#31506f"
BAR_BEST = "#6ea8fe"
CI = "#c3d3e7"
DIAGONAL = "#3d5a7a"


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def bar_chart(items: list[dict], *, width: int = 720, max_value: float | None = None,
              label_width: int = 200, axis_label: str = "") -> str:
    """Horizontal bars with 95% CI whiskers.

    items: [{"label": str, "value": float|None, "lo": float|None, "hi": float|None,
             "highlight": bool}]
    """
    items = [i for i in items if i.get("value") is not None]
    if not items:
        return '<p class="muted">No data for this run.</p>'
    top = max_value or max(1.0, max(i["value"] for i in items) * 1.18)
    row_h, pad_top, pad_bottom = 32, 14, 46
    height = pad_top + pad_bottom + row_h * len(items)
    plot_w = width - label_width - 64
    bottom = height - pad_bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Model comparison bar chart">']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = label_width + tick * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top - 4}" x2="{x:.1f}" '
                     f'y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16}" fill="{MUTED}" '
                     f'font-size="10" text-anchor="middle">{tick * top:.2f}</text>')
    for idx, item in enumerate(items):
        y = pad_top + idx * row_h + row_h / 2
        value = float(item["value"])
        bar_w = max(1.5, value / top * plot_w)
        colour = BAR_BEST if item.get("highlight") else BAR
        label = _esc(item["label"])
        parts.append(f'<text x="{label_width - 12}" y="{y + 4}" fill="{INK}" '
                     f'font-size="11.5" text-anchor="end">{label}</text>')
        parts.append(f'<rect x="{label_width}" y="{y - 9}" width="{bar_w:.1f}" '
                     f'height="18" rx="3" fill="{colour}"><title>{label}: '
                     f'{_fmt(value)}</title></rect>')
        if item.get("lo") is not None and item.get("hi") is not None:
            lo = min(max(float(item["lo"]), 0.0), top)
            hi = min(max(float(item["hi"]), 0.0), top)
            x1 = label_width + lo / top * plot_w
            x2 = label_width + hi / top * plot_w
            parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" '
                         f'stroke="{CI}" stroke-width="1.3" opacity="0.85"/>')
            for x in (x1, x2):
                parts.append(f'<line x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" '
                             f'y2="{y + 5}" stroke="{CI}" stroke-width="1.3" '
                             f'opacity="0.85"/>')
        parts.append(f'<text x="{label_width + plot_w + 10:.1f}" y="{y + 4}" '
                     f'fill="{MUTED}" font-size="10.5">{_fmt(value)}</text>')

    if axis_label:
        parts.append(f'<text x="{label_width + plot_w / 2:.1f}" y="{height - 8}" '
                     f'fill="{MUTED}" font-size="10" text-anchor="middle">'
                     f'{_esc(axis_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def calibration_chart(bins: list[dict], *, size: int = 320) -> str:
    """Reliability diagram: predicted probability vs observed frequency."""
    points = [(b["mean_pred"], b["frac_pos"], b["n"]) for b in bins
              if b.get("mean_pred") is not None and b.get("n")]
    if not points:
        return '<p class="muted">No calibration data for this run.</p>'
    pad_l, pad_r, pad_t, pad_b = 46, 14, 14, 40
    plot = size - pad_l - pad_r
    plot_h = size - pad_t - pad_b
    parts = [f'<svg viewBox="0 0 {size} {size}" width="100%" role="img" '
             f'aria-label="Calibration plot">']
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot}" height="{plot_h}" '
                 f'fill="#0b1219" stroke="{GRID}"/>')
    for i in range(5):
        t = i / 4
        x = pad_l + t * plot
        y = pad_t + plot_h - t * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot}" '
                     f'y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{pad_t + plot_h}" stroke="{GRID}"/>')
        parts.append(f'<text x="{pad_l - 7}" y="{y + 3:.1f}" fill="{MUTED}" '
                     f'font-size="9" text-anchor="end">{t:.2f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 15}" fill="{MUTED}" '
                     f'font-size="9" text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" '
                 f'x2="{pad_l + plot}" y2="{pad_t}" stroke="{DIAGONAL}" '
                 f'stroke-dasharray="4 4"/>')
    for pred, obs, n in points:
        x = pad_l + pred * plot
        y = pad_t + plot_h - obs * plot_h
        r = 3.2 + min(5.5, (n ** 0.5) / 3.2)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{BAR_BEST}" '
                     f'fill-opacity="0.85" stroke="#0b1219" stroke-width="1">'
                     f'<title>{n} pairs</title></circle>')
    parts.append(f'<text x="{pad_l + plot / 2}" y="{size - 10}" fill="{MUTED}" '
                 f'font-size="9.5" text-anchor="middle">predicted probability</text>')
    parts.append(f'<text x="12" y="{pad_t + plot_h / 2}" fill="{MUTED}" font-size="9.5" '
                 f'text-anchor="middle" transform="rotate(-90 12 '
                 f'{pad_t + plot_h / 2})">observed rate</text>')
    parts.append("</svg>")
    return "".join(parts)

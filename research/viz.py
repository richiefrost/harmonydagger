"""
Chart helpers for the research Streamlit pages.

Palette is the validated default categorical set, slots 1-3 only. Slots 1-3 are the
all-pairs-safe cap: worst-pair CVD dE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9.
A 4th slot would put yellow next to orange and fail the all-pairs floors.

Light-mode aqua (#1baf7a) sits at 2.74:1 against the light surface, below 3:1, so the
relief rule applies: every chart here ships visible direct value labels AND a table view.
"""
from __future__ import annotations

# Validated categorical slots 1-3 (see references/palette.md)
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
DARK = ["#3987e5", "#d95926", "#199e70"]
SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
INK_LIGHT = "#0b0b0b"
INK_DARK = "#ffffff"
MUTED_LIGHT = "#52514e"
MUTED_DARK = "#c3c2b7"


def _theme(dark: bool):
    return {
        "series": DARK if dark else LIGHT,
        "surface": SURFACE_DARK if dark else SURFACE_LIGHT,
        "ink": INK_DARK if dark else INK_LIGHT,
        "muted": MUTED_DARK if dark else MUTED_LIGHT,
        "grid": "#3a3a38" if dark else "#e6e5e1",
    }


def _fig(dark: bool, w=7.2, h=None, n_rows=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _theme(dark)
    h = h or max(2.0, 0.42 * n_rows + 1.1)
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    ax.xaxis.label.set_color(t["muted"])
    return fig, ax, t


def hbar(labels, values, dark=False, fmt="{:+.2f}", slot=0, xlabel="", highlight=None,
         zero_line=True):
    """Horizontal bars, recessive axes, a direct value label on every bar.

    highlight: index to draw in a second hue (e.g. the arm that matters).
    """
    fig, ax, t = _fig(dark, n_rows=len(labels))
    colors = [
        t["series"][slot] if (highlight is None or i != highlight) else t["series"][1]
        for i in range(len(labels))
    ]
    y = range(len(labels))
    ax.barh(list(y), values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=t["ink"], fontsize=9)
    ax.invert_yaxis()
    if zero_line and min(values) < 0 < max(values):
        ax.axvline(0, color=t["grid"], lw=1, zorder=2)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)

    # Direct value labels sit just beyond each bar tip. The axis limits must reserve room
    # for them on whichever side they fall, or the text collides with the tick labels.
    span = max(abs(max(values)), abs(min(values))) or 1
    pad = span * 0.025
    for i, v in enumerate(values):
        ax.text(
            v + (pad if v >= 0 else -pad), i, fmt.format(v),
            va="center", ha="left" if v >= 0 else "right",
            color=t["ink"], fontsize=8.5, zorder=4,
        )
    lo, hi = min(0, min(values)), max(0, max(values))
    label_room = span * 0.30
    ax.set_xlim(
        lo - (label_room if min(values) < 0 else span * 0.04),
        hi + (label_room if max(values) > 0 else span * 0.04),
    )
    fig.tight_layout()
    return fig


def grouped_hbar(groups, series_names, series_values, dark=False, fmt="{:.1f}",
                 xlabel=""):
    """Grouped horizontal bars. series_values[s][g]. Max 3 series (palette cap)."""
    assert len(series_names) <= 3, "palette validates 3 all-pairs slots; fold extras"
    fig, ax, t = _fig(dark, n_rows=len(groups) * len(series_names) * 0.9 + 1)
    nb = len(series_names)
    height = 0.72 / nb
    import numpy as np

    base = np.arange(len(groups))
    for s, name in enumerate(series_names):
        offs = (s - (nb - 1) / 2) * height
        # 2px surface gap between adjacent fills -> height*0.88
        ax.barh(base + offs, series_values[s], height=height * 0.88,
                color=t["series"][s], label=name, zorder=3)
        for g, v in enumerate(series_values[s]):
            ax.text(v + max(map(max, series_values)) * 0.015, base[g] + offs,
                    fmt.format(v), va="center", ha="left", color=t["ink"],
                    fontsize=7.5, zorder=4)
    ax.set_yticks(base)
    ax.set_yticklabels(groups, color=t["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xlim(0, max(map(max, series_values)) * 1.18)
    # Legend above the plot: inside the axes it collides with the longest bars.
    leg = ax.legend(frameon=False, fontsize=8, ncol=nb, loc="lower left",
                    bbox_to_anchor=(0, 1.01), handlelength=1.1, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    fig.tight_layout()
    return fig


def dot_strip(values, dark=False, xlabel="", ref=0.0, label_fmt="{:+.3f}"):
    """One dot per observation -- shows a paired effect's consistency directly."""
    fig, ax, t = _fig(dark, h=1.7)
    import numpy as np

    jitter = np.linspace(-0.16, 0.16, len(values))
    ax.scatter(values, jitter, s=42, color=t["series"][0], zorder=3,
               edgecolors=t["surface"], linewidths=1.2)

    span = (max(values) - min(values)) or 1
    off = span * 0.02  # keep labels clear of the rules they annotate
    ax.axvline(ref, color=t["grid"], lw=1.4, zorder=2)
    ax.text(ref + off, 0.31, "no effect", color=t["muted"], fontsize=7.5, ha="left")
    m = float(np.mean(values))
    ax.axvline(m, color=t["series"][1], lw=2, zorder=4)
    ax.text(m + off, -0.33, f"mean {label_fmt.format(m)}", color=t["series"][1],
            fontsize=8.5, ha="left")
    ax.set_yticks([])
    ax.set_ylim(-0.46, 0.46)
    ax.set_xlim(min(ref, min(values)) - span * 0.10, max(values) + span * 0.10)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    fig.tight_layout()
    return fig


def spectrogram_diff(original, protected, sr, dark=False):
    """Where in time-frequency the perturbation actually sits.

    Sequential single hue, light->dark (never a rainbow): magnitude has one direction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = _theme(dark)
    n = min(len(original), len(protected))
    d = np.asarray(protected[:n], dtype=float) - np.asarray(original[:n], dtype=float)
    nfft, hop = 2048, 512
    win = np.hanning(nfft)
    frames = 1 + (len(d) - nfft) // hop
    if frames < 2:
        return None
    S = np.abs(
        np.stack([np.fft.rfft(d[i * hop : i * hop + nfft] * win) for i in range(frames)]).T
    )
    S_db = 20 * np.log10(S + 1e-10)
    S_db -= S_db.max()

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    cmap = "magma" if dark else "Blues"
    im = ax.imshow(S_db, origin="lower", aspect="auto", cmap=cmap, vmin=-70, vmax=0,
                   extent=[0, len(d) / sr, 0, sr / 2000])
    ax.set_xlabel("time (s)", fontsize=8, color=t["muted"])
    ax.set_ylabel("kHz", fontsize=8, color=t["muted"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("dB below peak", fontsize=7.5, color=t["muted"])
    cb.ax.tick_params(colors=t["muted"], labelsize=7, length=0)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig

"""画出更容易读的 g 敏感性图。

上排：固定桩 / 移动桩随 g 的变化。
下排：三个日期类型的 before-after 覆盖图。每个区一行，左边是
g=0.15 的移动桩台数，右边是 g=0.30；实心点=仍有车，空心点=被撤掉。
最右侧单独标出固定桩增加的区。

    /opt/anaconda3/bin/python3.9 plot_g_reconfiguration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"

G_VALUES = (0.15, 0.2, 0.25, 0.3, 0.35)
LOW_G = 0.15
HIGH_G = 0.3
DATE_TYPES = ("workday", "weekend", "holiday")
DATE_LABELS = {"workday": "Workday", "weekend": "Weekend", "holiday": "Holiday"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.unicode_minus": False,
    })


def load_solution(value: float) -> dict:
    path = EXP_DIR / f"g_{value:g}" / "charge_locally.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("converged"):
        raise ValueError(f"{path} 未收敛")
    taz_ids = [int(taz) for taz in payload["provenance"]["taz_ids"]]
    n = [int(round(count)) for count in payload["decision"]["n"]]
    m = int(round(payload["decision"]["m"]))
    z = {
        date_type: [int(round(count)) for count in payload["decision"]["z"][date_type]]
        for date_type in DATE_TYPES
    }
    return {"g": value, "taz_ids": taz_ids, "n": n, "m": m, "z": z}


def draw_fleet(ax_fixed: plt.Axes, ax_mobile: plt.Axes, solutions: list[dict]) -> None:
    gs = [item["g"] for item in solutions]
    fixed = [sum(item["n"]) for item in solutions]
    mobile = [item["m"] for item in solutions]
    ax_fixed.plot(gs, fixed, color="#1F4E79", marker="D", markersize=6, linewidth=1.7)
    ax_mobile.plot(gs, mobile, color="#2F6F4E", marker="s", markersize=6, linewidth=1.7)
    for ax, title, ylabel, ylim, yticks in (
        (ax_fixed, "(a) Fixed chargers", "Fixed chargers", (291.5, 296.8), [292, 294, 296]),
        (ax_mobile, "(b) Mobile chargers", "Mobile chargers", (3.6, 7.4), [4, 5, 6, 7]),
    ):
        ax.set_title(title)
        ax.set_xlabel(r"Local replenishment premium $g$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(G_VALUES))
        ax.set_xlim(0.13, 0.37)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvspan(0.3, 0.35, color="#E4E4E4", alpha=0.7, zorder=0)
        ax.annotate(
            "same design",
            xy=(0.325, ylim[0] + 0.72 * (ylim[1] - ylim[0])),
            ha="center",
            fontsize=9,
            color="#666666",
        )


def coverage_rows(low: dict, high: dict, date_type: str) -> list[tuple[int, int, int]]:
    rows = []
    for taz, z0, z1 in zip(low["taz_ids"], low["z"][date_type], high["z"][date_type]):
        if z0 > 0 or z1 > 0:
            rows.append((taz, z0, z1))
    # kept first, then abandoned; within each group by taz id
    rows.sort(key=lambda item: (item[2] == 0, -item[1], item[0]))
    return rows


def draw_coverage(
    ax: plt.Axes,
    rows: list[tuple[int, int, int]],
    *,
    title: str,
    gained_fixed: dict[int, int],
) -> None:
    if not rows:
        ax.set_axis_off()
        return
    y_positions = list(range(len(rows)))[::-1]
    x0, x1 = 0.0, 1.0
    for y, (taz, z0, z1) in zip(y_positions, rows):
        abandoned = z1 == 0
        color = "#B0B0B0" if abandoned else "#6B2D12"
        ax.plot([x0, x1], [y, y], color="#D0D0D0", linewidth=1.2, zorder=1)
        # low-g marker
        ax.scatter([x0], [y], s=160 + 70 * (z0 - 1), c="#6B2D12", zorder=3)
        ax.text(x0, y, str(z0), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
        # high-g marker
        if abandoned:
            ax.scatter([x1], [y], s=140, facecolors="white", edgecolors="#8A8A8A", linewidths=1.6, zorder=3)
            ax.text(x1, y, "0", ha="center", va="center", color="#8A8A8A", fontsize=8, fontweight="bold", zorder=4)
        else:
            ax.scatter([x1], [y], s=160 + 70 * (z1 - 1), c="#6B2D12", zorder=3)
            ax.text(x1, y, str(z1), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
        label = f"TAZ {taz}"
        if taz in gained_fixed:
            label += f"  (+{gained_fixed[taz]} fixed)"
            ax.text(-0.08, y, label, ha="right", va="center", fontsize=9, color="#1F4E79", fontweight="bold")
        else:
            ax.text(-0.08, y, label, ha="right", va="center", fontsize=9, color="#333333")
        if abandoned:
            ax.text(1.12, y, "withdrawn", ha="left", va="center", fontsize=8, color="#888888")
        elif z1 < z0:
            ax.text(1.12, y, "thinned", ha="left", va="center", fontsize=8, color="#8C4A17")
        else:
            ax.text(1.12, y, "kept", ha="left", va="center", fontsize=8, color="#2F6F4E")

    ax.set_xlim(-0.85, 1.55)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([f"$g={LOW_G:g}$", f"$g={HIGH_G:g}$"], fontsize=10)
    ax.set_yticks([])
    ax.set_title(title, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0)


def draw_fixed_panel(ax: plt.Axes, gained_fixed: dict[int, int], low: dict, high: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("(f) What changes from $g=0.15$ to $0.30$", pad=8)

    lines = [
        r"Mobile fleet:  $7 \rightarrow 4$",
        r"Fixed chargers:  $292 \rightarrow 296$",
        "",
        "Fixed chargers are added only at:",
    ]
    for taz, delta in sorted(gained_fixed.items()):
        lines.append(f"    TAZ {taz}:  +{delta}")
    lines.extend([
        "",
        "Reading the panels above:",
        "  • brown filled = mobile still used",
        "  • hollow = mobile withdrawn",
        "  • blue label = also gets more fixed",
        "",
        "Takeaway:",
        "  first drop date-type-specific sites,",
        "  then replace shared mobile peaks",
        "  with fixed chargers at multiple zones.",
    ])
    ax.text(
        0.03,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10.2,
        family="DejaVu Sans",
        linespacing=1.35,
        transform=ax.transAxes,
    )
    # small legend swatches
    ax.scatter([0.12], [0.08], s=120, c="#6B2D12", transform=ax.transAxes, clip_on=False)
    ax.text(0.17, 0.08, "kept / thinned", transform=ax.transAxes, va="center", fontsize=9)
    ax.scatter([0.48], [0.08], s=110, facecolors="white", edgecolors="#8A8A8A", linewidths=1.5, transform=ax.transAxes, clip_on=False)
    ax.text(0.53, 0.08, "withdrawn", transform=ax.transAxes, va="center", fontsize=9)


def compose(solutions: list[dict]) -> plt.Figure:
    by_g = {item["g"]: item for item in solutions}
    low = by_g[LOW_G]
    high = by_g[HIGH_G]
    gained_fixed = {
        taz: n1 - n0
        for taz, n0, n1 in zip(low["taz_ids"], low["n"], high["n"])
        if n1 > n0
    }

    fig = plt.figure(figsize=(13.2, 9.6))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.0, 2.15],
        width_ratios=[1.05, 1.05, 1.05, 0.95],
        hspace=0.28,
        wspace=0.35,
    )

    ax_fixed = fig.add_subplot(gs[0, 0:2])
    ax_mobile = fig.add_subplot(gs[0, 2:4])
    draw_fleet(ax_fixed, ax_mobile, solutions)

    letters = ("c", "d", "e")
    for idx, date_type in enumerate(DATE_TYPES):
        ax = fig.add_subplot(gs[1, idx])
        rows = coverage_rows(low, high, date_type)
        draw_coverage(
            ax,
            rows,
            title=f"({letters[idx]}) {DATE_LABELS[date_type]} mobile coverage",
            gained_fixed=gained_fixed,
        )

    ax_note = fig.add_subplot(gs[1, 3])
    draw_fixed_panel(ax_note, gained_fixed, low, high)

    fig.suptitle(
        r"Charge locally: network reconfiguration as local replenishment premium $g$ rises",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    return fig


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solutions = [load_solution(value) for value in G_VALUES]
    figure = compose(solutions)
    for suffix in (".png", ".pdf"):
        path = OUTPUT_DIR / f"fig_g_reconfiguration{suffix}"
        figure.savefig(path, bbox_inches="tight", pad_inches=0.15, dpi=300)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()

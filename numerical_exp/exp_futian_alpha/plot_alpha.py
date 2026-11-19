"""画出 alpha 从 0 到 1 时，两种 mixed 相对 fixed-only 的利润差，以及车队规模。

左图是目标差。右图左轴是固定桩数量，右轴是移动桩数量，各画 charge centrally
和 charge locally。数据：outputs/g0.25_hub50/summary.csv。

    python3 plot_alpha.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY = Path(__file__).resolve().parent / "outputs" / "g0.25_hub50" / "summary.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "g0.25_hub50"

MODE_LABEL = {
    "fixed-only": "Fixed-only",
    "charge-locally": "Charge locally",
    "charge-centrally": "Charge centrally",
}
MODE_COLOR = {
    "fixed-only": "#4C4C4C",
    "charge-locally": "#2F6F4E",
    "charge-centrally": "#1F4E79",
}
MODE_MARKER = {
    "fixed-only": "o",
    "charge-locally": "s",
    "charge-centrally": "D",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.unicode_minus": False,
    })


def load_summary() -> pd.DataFrame:
    frame = pd.read_csv(SUMMARY)
    present = sorted(frame["alpha"].mul(10).round().astype(int).unique().tolist())
    if present != list(range(11)):
        raise ValueError(f"summary.csv 的 alpha 不是 0 到 1、步长 0.1：{present}")
    if not bool(frame["converged"].all()):
        raise ValueError("summary.csv 含未收敛的行")
    return frame.sort_values(["mode", "alpha"])


def _panel_frame(ax: plt.Axes) -> None:
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"Demand-shift factor $\alpha$")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.spines["top"].set_visible(False)


def draw(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))

    fixed = frame[frame["mode"] == "fixed-only"][["alpha", "objective"]].rename(
        columns={"objective": "fixed_objective"}
    )
    gap = axes[0]
    for mode in ("charge-locally", "charge-centrally"):
        part = frame[frame["mode"] == mode].merge(fixed, on="alpha")
        gap.plot(
            part["alpha"],
            part["objective"] - part["fixed_objective"],
            color=MODE_COLOR[mode],
            marker=MODE_MARKER[mode],
            markersize=4.5,
            linewidth=1.4,
            label=MODE_LABEL[mode],
        )
    gap.axhline(0.0, color="#4C4C4C", linewidth=0.8)
    gap.set_title("Profit minus fixed-only")
    gap.set_ylabel("Objective difference")
    _panel_frame(gap)
    gap.spines["right"].set_visible(False)
    gap.legend(frameon=False, fontsize=9)

    fleet = axes[1]
    mobile_ax = fleet.twinx()
    fleet_handles = []
    for mode in ("charge-locally", "charge-centrally"):
        part = frame[frame["mode"] == mode]
        fleet_handles.append(fleet.plot(
            part["alpha"],
            part["fixed_chargers"],
            color=MODE_COLOR[mode],
            marker=MODE_MARKER[mode],
            markersize=4.5,
            linewidth=1.4,
            label=f"{MODE_LABEL[mode]}, fixed",
        )[0])
        fleet_handles.append(mobile_ax.plot(
            part["alpha"],
            part["mobile_chargers"],
            color=MODE_COLOR[mode],
            marker=MODE_MARKER[mode],
            markersize=4.5,
            linewidth=1.4,
            linestyle="--",
            label=f"{MODE_LABEL[mode]}, mobile",
        )[0])
    fleet.set_title("Charger fleet")
    fleet.set_ylabel("Fixed chargers")
    mobile_ax.set_ylabel("Mobile chargers")
    _panel_frame(fleet)
    fleet.spines["top"].set_visible(False)
    mobile_ax.spines["top"].set_visible(False)
    mobile_ax.set_ylim(-0.6, 13.2)
    fleet.legend(fleet_handles, [handle.get_label() for handle in fleet_handles], frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = draw(load_summary())
    for suffix in (".png", ".pdf"):
        path = OUTPUT_DIR / f"fig_alpha_profit{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()

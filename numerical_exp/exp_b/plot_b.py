"""画出 b 从 0 到 2 时，charge centrally 与 charge locally 的车队规模。

左图是固定桩数量，右图是移动桩数量。数据读
numerical_exp/exp_b/b{0,0.5,1,1.5,2}/ 下已收敛的 JSON。

    python3 plot_b.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
B_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)

MODE_FILE = {
    "charge-locally": "charge_locally.json",
    "charge-centrally": "charge_centrally.json",
}
MODE_LABEL = {
    "charge-locally": "Charge locally",
    "charge-centrally": "Charge centrally",
}
MODE_COLOR = {
    "charge-locally": "#2F6F4E",
    "charge-centrally": "#1F4E79",
}
MODE_MARKER = {
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


def _b_dir(value: float) -> Path:
    name = "b0" if value == 0 else f"b{value:g}"
    return EXP_DIR / name


def _fleet(path: Path, mode: str) -> tuple[int, int]:
    payload = json.loads(path.read_text())
    if mode == "charge-centrally":
        if payload.get("status_name") != "OPTIMAL":
            raise ValueError(f"{path} 未收敛")
        return int(payload["fixed_chargers"]), int(payload["mobile_chargers"])
    if not payload.get("converged"):
        raise ValueError(f"{path} 未收敛")
    fixed = int(round(sum(payload["decision"]["n"])))
    mobile = int(round(payload["decision"]["m"]))
    return fixed, mobile


def load_fleet() -> dict[str, list[tuple[float, int, int]]]:
    rows: dict[str, list[tuple[float, int, int]]] = {}
    for mode, filename in MODE_FILE.items():
        series = []
        for value in B_VALUES:
            path = _b_dir(value) / filename
            fixed, mobile = _fleet(path, mode)
            series.append((value, fixed, mobile))
        rows[mode] = series
    return rows


def _panel(ax: plt.Axes, title: str, ylabel: str, ylim: tuple[float, float], yticks: list[int]) -> None:
    ax.set_title(title)
    ax.set_xlabel(r"Shortage penalty $b$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(B_VALUES))
    ax.set_xlim(-0.08, 2.08)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw(rows: dict[str, list[tuple[float, int, int]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    handles = []
    for mode in ("charge-locally", "charge-centrally"):
        values = [item[0] for item in rows[mode]]
        handles.append(axes[0].plot(
            values,
            [item[1] for item in rows[mode]],
            color=MODE_COLOR[mode],
            marker=MODE_MARKER[mode],
            markersize=5.5,
            linewidth=1.5,
            label=MODE_LABEL[mode],
        )[0])
        axes[1].plot(
            values,
            [item[2] for item in rows[mode]],
            color=MODE_COLOR[mode],
            marker=MODE_MARKER[mode],
            markersize=5.5,
            linewidth=1.5,
        )
    _panel(axes[0], "Fixed chargers", "Fixed chargers", (270, 350), [280, 300, 320, 340])
    _panel(axes[1], "Mobile chargers", "Mobile chargers", (-0.5, 16.5), [0, 4, 8, 12, 16])
    fig.legend(handles, [handle.get_label() for handle in handles], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = draw(load_fleet())
    for suffix in (".png", ".pdf"):
        path = OUTPUT_DIR / f"fig_b_fleet{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()

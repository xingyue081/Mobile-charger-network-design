"""画出 b 从 0 到 2 时，两种模式在历史样本上的 unmet 与服务水平。

上排是各日期类型的日均 unmet（kWh），下排是对应的服务水平
（样本合计 served / 合计 demand）。容量按该次求解的 fixed + 日期类型
条件 mobile 部署计算；需求用该次 provenance 指向的 futian_parameters.npz
（b=0 的 locally 指向 alpha_1，与 futian 样本一致）。

    python3 plot_b_service.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
B_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
DATE_TYPES = ("workday", "weekend", "holiday")
DATE_TITLE = {
    "workday": "Workday",
    "weekend": "Weekend",
    "holiday": "Holiday",
}
VF = 190.0
VM = 160.0

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
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.unicode_minus": False,
    })


def _b_dir(value: float) -> Path:
    name = "b0" if value == 0 else f"b{value:g}"
    return EXP_DIR / name


def _load_samples(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as stored:
        return {date: stored[f"samples_{date}"].astype(float) for date in DATE_TYPES}


def _artifact_path(payload: dict, mode: str) -> Path:
    if mode == "charge-locally":
        return Path(payload["provenance"]["artifact"])
    # centrally 结果未存 artifact；福田实验统一用 futian 包
    return EXP_DIR.parents[1] / "futian" / "futian_parameters.npz"


def _capacity_local(payload: dict) -> dict[str, np.ndarray]:
    fixed = np.asarray(payload["decision"]["n"], dtype=float)
    deployment = {
        date: np.asarray(payload["decision"]["z"][date], dtype=float)
        for date in DATE_TYPES
    }
    return {date: VF * fixed + VM * deployment[date] for date in DATE_TYPES}


def _capacity_centrally(payload: dict) -> dict[str, np.ndarray]:
    zone_count = len(payload["zone_ids"])
    fixed = np.asarray(
        [payload["decision"]["n"][str(index)] for index in range(zone_count)],
        dtype=float,
    )
    mobile = {index: np.zeros(zone_count, dtype=float) for index in range(3)}
    for key, value in payload["decision"]["z"].items():
        zone, _hub, date_index = map(int, key.split(","))
        mobile[date_index][zone] += float(value)
    return {
        date: VF * fixed + VM * mobile[date_index]
        for date_index, date in enumerate(DATE_TYPES)
    }


def _metrics(capacity: dict[str, np.ndarray], samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for date in DATE_TYPES:
        demand_matrix = samples[date]
        demand = demand_matrix.sum(axis=1)
        unmet = np.maximum(demand_matrix - capacity[date], 0.0).sum(axis=1)
        served = demand - unmet
        rows[date] = {
            "mean_unmet": float(unmet.mean()),
            "mean_demand": float(demand.mean()),
            "service_level": float(served.sum() / demand.sum()),
            "mean_fill_rate": float((served / np.maximum(demand, 1e-12)).mean()),
            "shortage_day_share": float((unmet > 1e-6).mean()),
        }
    return rows


def evaluate() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for mode, filename in MODE_FILE.items():
        for value in B_VALUES:
            path = _b_dir(value) / filename
            payload = json.loads(path.read_text())
            if mode == "charge-centrally" and payload.get("status_name") != "OPTIMAL":
                raise ValueError(f"{path} 未收敛")
            if mode == "charge-locally" and not payload.get("converged"):
                raise ValueError(f"{path} 未收敛")
            samples = _load_samples(_artifact_path(payload, mode))
            if mode == "charge-locally":
                capacity = _capacity_local(payload)
            else:
                capacity = _capacity_centrally(payload)
            metrics = _metrics(capacity, samples)
            for date in DATE_TYPES:
                item = metrics[date]
                rows.append({
                    "mode": mode,
                    "b": value,
                    "date_type": date,
                    "mean_unmet": item["mean_unmet"],
                    "mean_demand": item["mean_demand"],
                    "service_level": item["service_level"],
                    "mean_fill_rate": item["mean_fill_rate"],
                    "shortage_day_share": item["shortage_day_share"],
                })
    return rows


def write_csv(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "b_service_by_date.csv"
    fieldnames = [
        "mode", "b", "date_type", "mean_unmet", "mean_demand",
        "service_level", "mean_fill_rate", "shortage_day_share",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _series(rows: list[dict[str, float | str]], mode: str, date: str, field: str) -> list[float]:
    selected = [
        row for row in rows
        if row["mode"] == mode and row["date_type"] == date
    ]
    selected.sort(key=lambda row: float(row["b"]))
    return [float(row[field]) for row in selected]


def draw(rows: list[dict[str, float | str]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), sharex=True)
    handles = []
    for column, date in enumerate(DATE_TYPES):
        unmet_ax = axes[0, column]
        service_ax = axes[1, column]
        for mode in ("charge-locally", "charge-centrally"):
            unmet = _series(rows, mode, date, "mean_unmet")
            service = [100.0 * value for value in _series(rows, mode, date, "service_level")]
            line = unmet_ax.plot(
                list(B_VALUES),
                unmet,
                color=MODE_COLOR[mode],
                marker=MODE_MARKER[mode],
                markersize=5.0,
                linewidth=1.5,
                label=MODE_LABEL[mode],
            )[0]
            service_ax.plot(
                list(B_VALUES),
                service,
                color=MODE_COLOR[mode],
                marker=MODE_MARKER[mode],
                markersize=5.0,
                linewidth=1.5,
            )
            if column == 0:
                handles.append(line)
        unmet_ax.set_title(DATE_TITLE[date])
        unmet_ax.set_ylim(0, 3200)
        unmet_ax.set_yticks([0, 800, 1600, 2400, 3200])
        unmet_ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        unmet_ax.spines["top"].set_visible(False)
        unmet_ax.spines["right"].set_visible(False)
        service_ax.set_xlabel(r"Shortage penalty $b$")
        service_ax.set_xticks(list(B_VALUES))
        service_ax.set_xlim(-0.08, 2.08)
        service_ax.set_ylim(94.0, 100.0)
        service_ax.set_yticks([94, 96, 98, 100])
        service_ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        service_ax.spines["top"].set_visible(False)
        service_ax.spines["right"].set_visible(False)
        if column == 0:
            unmet_ax.set_ylabel("Mean unmet demand (kWh)")
            service_ax.set_ylabel("Service level (%)")
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    rows = evaluate()
    csv_path = write_csv(rows)
    print(csv_path)
    figure = draw(rows)
    for suffix in (".png", ".pdf"):
        path = OUTPUT_DIR / f"fig_b_service{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()

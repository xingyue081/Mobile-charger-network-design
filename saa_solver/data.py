"""读取福田 SAA 使用的区域、成本和训练日需求。

业务常数、距离、固定桩上界和日期类型概率 q_l 来自 CG_solver/futian，
与 charge_centrally_solver 的福田实例相同。
需求来自构建该 npz 时使用的同一张日需求表，字段是 daily_volume。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATE_TYPES = ("workday", "weekend", "holiday")
DEMAND_CSV = Path("/Users/xingyueyi/Downloads/论文/models/demand_samples_by_taz_date_type.csv")
FUTIAN_DIR = Path(__file__).resolve().parents[1] / "futian"
# 与 charge_centrally_solver.main.load_futian_data 相同：每个入选 TAZ 都是候选 hub。
FLEET_PER_HUB = 30.0
SHARED_BUSINESS_FIELDS = (
    "rho",
    "shortage_penalty",
    "fixed_capacity",
    "mobile_capacity",
    "fixed_unit_cost",
    "mobile_unit_cost",
    "transport_cost_per_distance",
    "hub_open_cost",
)


def load_shared_business(data_dir: Path, overrides=None) -> tuple[dict, dict]:
    path = Path(data_dir) / "business.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in SHARED_BUSINESS_FIELDS if field not in stored]
    if missing:
        raise ValueError(f"business.json 缺少 {missing}")
    values = {field: float(stored[field]) for field in SHARED_BUSINESS_FIELDS}
    applied = {}
    for field, value in (overrides or {}).items():
        if value is None:
            continue
        if field not in SHARED_BUSINESS_FIELDS:
            raise ValueError(f"不能覆盖业务常数 {field}")
        values[field] = float(value)
        applied[field] = float(value)
    return values, applied


def _training_panel(taz_ids: tuple[int, ...], metadata: dict) -> tuple[pd.DataFrame, list[str]]:
    if metadata.get("demand_field") != "daily_volume":
        raise ValueError("福田元数据的需求字段应为 daily_volume")
    start = metadata["train_start_inclusive"]
    end = metadata["train_end_exclusive"]
    raw = pd.read_csv(DEMAND_CSV)
    required = {"date", "taz_id", "date_type", "daily_volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"需求表缺少列 {sorted(missing)}")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= start) & (raw["date"] < end)]
    raw = raw[raw["taz_id"].isin(taz_ids)]
    if set(raw["date_type"]) != set(DATE_TYPES):
        raise ValueError(f"日期类型应为 {DATE_TYPES}，实际是 {sorted(raw['date_type'].unique())}")
    panel = raw.pivot(index="date", columns="taz_id", values="daily_volume").sort_index()
    panel = panel.reindex(columns=list(taz_ids))
    if panel.isna().any().any():
        raise ValueError("训练窗内有 zone 缺日需求")
    labels = (
        raw.drop_duplicates("date")
        .set_index("date")
        .loc[panel.index, "date_type"]
        .astype(str)
        .tolist()
    )
    if len(labels) != len(panel):
        raise ValueError("日期类型没有覆盖全部训练日")
    if len(panel) != int(metadata["training_days"]):
        raise ValueError(
            f"训练日数量是 {len(panel)}，元数据记录的是 {metadata['training_days']}"
        )
    values = panel.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("daily_volume 必须有限且非负")
    return panel, labels


def _support_mask(values: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, radius_squared: float) -> np.ndarray:
    diff = values - mu
    distance = np.einsum("ij,jk,ik->i", diff, sigma_inv, diff)
    return distance <= radius_squared


def load_futian(sample_set: str = "all", business_overrides=None, data_dir: Path | None = None) -> dict:
    """装入 charge-centrally SAA 的一份福田数据。

    sample_set="all" 使用训练窗内的全部日期。
    sample_set="support" 只保留 Mahalanobis 椭球内的日期，样本与 DRO 的 npz 相同。
    两种情况下 q_l 都用 npz 里的日期类型概率，不按本次保留的天数重算。
    """
    if sample_set not in ("all", "support"):
        raise ValueError("sample_set 只能是 all 或 support")
    root = FUTIAN_DIR if data_dir is None else Path(data_dir)
    metadata = json.loads((root / "futian_metadata.json").read_text(encoding="utf-8"))
    if list(metadata["q_l"]) != list(DATE_TYPES):
        raise ValueError(f"q_l 的日期类型顺序应为 {DATE_TYPES}")
    business, applied = load_shared_business(root, business_overrides)
    with np.load(root / "futian_parameters.npz", allow_pickle=False) as stored:
        zone_ids = tuple(int(value) for value in stored["taz_ids"].tolist())
        distance = stored["distance"].astype(float)
        n_bar = stored["n_bar"].astype(float)
        mu = stored["mu"].astype(float)
        sigma_inv = stored["Sigma_inv"].astype(float)
        q_l = stored["q_l"].astype(float)
        radius_squared = float(stored["r_squared"])
        stored_samples = {
            name: stored[f"samples_{name}"].astype(float)
            for name in DATE_TYPES
        }
    zone_count = len(zone_ids)
    if distance.shape != (zone_count, zone_count) or n_bar.shape != (zone_count,):
        raise ValueError("数据包维度与 taz_ids 不一致")
    if q_l.shape != (len(DATE_TYPES),):
        raise ValueError("q_l 必须与日期类型一一对应")
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("距离矩阵必须有限且非负")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise ValueError("距离矩阵必须对称且对角线为 0")
    if abs(float(q_l.sum()) - 1.0) > 1e-8:
        raise ValueError("q_l 之和必须为 1")

    panel, labels = _training_panel(zone_ids, metadata)
    values = panel.to_numpy(dtype=float)
    inside = _support_mask(values, mu, sigma_inv, radius_squared)
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in panel.index]
    grouped_rows = {name: [] for name in DATE_TYPES}
    grouped_dates = {name: [] for name in DATE_TYPES}
    support_rows = {name: [] for name in DATE_TYPES}
    support_dates = {name: [] for name in DATE_TYPES}
    outside_dates = []
    for row, label, day, keep in zip(values, labels, dates, inside):
        grouped_rows[label].append(row.copy())
        grouped_dates[label].append(day)
        if keep:
            support_rows[label].append(row.copy())
            support_dates[label].append(day)
        else:
            outside_dates.append({"date": day, "date_type": label})
    for name in DATE_TYPES:
        expected = int(metadata["support_counts"][name])
        actual = np.vstack(support_rows[name])
        if len(actual) != expected or not np.allclose(actual, stored_samples[name]):
            raise ValueError(f"{name} 的 support 内样本与 futian_parameters.npz 不一致")
        count = len(grouped_rows[name])
        if abs(count / len(panel) - float(q_l[DATE_TYPES.index(name)])) > 1e-8:
            raise ValueError(f"{name} 的训练日频率与 npz 中的 q_l 不一致")

    if sample_set == "all":
        chosen_rows, chosen_dates = grouped_rows, grouped_dates
    else:
        chosen_rows, chosen_dates = support_rows, support_dates
    samples = {
        index: np.vstack(chosen_rows[name])
        for index, name in enumerate(DATE_TYPES)
    }
    gamma = business["transport_cost_per_distance"]
    return {
        "name": "futian",
        "sample_set": sample_set,
        "zone_ids": zone_ids,
        "hub_ids": zone_ids,
        "date_names": DATE_TYPES,
        "I_num": zone_count,
        "J_num": zone_count,
        "S_num": len(DATE_TYPES),
        "samples": samples,
        "sample_dates": chosen_dates,
        "q_l": q_l.tolist(),
        "f": [business["hub_open_cost"]] * zone_count,
        "c_m": business["mobile_unit_cost"],
        "c_f": business["fixed_unit_cost"],
        "rho": business["rho"],
        "b": business["shortage_penalty"],
        "tau": business["fixed_capacity"],
        "kappa": business["mobile_capacity"],
        "M_m": [FLEET_PER_HUB] * zone_count,
        "n_bar": n_bar,
        "M_z": np.full((zone_count, zone_count), FLEET_PER_HUB),
        "t_ij": 2.0 * gamma * distance,
        "record": {
            "demand_csv": str(DEMAND_CSV),
            "demand_field": "daily_volume",
            "parameter_dir": str(root),
            "train_start_inclusive": metadata["train_start_inclusive"],
            "train_end_exclusive": metadata["train_end_exclusive"],
            "training_days": int(len(panel)),
            "full_counts": {name: len(grouped_dates[name]) for name in DATE_TYPES},
            "support_counts": {name: len(support_dates[name]) for name in DATE_TYPES},
            "out_of_support_dates": outside_dates,
            "business_overrides": applied,
            "transport_cost_per_distance": gamma,
            "round_trip_factor": 2.0,
            "fleet_per_hub": FLEET_PER_HUB,
        },
    }


def format_summary(data: dict) -> str:
    record = data["record"]
    counts = {data["date_names"][index]: int(data["samples"][index].shape[0]) for index in range(data["S_num"])}
    lines = [
        f"实例 {data['name']}，区域数 {data['I_num']}，候选 hub 数 {data['J_num']}",
        f"本次样本：{data['sample_set']}，各日期类型天数 {counts}",
        f"训练窗 {record['train_start_inclusive']} 至 {record['train_end_exclusive']}，共 {record['training_days']} 天",
        f"全部训练日 {record['full_counts']}",
        f"support 内 {record['support_counts']}",
        f"support 外 {record['out_of_support_dates']}",
        f"q_l {dict(zip(data['date_names'], data['q_l']))}",
        (
            f"rho={data['rho']}，b={data['b']}，v^f={data['tau']}，v^m={data['kappa']}，"
            f"c^f={data['c_f']}，c^m={data['c_m']}，f={data['f'][0]}，"
            f"每个 hub 的移动桩上界={record['fleet_per_hub']}"
        ),
    ]
    return "\n".join(lines)

"""把一份已校准的福田数据包装成本地补能割平面使用的实例。

默认数据包是：

/Users/xingyueyi/Desktop/CG_solver/futian

需求样本、均值、协方差、椭球半径、Wasserstein 半径、q_l、fixed charger 上界
和距离都直接用该目录里的 npz，不重新估计。换数据时把另一个含有同样字段的
目录传给 load_zone_dataset，不要改割平面循环。

npz 里没有本地增量成本 g，也没有日期转移概率 p_kl。g 由调用者传入。
rho、b、tau、kappa、c^f、c^m、gamma 从同一目录的 business.json 读取。
p_kl 由 joint_historical_samples.csv 的 s_n 列、按相邻两天的频数计算。
这是联合概率，不是条件转移概率 P(l|k)。

区的顺序以 npz 里的 taz_ids 为准。zones = 0,1,2,... 的第 i 个位置就是 taz_ids[i]，
需求列、均值、协方差和距离矩阵都使用这个顺序。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

FUTIAN_DIR = Path(__file__).resolve().parents[1] / "futian"
JOINT_SAMPLES = Path(
    "/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/"
    "real_data_preprocessing/joint_historical_samples.csv"
)
SHARED_BUSINESS_FIELDS = (
    "rho", "shortage_penalty", "fixed_capacity", "mobile_capacity",
    "fixed_unit_cost", "mobile_unit_cost", "transport_cost_per_distance",
)

# 集中补能每个 hub 的车队上界。本地模式没有 hub，后面乘以区数得到 m 的上界。
MOBILE_FLEET_CAPACITY_PER_HUB = 30.0
DATE_TYPES = ("workday", "weekend", "holiday")


def load_shared_business(artifact_dir: Path, overrides=None) -> tuple[dict, dict, str]:
    """读取数据包目录里的 business.json。overrides 里非 None 的项只覆盖这一次。"""
    path = Path(artifact_dir) / "business.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少共享业务常数：{path}")
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
    return values, applied, str(path)


def _parameter_npz(artifact_dir: Path) -> Path:
    path = artifact_dir / "futian_parameters.npz"
    if path.is_file():
        return path
    raise FileNotFoundError(f"缺少参数文件 futian_parameters.npz：{artifact_dir}")


def _metadata_json(artifact_dir: Path) -> Path:
    path = artifact_dir / "futian_metadata.json"
    if path.is_file():
        return path
    raise FileNotFoundError(f"缺少 metadata futian_metadata.json：{artifact_dir}")


def load_zone_dataset(artifact_dir: Path, local_g: float, joint_samples: Path = JOINT_SAMPLES,
                      business_overrides=None) -> dict:
    """装入一个数据包目录。目录里要有参数 npz、同名 metadata 和 business.json。

    三仿射下包络要求 0 <= g <= rho+b。rho 与 b 来自 business.json，本次覆盖后的值也算。
    g=0 能通过这道检查，但目标里就没有本地补能溢价。
    gamma 是单程距离成本，这里不再乘 2。
    """
    artifact_dir = Path(artifact_dir)
    business, applied_overrides, business_path = load_shared_business(artifact_dir, business_overrides)
    rho = business["rho"]
    penalty = business["shortage_penalty"]
    if not 0.0 <= float(local_g) <= rho + penalty:
        raise ValueError("三仿射下包络要求 0 <= g <= rho+b")
    stored_path = _parameter_npz(artifact_dir)
    with np.load(stored_path, allow_pickle=False) as stored:
        taz_ids = tuple(int(value) for value in stored["taz_ids"].tolist())
        samples = {
            "workday": tuple(row.copy() for row in stored["samples_workday"]),
            "weekend": tuple(row.copy() for row in stored["samples_weekend"]),
            "holiday": tuple(row.copy() for row in stored["samples_holiday"]),
        }
        mean = stored["mu"].astype(float).copy()
        covariance = stored["Sigma"].astype(float).copy()
        # r 是共享椭球的 Mahalanobis 半径。epsilon_l 是各日期类型的 l_infinity 半径。
        radius = float(stored["r"])
        epsilon = {
            event: float(value)
            for event, value in zip(DATE_TYPES, stored["epsilon_l"].tolist())
        }
        q = {
            event: float(value)
            for event, value in zip(DATE_TYPES, stored["q_l"].tolist())
        }
        n_bar = [float(value) for value in stored["n_bar"].tolist()]
        distance = stored["distance"].astype(float).copy()
    zone_count = len(taz_ids)
    if distance.shape != (zone_count, zone_count) or mean.shape != (zone_count,):
        raise ValueError("数据包尺寸与 taz_ids 不一致")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise ValueError("距离矩阵必须对称且对角线为 0")
    precision = np.linalg.inv(covariance)
    for event, observations in samples.items():
        for sample_idx, demand in enumerate(observations):
            gap = float((demand - mean) @ precision @ (demand - mean))
            if gap > radius ** 2 + 1e-6:
                raise ValueError(f"样本 {(event, sample_idx)} 不在数据包给出的椭球内")
    metadata_path = _metadata_json(artifact_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    transitions = _date_type_transitions(joint_samples)
    return {
        "zones": tuple(range(zone_count)),
        "taz_ids": taz_ids,
        "date_types": DATE_TYPES,
        "samples": samples,
        "epsilon": epsilon,
        "mean": mean,
        "covariance": covariance,
        "radius": radius,
        "rho": rho,
        "penalty": penalty,
        "local_g": float(local_g),
        "tau": business["fixed_capacity"],
        "kappa": business["mobile_capacity"],
        "fixed_cost": business["fixed_unit_cost"],
        "mobile_cost": business["mobile_unit_cost"],
        "gamma": business["transport_cost_per_distance"],
        "distance": distance,
        "q": q,
        # 分母是这份 npz 里、已经落在椭球内的样本数，不是完整训练期天数。
        # q_l 仍用 npz 里按完整训练期算好的概率，不在这里重算。
        "sample_weight": {
            event: q[event] / len(samples[event]) for event in DATE_TYPES
        },
        "transition_probability": transitions,
        "n_bar": n_bar,
        # 【核对】600 = 20 个区 * 集中补能每个 hub 的上界 30。本地模式并没有 20 个 hub。
        "m_bar": MOBILE_FLEET_CAPACITY_PER_HUB * len(taz_ids),
        # 与 charge centrally 的 cut_tolerance 相同：样本利润上的绝对差。
        "separation_tolerance": 1e-4,
        "provenance": {
            "artifact": str(stored_path),
            "taz_ids": list(taz_ids),
            "in_support_counts": {event: len(samples[event]) for event in DATE_TYPES},
            "q_l": q,
            "epsilon_l": epsilon,
            "radius": radius,
            "covariance_condition_number": float(np.linalg.cond(covariance)),
            "selection": metadata.get("selection", {}),
            "m_bar_rule": "区数 * 每个 hub 的 mobile 上界 30",
            "business_path": business_path,
            "business_overrides": applied_overrides,
            "transition_probability": {
                f"{origin}->{destination}": probability
                for (origin, destination), probability in sorted(transitions.items())
            },
            "note": (
                "使用调用者指定目录里的已校准样本和歧义集。"
                "npz 没有日期转移概率，p_kl 由相邻日期类型频数计算。"
                "本地运输成本是 gamma 乘单程距离，不乘 centrally 使用的往返系数 2。"
            ),
        },
    }


def _date_type_transitions(joint_samples: Path = JOINT_SAMPLES) -> dict[tuple[str, str], float]:
    """p_kl 是相邻两天日期类型的经验联合概率，只保留 k != l。

    读的是 s_n 这一列，按文件中的前后两行配对。分母是配对数，也就是天数减 1。
    没有出现的转移不放进字典，主问题也就不为它建运输变量。
    """
    with Path(joint_samples).open(encoding="utf-8") as handle:
        labels = [row["s_n"] for row in csv.DictReader(handle)]
    counts = Counter(zip(labels[:-1], labels[1:]))
    total = len(labels) - 1
    return {
        (origin, destination): counts[origin, destination] / total
        for origin in DATE_TYPES
        for destination in DATE_TYPES
        if origin != destination and counts[origin, destination] > 0
    }

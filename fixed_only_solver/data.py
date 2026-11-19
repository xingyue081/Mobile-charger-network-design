"""读取和 charge centrally、charge locally 相同的福田数据包。

默认目录是 /Users/xingyueyi/Desktop/CG_solver/futian。那里的 npz 和
business.json 不在这里复制，也不重新估计均值、协方差、半径或 epsilon。

本模型只用固定桩。数据包里的移动桩容量、移动桩成本、运输成本和 hub
开设成本会被读出来核对来源，但不会进入变量、约束或目标。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FUTIAN_DIR = Path("/Users/xingyueyi/Desktop/CG_solver/futian")
DATE_TYPES = ("workday", "weekend", "holiday")

# business.json 里固定桩模型真正使用的四个数。
USED_BUSINESS_FIELDS = (
    "rho",
    "shortage_penalty",
    "fixed_capacity",
    "fixed_unit_cost",
)
# 同一份 business.json 里的移动桩参数。读它们只为确认数据包没换，模型不用。
EXCLUDED_BUSINESS_FIELDS = (
    "mobile_capacity",
    "mobile_unit_cost",
    "transport_cost_per_distance",
    "hub_open_cost",
)


def _parameter_npz(data_dir: Path) -> Path:
    """福田目录仍用 futian_parameters.npz。其他区用该目录里唯一的 *_parameters.npz。"""
    path = data_dir / "futian_parameters.npz"
    if path.is_file():
        return path
    matches = sorted(data_dir.glob("*_parameters.npz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"缺少参数文件 futian_parameters.npz：{data_dir}")


def load_dataset(data_dir: Path = FUTIAN_DIR, business_overrides=None) -> dict:
    """读取 data_dir 里的 futian_parameters.npz 和 business.json。

    npz 数组的日期顺序必须是 workday、weekend、holiday。
    第 i 行需求、mu、协方差和 n_bar 都对应 taz_ids[i]。
    """
    data_dir = Path(data_dir)
    stored_path = _parameter_npz(data_dir)
    business_path = data_dir / "business.json"
    business, applied_overrides = _load_business(business_path, business_overrides)

    with np.load(stored_path, allow_pickle=False) as stored:
        taz_ids = tuple(int(value) for value in stored["taz_ids"].tolist())
        samples = {
            event: stored[f"samples_{event}"].astype(float).copy()
            for event in DATE_TYPES
        }
        mean = stored["mu"].astype(float).copy()
        covariance = stored["Sigma"].astype(float).copy()
        precision = stored["Sigma_inv"].astype(float).copy()
        radius = float(stored["r"])
        radius_squared = float(stored["r_squared"])
        epsilon = {
            event: float(value)
            for event, value in zip(DATE_TYPES, stored["epsilon_l"].tolist(), strict=True)
        }
        probability = {
            event: float(value)
            for event, value in zip(DATE_TYPES, stored["q_l"].tolist(), strict=True)
        }
        fixed_upper = _integer_bounds(stored["n_bar"].astype(float))
        distance = stored["distance"].astype(float).copy()

    zone_count = len(taz_ids)
    _check_shapes(zone_count, mean, covariance, precision, distance, samples)
    _check_ambiguity_parameters(probability, epsilon, radius, radius_squared)
    _check_nominal_samples(samples, mean, precision, radius)
    # 本地补能用 Sigma 的 Cholesky 写同一颗椭球。这里先确认 Sigma 正定，
    # 避免两套写法一个能分解、另一个不能。
    np.linalg.cholesky(covariance)

    sample_count = {event: int(samples[event].shape[0]) for event in DATE_TYPES}
    return {
        "zones": tuple(range(zone_count)),
        "taz_ids": taz_ids,
        "date_types": DATE_TYPES,
        "samples": samples,
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "radius": radius,
        "epsilon": epsilon,
        "probability": probability,
        # 权重是 q_l / N_{l,Omega}。N 是这份 npz 里已经落在椭球内的样本数。
        # q_l 沿用 npz，不按样本数重算。样本筛选已经做过一次，重算会改变日期权重。
        "sample_weight": {
            event: probability[event] / sample_count[event] for event in DATE_TYPES
        },
        "rho": business["rho"],
        "penalty": business["shortage_penalty"],
        "tau": business["fixed_capacity"],
        "fixed_cost": business["fixed_unit_cost"],
        "n_bar": fixed_upper,
        "provenance": {
            "data_dir": str(data_dir.resolve()),
            "parameters": str(stored_path.resolve()),
            "business": str(business_path.resolve()),
            "taz_ids": list(taz_ids),
            "sample_count": sample_count,
            "probability": probability,
            "epsilon": epsilon,
            "radius": radius,
            "business_overrides": applied_overrides,
            "excluded_business_fields": {
                field: business[field] for field in EXCLUDED_BUSINESS_FIELDS
            },
            "distance_shape": list(distance.shape),
            "note": (
                "歧义集使用这份 npz 的样本、mu、Sigma_inv、r 和 epsilon_l。"
                "距离矩阵和移动桩业务参数只做来源核对，fixed-only 模型不使用。"
            ),
        },
    }


def local_box(instance: dict, event: str, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """样本周围的 l_infinity 盒子，下界截到 0。

    第 i 个坐标是 [max(0, d_i - epsilon_l), d_i + epsilon_l]。
    这是局部集合的盒子部分。共享椭球在分离和支持函数里再与它取交集。
    """
    center = np.asarray(instance["samples"][event][sample_idx], dtype=float)
    epsilon = float(instance["epsilon"][event])
    lower = np.maximum(0.0, center - epsilon)
    upper = center + epsilon
    if np.any(upper < lower):
        raise ValueError(f"空的局部盒子: {(event, sample_idx)}")
    return lower, upper


def sample_keys(instance: dict):
    """按日期类型、样本下标遍历。下标是该日期类型数组里的行号。"""
    for event in instance["date_types"]:
        for sample_idx in range(instance["samples"][event].shape[0]):
            yield event, sample_idx


def _load_business(path: Path, overrides) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少共享业务常数：{path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    required = USED_BUSINESS_FIELDS + EXCLUDED_BUSINESS_FIELDS
    missing = [field for field in required if field not in stored]
    if missing:
        raise ValueError(f"business.json 缺少 {missing}")
    values = {field: float(stored[field]) for field in required}
    applied = {}
    for field, value in (overrides or {}).items():
        if value is None:
            continue
        if field not in USED_BUSINESS_FIELDS:
            raise ValueError(f"fixed-only 不能覆盖或使用业务常数 {field}")
        values[field] = float(value)
        applied[field] = float(value)
    if values["rho"] < 0 or values["shortage_penalty"] < 0 or values["fixed_capacity"] <= 0:
        raise ValueError("rho、shortage_penalty 不能为负，fixed_capacity 必须为正")
    if values["fixed_unit_cost"] < 0:
        raise ValueError("fixed_unit_cost 不能为负")
    return values, applied


def _integer_bounds(values: np.ndarray) -> list[int]:
    if np.any(values < -1e-8) or not np.allclose(values, np.round(values)):
        raise ValueError("n_bar 必须是非负整数上界")
    return [int(round(float(value))) for value in values]


def _check_shapes(zone_count, mean, covariance, precision, distance, samples) -> None:
    if zone_count == 0:
        raise ValueError("数据包没有区域")
    expected = (zone_count, zone_count)
    if mean.shape != (zone_count,) or covariance.shape != expected or precision.shape != expected:
        raise ValueError("均值或协方差尺寸与 taz_ids 不一致")
    if distance.shape != expected:
        raise ValueError("距离矩阵尺寸与 taz_ids 不一致")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise ValueError("距离矩阵必须对称且对角线为 0")
    for event, rows in samples.items():
        if rows.ndim != 2 or rows.shape[1] != zone_count or rows.shape[0] == 0:
            raise ValueError(f"{event} 样本尺寸不正确")
        if not np.isfinite(rows).all() or np.any(rows < -1e-8):
            raise ValueError(f"{event} 样本必须有限且非负")


def _check_ambiguity_parameters(probability, epsilon, radius, radius_squared) -> None:
    if any(value <= 0 for value in probability.values()):
        raise ValueError("q_l 必须为正")
    if abs(sum(probability.values()) - 1.0) > 1e-8:
        raise ValueError("q_l 之和必须为 1，不能在求解时另做归一化")
    if any(value < 0 for value in epsilon.values()):
        raise ValueError("epsilon_l 不能为负")
    if radius <= 0:
        raise ValueError("椭球半径 r 必须为正")
    if abs(radius ** 2 - radius_squared) > 1e-8:
        raise ValueError("npz 中的 r 与 r_squared 不一致")


def _check_nominal_samples(samples, mean, precision, radius) -> None:
    """名义样本本身必须落在共享椭球里，否则盒子与椭球的交可能不含该样本。"""
    radius_squared = radius ** 2
    for event, rows in samples.items():
        centered = rows - mean
        quadratic = np.einsum("ni,ij,nj->n", centered, precision, centered)
        if np.any(quadratic > radius_squared + 1e-6):
            raise ValueError(f"{event} 有名义样本不在数据包给出的椭球内")

"""用解析例子和全枚举核对 fixed-only，并检查福田数据包能被读入。

这里不求解福田。福田只核对样本数、半径、epsilon、q_l，
以及一个容量为 0 的样本能在同一颗椭球上完成分离。
"""

from __future__ import annotations

import json
from itertools import product

import numpy as np

from data import FUTIAN_DIR, load_dataset, local_box
from master import solve_master
from separation import piecewise_profit, support_value, worst_operating_profit
from solve import evaluate_plan, solve_fixed_only


def run_self_check() -> None:
    _check_box_only_support()
    _check_ellipsoid_binds()
    _check_cutting_plane_against_enumeration()
    _check_futian_package()
    print("fixed-only 自检通过")


def _check_box_only_support() -> None:
    """椭球足够大时，支持函数取到盒子端点。下界被截到 0。"""
    instance = _one_zone(radius=100.0)
    lower, upper = local_box(instance, "workday", 0)
    if abs(lower[0]) > 1e-8 or abs(upper[0] - 20.0) > 1e-8:
        raise AssertionError(f"盒子下界应截到 0、上界应为 20，实际是 {(lower[0], upper[0])}")
    served = support_value((0,), lower, upper, instance)
    shortage = support_value((1,), lower, upper, instance)
    if abs(served) > 1e-6 or abs(shortage - 40.0) > 1e-6:
        raise AssertionError(f"盒子支持函数不正确: served={served}, shortage={shortage}")


def _check_ellipsoid_binds() -> None:
    """盒子比椭球大时，最坏需求停在 Mahalanobis 半径上。"""
    instance = _one_zone(radius=1.0)
    lower, upper = local_box(instance, "workday", 0)
    shortage = support_value((1,), lower, upper, instance)
    # 方向是 b=2，椭球把 D 限制在 1 以内，所以 h=2。
    if abs(shortage - 2.0) > 1e-5:
        raise AssertionError(f"椭球上的缺电支持函数应为 2，实际是 {shortage}")
    uncovered = worst_operating_profit(np.array([0.0]), "workday", 0, instance)
    covered = worst_operating_profit(np.array([10.0]), "workday", 0, instance)
    if abs(uncovered["profit"] + 2.0) > 1e-5 or abs(covered["profit"]) > 1e-5:
        raise AssertionError(
            f"一区最坏利润不正确: uncovered={uncovered['profit']}, covered={covered['profit']}"
        )


def _check_cutting_plane_against_enumeration() -> None:
    """两区全枚举的最优目标必须和割平面一致，并且割平面会加入混合 pattern。"""
    instance = _two_zone()
    best_objective = None
    for counts in product(range(instance["n_bar"][0] + 1), range(instance["n_bar"][1] + 1)):
        objective = evaluate_plan(instance, list(counts))["objective"]
        if best_objective is None or objective > best_objective:
            best_objective = objective
    # 1e-4 与正式求解的加割容差相同。支持函数和分离的数值差大约在 1e-6，
    # 更小的容差会把这个差当成“已有割仍被违反”。
    result = solve_fixed_only(instance, mip_gap=1e-8, cut_tolerance=1e-4)
    if not result["converged"]:
        raise AssertionError(f"两区例子没有收敛: {result['reason']}")
    if abs(result["objective"] - best_objective) > 1e-4:
        raise AssertionError(
            f"割平面目标 {result['objective']} 与全枚举 {best_objective} 不一致"
        )
    sample_count = sum(rows.shape[0] for rows in instance["samples"].values())
    if result["pattern_count"] <= 2 * sample_count:
        raise AssertionError("割平面没有加入全部满足、全部缺电之外的混合 pattern")
    if result["decision"]["mobile_chargers"] != 0:
        raise AssertionError("fixed-only 结果里出现了移动桩")
    _check_master_has_only_fixed_chargers(instance)


def _check_master_has_only_fixed_chargers(instance: dict) -> None:
    from data import sample_keys
    from solve import _store_support

    active = {}
    supports = {}
    for key in sample_keys(instance):
        patterns = [
            tuple(0 for _ in instance["zones"]),
            tuple(1 for _ in instance["zones"]),
        ]
        active[key] = patterns
        for pattern in patterns:
            _store_support(supports, instance, key, pattern)
    solved = solve_master(instance, active, supports, {"mip_gap": 1e-8})
    if not solved["solvable"]:
        raise AssertionError("两区主问题不可行")
    for name in solved["variable_names"]:
        # C 是容量提成变量，等式钉死为 tau*n，不是新的业务决策。
        if not (name.startswith("n") or name.startswith("eta") or name.startswith("C")):
            raise AssertionError(f"主问题出现固定桩、容量提成和 eta 以外的变量 {name}")


def _check_futian_package() -> None:
    instance = load_dataset(FUTIAN_DIR)
    counts = instance["provenance"]["sample_count"]
    if counts != {"workday": 82, "weekend": 28, "holiday": 10}:
        raise AssertionError(f"福田样本数不是数据包中的 82/28/10: {counts}")
    metadata = json.loads((FUTIAN_DIR / "futian_metadata.json").read_text(encoding="utf-8"))
    for event, value in instance["epsilon"].items():
        if abs(value - metadata["epsilon_l"][event]) > 1e-8:
            raise AssertionError(f"{event} 的 epsilon 与 metadata 不一致")
    for event, value in instance["probability"].items():
        if abs(value - metadata["q_l"][event]) > 1e-8:
            raise AssertionError(f"{event} 的 q_l 与 metadata 不一致")
    if abs(instance["radius"] - metadata["r"]) > 1e-8:
        raise AssertionError("椭球半径与 metadata 不一致")
    business = json.loads((FUTIAN_DIR / "business.json").read_text(encoding="utf-8"))
    if (
        instance["rho"] != business["rho"]
        or instance["penalty"] != business["shortage_penalty"]
        or instance["tau"] != business["fixed_capacity"]
        or instance["fixed_cost"] != business["fixed_unit_cost"]
    ):
        raise AssertionError("固定桩参数没有按 business.json 原值读入")
    excluded = instance["provenance"]["excluded_business_fields"]
    if (
        excluded["mobile_unit_cost"] != business["mobile_unit_cost"]
        or excluded["mobile_capacity"] != business["mobile_capacity"]
        or excluded["hub_open_cost"] != business["hub_open_cost"]
        or excluded["transport_cost_per_distance"] != business["transport_cost_per_distance"]
    ):
        raise AssertionError("移动桩参数没有从同一份 business.json 读到核对值")
    capacity = np.zeros(len(instance["zones"]))
    separated = worst_operating_profit(capacity, "workday", 0, instance)
    direct = piecewise_profit(separated["demand"], capacity, instance["rho"], instance["penalty"])
    if abs(direct - separated["profit"]) > 1e-4:
        raise AssertionError("福田样本的分离目标与两段利润不一致")
    lower, upper = local_box(instance, "workday", 0)
    demand = separated["demand"]
    if np.any(demand < lower - 1e-5) or np.any(demand > upper + 1e-5):
        raise AssertionError("福田分离需求跑出了 l_infinity 盒子")
    centered = demand - instance["mean"]
    quadratic = float(centered @ instance["precision"] @ centered)
    if quadratic > instance["radius"] ** 2 + 1e-4:
        raise AssertionError("福田分离需求跑出了共享椭球")


def _one_zone(radius: float) -> dict:
    """epsilon=10。半径很大时中心放在 10，盒子是 [0, 20]；半径为 1 时中心是 0，盒子是 [0, 10]。"""
    covariance = np.array([[1.0]])
    center = 10.0 if radius > 10.0 else 0.0
    samples = {
        event: np.array([[center]]) for event in ("workday", "weekend", "holiday")
    }
    return _finish_instance(
        mean=np.array([center]),
        covariance=covariance,
        samples=samples,
        radius=radius,
        epsilon=10.0,
        n_bar=[2],
        tau=10.0,
        fixed_cost=1.0,
        probability={"workday": 1.0, "weekend": 0.0, "holiday": 0.0},
    )


def _two_zone() -> dict:
    """区 0 的需求盒子低于一台固定桩，区 1 的需求盒子高于两台固定桩。

    因此最坏分段会在区 0 取满足、在区 1 取缺电，而不是两个极端 pattern。
    """
    covariance = np.eye(2)
    samples = {
        "workday": np.array([[5.0, 50.0], [5.2, 49.5]]),
        "weekend": np.array([[4.8, 50.4]]),
        "holiday": np.array([[5.1, 49.8]]),
    }
    return _finish_instance(
        mean=np.array([5.0, 50.0]),
        covariance=covariance,
        samples=samples,
        radius=5.0,
        epsilon=1.0,
        n_bar=[2, 2],
        tau=10.0,
        fixed_cost=1.0,
        probability={"workday": 0.7, "weekend": 0.2, "holiday": 0.1},
    )


def _finish_instance(mean, covariance, samples, radius, epsilon, n_bar, tau,
                     fixed_cost, probability) -> dict:
    precision = np.linalg.inv(covariance)
    date_types = ("workday", "weekend", "holiday")
    sample_count = {event: int(samples[event].shape[0]) for event in date_types}
    taz_ids = tuple(range(len(mean)))
    return {
        "zones": taz_ids,
        "taz_ids": taz_ids,
        "date_types": date_types,
        "samples": samples,
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "radius": radius,
        "epsilon": {event: epsilon for event in date_types},
        "probability": probability,
        "sample_weight": {
            event: probability[event] / sample_count[event] for event in date_types
        },
        "rho": 2.0,
        "penalty": 2.0,
        "tau": tau,
        "fixed_cost": fixed_cost,
        "n_bar": n_bar,
    }

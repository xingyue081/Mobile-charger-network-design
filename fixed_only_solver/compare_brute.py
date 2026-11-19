"""用 5 个 toy example 比较全枚举和 fixed-only 割平面。

暴力求解做两件与割平面不同的事：

1. 外层枚举每一个可行的整数 n，不经过主问题。
2. 内层对每个样本枚举全部 2^I 个分段。每个分段只解一次凸的支持函数，
   最坏利润取这些分段的最小值。

割平面仍调用 solve_fixed_only。它的分离模型含缺电二元变量，
不读取这里预先算好的全部分段。

一区例子还可以不用 Gurobi，直接在盒子与椭球交集的两个端点上计算利润。
这个函数在区间内先升后降，最小值一定在端点。
"""

from __future__ import annotations

import contextlib
import io
from itertools import product

import numpy as np

from data import local_box, sample_keys
from separation import pattern_cut_value, piecewise_profit, support_value, worst_operating_profit
from solve import solve_fixed_only

DATE_TYPES = ("workday", "weekend", "holiday")
OBJECTIVE_TOLERANCE = 1e-4


def main() -> None:
    rows = []
    for name, instance in _toys():
        brute = _enumerate_plans(instance)
        # 紧间隙：核对容量提成 + 热启动 + 未证明不加割。
        tight = _run_cutting(instance, adaptive_gap=False, mip_gap=1e-8)
        # 自适应间隙：再走一遍松→紧路径，目标仍须与暴力枚举一致。
        adaptive = _run_cutting(instance, adaptive_gap=True, mip_gap=1e-8)
        for mode, cutting in (("tight", tight), ("adaptive", adaptive)):
            row = _compare_one(name, mode, instance, brute, cutting)
            rows.append(row)
    _print_report(rows)
    if not all(row["consistent"] for row in rows):
        raise SystemExit(1)


def _run_cutting(instance: dict, *, adaptive_gap: bool, mip_gap: float) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return solve_fixed_only(
            instance,
            mip_gap=mip_gap,
            cut_tolerance=OBJECTIVE_TOLERANCE,
            adaptive_gap=adaptive_gap,
        )


def _compare_one(name: str, mode: str, instance: dict, brute: dict, cutting: dict) -> dict:
    chosen = None if cutting["decision"] is None else tuple(cutting["decision"]["n"])
    chosen_objective = None if chosen is None else _objective(instance, chosen, brute["supports"])
    inner_gap = None if chosen is None else _separation_gap(instance, chosen, brute["supports"])
    endpoint_gap = _endpoint_gap(instance, brute)
    optimal = chosen is not None and any(chosen == plan for plan in brute["optimal_plans"])
    objective_gap = None
    if cutting["objective"] is not None:
        objective_gap = abs(cutting["objective"] - brute["objective"])
    consistent = (
        cutting["converged"]
        and optimal
        and objective_gap is not None
        and objective_gap <= OBJECTIVE_TOLERANCE
        and chosen_objective is not None
        and abs(chosen_objective - brute["objective"]) <= OBJECTIVE_TOLERANCE
        and inner_gap is not None
        and inner_gap <= 1e-4
        and (endpoint_gap is None or endpoint_gap <= 1e-5)
    )
    return {
        "name": f"{name} [{mode}]",
        "consistent": consistent,
        "brute_n": brute["optimal_plans"],
        "brute_objective": brute["objective"],
        "cutting_n": chosen,
        "cutting_objective": cutting["objective"],
        "objective_gap": objective_gap,
        "inner_gap": inner_gap,
        "endpoint_gap": endpoint_gap,
        "iterations": cutting["iterations"],
        "patterns": cutting["pattern_count"],
        "reason": cutting["reason"],
        "plans": brute["plan_count"],
    }


def _enumerate_plans(instance: dict) -> dict:
    """枚举 n，并用全部分段的支持函数计算每个方案的目标。"""
    supports = _all_supports(instance)
    ranges = [range(int(bound) + 1) for bound in instance["n_bar"]]
    best = None
    optimal = []
    plan_count = 0
    for counts in product(*ranges):
        plan_count += 1
        objective = _objective(instance, counts, supports)
        if best is None or objective > best + 1e-8:
            best = objective
            optimal = [counts]
        elif abs(objective - best) <= 1e-8:
            optimal.append(counts)
    return {
        "objective": best,
        "optimal_plans": optimal,
        "supports": supports,
        "plan_count": plan_count,
    }


def _all_supports(instance: dict) -> dict:
    """每个样本、每个分段各算一次 h。h 与 n 无关，所以可以先算完。"""
    zone_count = len(instance["zones"])
    supports = {}
    for event, sample_idx in sample_keys(instance):
        lower, upper = local_box(instance, event, sample_idx)
        for pattern in product((0, 1), repeat=zone_count):
            supports[event, sample_idx, pattern] = support_value(pattern, lower, upper, instance)
    return supports


def _objective(instance: dict, counts, supports: dict) -> float:
    capacity = float(instance["tau"]) * np.asarray(counts, dtype=float)
    operating = 0.0
    for event, sample_idx in sample_keys(instance):
        operating += float(instance["sample_weight"][event]) * _worst_profit(
            instance, capacity, event, sample_idx, supports
        )
    return -float(instance["fixed_cost"]) * sum(counts) + operating


def _worst_profit(instance: dict, capacity, event: str, sample_idx: int, supports: dict) -> float:
    zone_count = len(capacity)
    return min(
        pattern_cut_value(pattern, capacity, supports[event, sample_idx, pattern], instance)
        for pattern in product((0, 1), repeat=zone_count)
    )


def _separation_gap(instance: dict, counts, supports: dict) -> float:
    """割平面使用的分离目标，与全部分段最小值之间的最大差。"""
    capacity = float(instance["tau"]) * np.asarray(counts, dtype=float)
    gap = 0.0
    for event, sample_idx in sample_keys(instance):
        separated = worst_operating_profit(capacity, event, sample_idx, instance)
        direct = piecewise_profit(
            separated["demand"], capacity, instance["rho"], instance["penalty"]
        )
        enumerated = _worst_profit(instance, capacity, event, sample_idx, supports)
        gap = max(gap, abs(separated["profit"] - enumerated), abs(separated["profit"] - direct))
    return gap


def _endpoint_gap(instance: dict, brute: dict) -> float | None:
    """一区时，用区间端点重算最优目标。多区没有这个闭式，返回 None。"""
    if len(instance["zones"]) != 1:
        return None
    ranges = [range(int(bound) + 1) for bound in instance["n_bar"]]
    best = None
    for counts in product(*ranges):
        capacity = float(instance["tau"]) * float(counts[0])
        operating = 0.0
        for event, sample_idx in sample_keys(instance):
            low, high = _feasible_interval(instance, event, sample_idx)
            profit = min(
                piecewise_profit(np.array([low]), np.array([capacity]), instance["rho"], instance["penalty"]),
                piecewise_profit(np.array([high]), np.array([capacity]), instance["rho"], instance["penalty"]),
            )
            operating += float(instance["sample_weight"][event]) * profit
        objective = -float(instance["fixed_cost"]) * counts[0] + operating
        if best is None or objective > best:
            best = objective
    return abs(best - brute["objective"])


def _feasible_interval(instance: dict, event: str, sample_idx: int) -> tuple[float, float]:
    lower, upper = local_box(instance, event, sample_idx)
    half_width = float(instance["radius"]) / np.sqrt(float(instance["precision"][0, 0]))
    low = max(float(lower[0]), float(instance["mean"][0]) - half_width)
    high = min(float(upper[0]), float(instance["mean"][0]) + half_width)
    if low > high + 1e-8:
        raise ValueError(f"一区局部集合为空: {(event, sample_idx)}")
    return low, high


def _print_report(rows: list[dict]) -> None:
    print("五个 toy example：暴力枚举 vs 割平面")
    for row in rows:
        status = "一致" if row["consistent"] else "不一致"
        print(f"\n{row['name']}: {status}")
        print(f"  枚举方案数 {row['plans']}，割平面迭代 {row['iterations']}，使用分段 {row['patterns']}")
        print(f"  暴力最优 n { _format_plans(row['brute_n']) }，目标 {row['brute_objective']:.8f}")
        print(f"  割平面 n {row['cutting_n']}，目标 {row['cutting_objective']}")
        print(
            f"  目标差 {row['objective_gap']}，分离与全分段差 {row['inner_gap']}，"
            f"一区端点差 {row['endpoint_gap']}"
        )
        if not row["consistent"]:
            print(f"  停止原因: {row['reason']}")


def _format_plans(plans: list[tuple]) -> str:
    shown = ", ".join(str(list(plan)) for plan in plans[:6])
    if len(plans) > 6:
        return f"{shown} 等 {len(plans)} 个"
    return shown


def _toys() -> list[tuple[str, dict]]:
    return [
        ("1 单区高成本，最优不建桩", _high_cost()),
        ("2 单区椭球收紧，最优建桩数在内部", _interior_ellipsoid()),
        ("3 两区负相关，椭球切掉盒子角落", _negative_correlation()),
        ("4 需求下界截到 0，且 rho 不等于 b", _truncated_demand()),
        ("5 三区三种日期半径，上界绑住高需求区", _three_zones()),
    ]


def _high_cost() -> dict:
    """一台桩的容量价值不超过 (rho+b)*tau=20，单价是 100。"""
    return _instance(
        mean=[10.0],
        covariance=[[1.0]],
        samples={
            "workday": [[10.0], [10.4]],
            "weekend": [[9.7]],
            "holiday": [[10.2]],
        },
        radius=3.0,
        epsilon={"workday": 2.0, "weekend": 2.0, "holiday": 2.0},
        probability={"workday": 0.5, "weekend": 0.3, "holiday": 0.2},
        rho=2.0,
        penalty=2.0,
        tau=5.0,
        fixed_cost=100.0,
        n_bar=[4],
    )


def _interior_ellipsoid() -> dict:
    """盒子比椭球宽。可行需求约在 [18, 22]，tau=2，单价 6。

    容量低于 20 时，最坏利润随容量上升；达到 20 后最坏需求改走下端，
    再加桩不再增加最坏利润。因此最优点不在 0，也不在上界。
    """
    return _instance(
        mean=[20.0],
        covariance=[[1.0]],
        samples={
            "workday": [[20.0], [19.2]],
            "weekend": [[20.5]],
            "holiday": [[19.6]],
        },
        radius=2.0,
        epsilon={"workday": 10.0, "weekend": 10.0, "holiday": 10.0},
        probability={"workday": 0.5, "weekend": 0.3, "holiday": 0.2},
        rho=2.0,
        penalty=2.0,
        tau=2.0,
        fixed_cost=6.0,
        n_bar=[15],
    )


def _negative_correlation() -> dict:
    """两区需求负相关，半径只比样本稍大，四个盒子角落不能同时取到。"""
    return _instance(
        mean=[12.0, 28.0],
        covariance=[[5.0, -3.5], [-3.5, 8.0]],
        samples={
            "workday": [[12.0, 28.0], [12.3, 27.6]],
            "weekend": [[11.8, 28.3]],
            "holiday": [[12.2, 27.8]],
        },
        radius=1.2,
        epsilon={"workday": 12.0, "weekend": 12.0, "holiday": 12.0},
        probability={"workday": 0.7, "weekend": 0.2, "holiday": 0.1},
        rho=2.0,
        penalty=3.0,
        tau=6.0,
        fixed_cost=10.0,
        n_bar=[4, 6],
    )


def _truncated_demand() -> dict:
    """第一区名义需求小于 epsilon，盒子下界被截成 0。rho 与 b 取不同的数。"""
    return _instance(
        mean=[1.0, 22.0],
        covariance=[[1.0, 0.2], [0.2, 4.0]],
        samples={
            "workday": [[1.0, 22.0], [1.4, 21.0]],
            "weekend": [[0.8, 23.0]],
            "holiday": [[1.2, 20.5]],
        },
        radius=2.5,
        epsilon={"workday": 5.0, "weekend": 6.0, "holiday": 4.0},
        probability={"workday": 0.6, "weekend": 0.25, "holiday": 0.15},
        rho=3.0,
        penalty=1.0,
        tau=4.0,
        fixed_cost=8.0,
        n_bar=[3, 8],
    )


def _three_zones() -> dict:
    """三个区、三种 epsilon。高需求区的 n 上界只有 2，容量盖不住需求盒子。"""
    return _instance(
        mean=[8.0, 40.0, 15.0],
        covariance=[[4.0, 0.5, 0.3], [0.5, 9.0, -0.4], [0.3, -0.4, 4.0]],
        samples={
            "workday": [[8.0, 40.0, 15.0], [8.4, 39.2, 15.3]],
            "weekend": [[7.6, 40.8, 14.7]],
            "holiday": [[8.2, 39.5, 15.4]],
        },
        radius=1.5,
        epsilon={"workday": 2.0, "weekend": 4.0, "holiday": 1.0},
        probability={"workday": 0.6, "weekend": 0.3, "holiday": 0.1},
        rho=2.0,
        penalty=4.0,
        tau=10.0,
        fixed_cost=20.0,
        n_bar=[3, 2, 3],
    )


def _instance(mean, covariance, samples, radius, epsilon, probability, rho, penalty,
              tau, fixed_cost, n_bar) -> dict:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    np.linalg.cholesky(covariance)
    precision = np.linalg.inv(covariance)
    stored = {event: np.asarray(rows, dtype=float) for event, rows in samples.items()}
    if abs(sum(probability.values()) - 1.0) > 1e-12:
        raise ValueError("toy 的 q_l 之和必须为 1")
    for event, rows in stored.items():
        if rows.ndim != 2 or rows.shape[1] != len(mean) or rows.shape[0] == 0:
            raise ValueError(f"{event} 样本尺寸不正确")
        centered = rows - mean
        quadratic = np.einsum("ni,ij,nj->n", centered, precision, centered)
        if np.any(quadratic > radius ** 2 + 1e-8):
            raise ValueError(f"{event} 有样本不在椭球内: {quadratic}")
    counts = {event: int(stored[event].shape[0]) for event in DATE_TYPES}
    return {
        "zones": tuple(range(len(mean))),
        "taz_ids": tuple(range(len(mean))),
        "date_types": DATE_TYPES,
        "samples": stored,
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "radius": float(radius),
        "epsilon": {event: float(epsilon[event]) for event in DATE_TYPES},
        "probability": {event: float(probability[event]) for event in DATE_TYPES},
        "sample_weight": {
            event: float(probability[event]) / counts[event] for event in DATE_TYPES
        },
        "rho": float(rho),
        "penalty": float(penalty),
        "tau": float(tau),
        "fixed_cost": float(fixed_cost),
        "n_bar": [int(value) for value in n_bar],
    }


if __name__ == "__main__":
    main()

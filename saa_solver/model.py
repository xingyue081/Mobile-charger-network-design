"""Charge-centrally 的样本平均近似。

对应已知分布模型 (SP)。日期类型 l 的条件期望换成该类型样本的算术平均：

    max  -sum_j f_j x_j - c^m sum_j m_j - c^f sum_i n_i
         + sum_l q_l (1/N_l) sum_k pi^C(n, z_l, d^{l,k})

    pi^C = sum_i p d_i
         - sum_i (p+b) (d_i - v^f n_i - v^m sum_j z_ijl)^+
         - sum_{i,j} t_ij z_ijl

代码里的 rho、b、tau、kappa 分别是 p、b、v^f、v^m。
同一个 z_l 用于日期类型 l 的全部样本：先看到日期类型，再实现需求。
u_{ilk} 表示 zone i、日期类型 l、第 k 个样本的未满足需求。
p+b > 0 且目标是最大化时，u 会取到上面的正部。
"""

from __future__ import annotations

import itertools

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    return names.get(status, str(status))


def operating_profit(n, z, data: dict) -> float:
    """用容量直接计算 sum_l q_l 乘以日期类型 l 的平均日运营利润。"""
    installed = np.asarray(n, dtype=float)
    dispatch = np.asarray(z, dtype=float)
    total = 0.0
    for date_index in range(data["S_num"]):
        demand = np.asarray(data["samples"][date_index], dtype=float)
        capacity = data["tau"] * installed + data["kappa"] * dispatch[:, :, date_index].sum(axis=1)
        unmet = np.maximum(demand - capacity, 0.0)
        transport = float(np.sum(data["t_ij"] * dispatch[:, :, date_index]))
        per_day = (
            data["rho"] * demand.sum(axis=1)
            - (data["rho"] + data["b"]) * unmet.sum(axis=1)
            - transport
        )
        total += float(data["q_l"][date_index]) * float(per_day.mean())
    return total


def investment_cost(x, m, n, data: dict) -> float:
    opened = np.asarray(x, dtype=float)
    fleet = np.asarray(m, dtype=float)
    installed = np.asarray(n, dtype=float)
    return float(np.dot(data["f"], opened) + data["c_m"] * fleet.sum() + data["c_f"] * installed.sum())


def _as_int(value: float, name: str) -> int:
    rounded = int(round(value))
    if abs(value - rounded) > 1e-4:
        raise RuntimeError(f"{name} 的解 {value} 离整数超过 1e-4")
    return rounded


def solve_saa(data: dict, solver_limits=None) -> dict:
    """求解 charge-centrally SAA。可行域与集中补能主问题的物理约束相同。"""
    if data["rho"] + data["b"] < 0:
        raise ValueError("p+b 为负时，未满足需求的线性化不能用于这个最大化模型")
    limits = {} if solver_limits is None else solver_limits
    zone_count, hub_count, date_count = data["I_num"], data["J_num"], data["S_num"]
    zones, hubs, dates = range(zone_count), range(hub_count), range(date_count)
    samples = data["samples"]
    sample_counts = {date_index: int(samples[date_index].shape[0]) for date_index in dates}
    if any(count < 1 for count in sample_counts.values()):
        raise ValueError("每个日期类型至少要有一个样本")

    model = gp.Model("Charge_Centrally_SAA")
    model.Params.OutputFlag = 1 if limits.get("output_flag") else 0
    if limits.get("mip_gap") is not None:
        model.Params.MIPGap = float(limits["mip_gap"])
    if limits.get("time_limit") is not None:
        model.Params.TimeLimit = float(limits["time_limit"])

    opened = model.addVars(hubs, vtype=GRB.BINARY, name="x")
    assigned = model.addVars(zones, hubs, vtype=GRB.BINARY, name="y")
    fleet = model.addVars(hubs, vtype=GRB.INTEGER, name="m")
    installed = model.addVars(zones, vtype=GRB.INTEGER, name="n")
    dispatch = model.addVars(zones, hubs, dates, vtype=GRB.INTEGER, name="z")
    unmet = model.addVars(
        [
            (zone, date_index, sample_index)
            for date_index in dates
            for sample_index in range(sample_counts[date_index])
            for zone in zones
        ],
        lb=0.0,
        name="u",
    )

    for hub in hubs:
        for date_index in dates:
            model.addConstr(gp.quicksum(dispatch[zone, hub, date_index] for zone in zones) <= fleet[hub])
        model.addConstr(fleet[hub] <= data["M_m"][hub] * opened[hub])
    for zone in zones:
        model.addConstr(gp.quicksum(assigned[zone, hub] for hub in hubs) <= 1)
        model.addConstr(installed[zone] <= data["n_bar"][zone])
        for hub in hubs:
            model.addConstr(assigned[zone, hub] <= opened[hub])
            for date_index in dates:
                model.addConstr(
                    dispatch[zone, hub, date_index] <= data["M_z"][zone, hub] * assigned[zone, hub]
                )
    for date_index in dates:
        demand = samples[date_index]
        for sample_index in range(sample_counts[date_index]):
            for zone in zones:
                capacity = data["tau"] * installed[zone] + data["kappa"] * gp.quicksum(
                    dispatch[zone, hub, date_index] for hub in hubs
                )
                model.addConstr(unmet[zone, date_index, sample_index] >= float(demand[sample_index, zone]) - capacity)

    expected_profit = gp.LinExpr()
    for date_index in dates:
        weight = data["q_l"][date_index] / sample_counts[date_index]
        demand = samples[date_index]
        for sample_index in range(sample_counts[date_index]):
            revenue = data["rho"] * float(demand[sample_index].sum())
            penalty = (data["rho"] + data["b"]) * gp.quicksum(
                unmet[zone, date_index, sample_index] for zone in zones
            )
            expected_profit += weight * revenue
            expected_profit -= weight * penalty
        expected_profit -= data["q_l"][date_index] * gp.quicksum(
            data["t_ij"][zone, hub] * dispatch[zone, hub, date_index]
            for zone in zones
            for hub in hubs
        )
    investment = (
        gp.quicksum(data["f"][hub] * opened[hub] for hub in hubs)
        + data["c_m"] * gp.quicksum(fleet[hub] for hub in hubs)
        + data["c_f"] * gp.quicksum(installed[zone] for zone in zones)
    )
    model.setObjective(-investment + expected_profit, GRB.MAXIMIZE)
    model.optimize()

    solved = model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED)
    if model.SolCount == 0 or not solved:
        return {
            "status": int(model.Status),
            "status_name": _status_name(model.Status),
            "runtime": float(model.Runtime),
            "mip_gap": None,
            "objective": None,
            "bound": None,
        }

    x_val = np.array([opened[hub].X for hub in hubs])
    m_val = np.array([fleet[hub].X for hub in hubs])
    n_val = np.array([installed[zone].X for zone in zones])
    z_val = np.array([
        [[dispatch[zone, hub, date_index].X for date_index in dates] for hub in hubs]
        for zone in zones
    ])
    profit = operating_profit(n_val, z_val, data)
    spent = investment_cost(x_val, m_val, n_val, data)
    if abs(model.ObjVal - (-spent + profit)) > 1e-4 * (1.0 + abs(model.ObjVal)):
        raise RuntimeError(
            f"Gurobi 目标 {model.ObjVal} 与按容量回代的 SAA 目标 {-spent + profit} 不一致"
        )

    decision = {
        "x": {hub: _as_int(opened[hub].X, f"x[{hub}]") for hub in hubs},
        "y": {
            (zone, hub): _as_int(assigned[zone, hub].X, f"y[{zone},{hub}]")
            for zone in zones
            for hub in hubs
        },
        "m": {hub: _as_int(fleet[hub].X, f"m[{hub}]") for hub in hubs},
        "n": {zone: _as_int(installed[zone].X, f"n[{zone}]") for zone in zones},
        "z": {
            (zone, hub, date_index): _as_int(
                dispatch[zone, hub, date_index].X, f"z[{zone},{hub},{date_index}]"
            )
            for zone in zones
            for hub in hubs
            for date_index in dates
        },
    }
    return {
        "status": int(model.Status),
        "status_name": _status_name(model.Status),
        "runtime": float(model.Runtime),
        "mip_gap": float(model.MIPGap),
        "objective": float(model.ObjVal),
        "bound": float(model.ObjBound),
        "node_count": int(model.NodeCount),
        "investment_cost": spent,
        "sample_average_profit": profit,
        "decision": decision,
    }


def _toy_data() -> dict:
    """一区、两 hub、两种日期。q_l 故意不等于样本天数占比，用来核对权重。"""
    return {
        "I_num": 1,
        "J_num": 2,
        "S_num": 2,
        "samples": {
            0: np.array([[3.0], [12.0]]),
            1: np.array([[20.0], [8.0], [15.0]]),
        },
        "q_l": [0.25, 0.75],
        "f": [6.0, 4.0],
        "c_m": 1.5,
        "c_f": 2.0,
        "rho": 2.0,
        "b": 3.0,
        "tau": 5.0,
        "kappa": 4.0,
        "n_bar": [1.0],
        "M_m": [1.0, 1.0],
        "M_z": np.ones((1, 2)),
        "t_ij": np.array([[0.5, 1.25]]),
    }


def _enumerate_toy(data: dict) -> float:
    zone_count, hub_count, date_count = data["I_num"], data["J_num"], data["S_num"]
    best = None
    for opened in itertools.product((0, 1), repeat=hub_count):
        for assignment in itertools.product((0, 1), repeat=zone_count * hub_count):
            assigned = np.array(assignment, dtype=float).reshape(zone_count, hub_count)
            if np.any(assigned > np.asarray(opened)[None, :]):
                continue
            if np.any(assigned.sum(axis=1) > 1):
                continue
            fleet_ranges = [range(int(data["M_m"][hub]) * opened[hub] + 1) for hub in range(hub_count)]
            for fleet in itertools.product(*fleet_ranges):
                dispatch_ranges = []
                for _date_index in range(date_count):
                    for zone in range(zone_count):
                        for hub in range(hub_count):
                            upper = min(fleet[hub], int(data["M_z"][zone, hub]) * int(assigned[zone, hub]))
                            dispatch_ranges.append(range(upper + 1))
                for installed in itertools.product(*[range(int(data["n_bar"][zone]) + 1) for zone in range(zone_count)]):
                    for flat in itertools.product(*dispatch_ranges):
                        dispatch = np.array(flat, dtype=float).reshape(date_count, zone_count, hub_count)
                        dispatch = np.transpose(dispatch, (1, 2, 0))
                        if any(
                            dispatch[:, hub, date_index].sum() > fleet[hub]
                            for date_index in range(date_count)
                            for hub in range(hub_count)
                        ):
                            continue
                        value = -investment_cost(opened, fleet, installed, data) + operating_profit(
                            installed, dispatch, data
                        )
                        best = value if best is None else max(best, value)
    if best is None:
        raise RuntimeError("手写例子没有可行解")
    return best


def _shared_fleet_toy() -> dict:
    """两区共用一个 hub 的一辆车，用来核对 sum_i z_ijl <= m_j。"""
    return {
        "I_num": 2,
        "J_num": 1,
        "S_num": 1,
        "samples": {0: np.array([[10.0, 10.0], [0.0, 9.0]])},
        "q_l": [1.0],
        "f": [0.0],
        "c_m": 0.0,
        "c_f": 30.0,
        "rho": 2.0,
        "b": 2.0,
        "tau": 1.0,
        "kappa": 10.0,
        "n_bar": [1.0, 1.0],
        "M_m": [1.0],
        "M_z": np.ones((2, 1)),
        "t_ij": np.array([[0.0], [0.4]]),
    }


def _check_instance(data: dict, label: str) -> None:
    enumerated = _enumerate_toy(data)
    solved = solve_saa(data, {"output_flag": False, "mip_gap": 0.0})
    if solved["status_name"] != "OPTIMAL":
        raise RuntimeError(f"{label} 没有解到最优：{solved['status_name']}")
    if abs(solved["objective"] - enumerated) > 1e-5:
        raise RuntimeError(f"{label} 的 SAA 最优值 {solved['objective']} 与枚举 {enumerated} 不一致")
    print(f"{label} 通过：最优值 {solved['objective']:.6g}")


def self_check() -> None:
    demand = np.array([3.0, 9.0])
    capacity = np.array([5.0, 4.0])
    unmet = np.maximum(demand - capacity, 0.0)
    served = np.minimum(demand, capacity)
    compact = 2.0 * demand.sum() - 5.0 * unmet.sum()
    expanded = 2.0 * served.sum() - 3.0 * unmet.sum()
    if abs(compact - expanded) > 1e-9:
        raise RuntimeError("日运营利润的两种写法不相等")
    _check_instance(_toy_data(), "日期权重例子")
    _check_instance(_shared_fleet_toy(), "共用电队例子")

"""fixed-only 的受限主问题。

决策只有每个区的固定桩数量 n_i，0 <= n_i <= n_bar_i，且 n_i 为整数。
n 在看到日期类型和需求之前决定，所有日期类型共用同一组 n。

目标是

    - c^f * sum_i n_i
    + sum_l sum_n (q_l / N_{l,Omega}) * eta_{l,n}。

eta 只有已经加入的 pattern 上界。最大化会把 eta 推到这些上界的最低值。
pattern 没有加全时，目标是真最坏利润的上界。

加速（与 charge_centrally_solver_1030 对齐，只改求解，不改模型含义）：
- 容量提成：C_i = tau * n_i 用等式钉死，割只引用 C；
- start_solution：上一轮整数 n 作为本轮 MIP start。
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from data import sample_keys


def solve_master(instance: dict, active: dict[tuple, list[tuple[int, ...]]],
                 supports: dict[tuple, float], limits: dict | None = None,
                 start_solution: dict | None = None) -> dict:
    """求解当前割集合下的主问题。没有可行解时 solvable 为 False。"""
    limits = {} if limits is None else limits
    zones = instance["zones"]
    model = gp.Model("fixed_only_master")
    model.Params.OutputFlag = 0
    # 不写 MIPGap 时，Gurobi 默认相对间隙是 1e-4。
    if limits.get("mip_gap") is not None:
        model.Params.MIPGap = float(limits["mip_gap"])
    # 绝对空隙到这个数，Gurobi 也会宣布最优。相对间隙更紧时，先达到的那个生效。
    if limits.get("mip_gap_abs") is not None:
        model.Params.MIPGapAbs = float(limits["mip_gap_abs"])
    if limits.get("time_limit") is not None:
        model.Params.TimeLimit = float(limits["time_limit"])

    fixed_count = model.addVars(
        zones, vtype=GRB.INTEGER, lb=0,
        ub={zone: int(instance["n_bar"][zone]) for zone in zones},
        name="n",
    )
    # 容量提成：每条割只碰 C，不再把 tau*n 复制进每一条割。
    capacity = model.addVars(zones, lb=0.0, name="C")
    tau = float(instance["tau"])
    for zone in zones:
        model.addConstr(capacity[zone] == tau * fixed_count[zone], name=f"capacity_def[{zone}]")

    recourse = {
        key: model.addVar(lb=-GRB.INFINITY, name=f"eta_{key[0]}_{key[1]}")
        for key in sample_keys(instance)
    }
    rho, penalty = float(instance["rho"]), float(instance["penalty"])
    for key, patterns in active.items():
        for index, pattern in enumerate(patterns):
            served_capacity = gp.quicksum(
                int(pattern[zone]) * capacity[zone] for zone in zones
            )
            support = supports[key[0], key[1], pattern]
            model.addConstr(
                recourse[key] <= (rho + penalty) * served_capacity - support,
                name=f"cut_{key[0]}_{key[1]}_{index}",
            )

    # 上一轮整数方案。eta 不填：新割可能让旧 eta 不可行。
    if start_solution is not None and "n" in start_solution:
        for zone, count in zip(zones, start_solution["n"]):
            fixed_count[zone].Start = int(count)

    investment = instance["fixed_cost"] * gp.quicksum(fixed_count[zone] for zone in zones)
    operating = gp.quicksum(
        instance["sample_weight"][event] * variable
        for (event, _), variable in recourse.items()
    )
    model.setObjective(-investment + operating, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0 or model.Status not in (
        GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED,
    ):
        return {
            "solvable": False,
            "status": _status_name(model.Status),
            "runtime": float(model.Runtime),
        }

    counts = [int(round(fixed_count[zone].X)) for zone in zones]
    eta = {key: float(recourse[key].X) for key in recourse}
    investment_cost = float(instance["fixed_cost"] * sum(counts))
    return {
        "solvable": True,
        "status": _status_name(model.Status),
        "objective": float(model.ObjVal),
        "bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
        "runtime": float(model.Runtime),
        "n": counts,
        "eta": eta,
        "investment_cost": investment_cost,
        "operating_profit": float(model.ObjVal) + investment_cost,
        "variable_names": [variable.VarName for variable in model.getVars()],
        "warm_start": start_solution is not None and "n" in start_solution,
    }


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

"""给定一组 n、z 和当前 eta，求一个样本上的最坏三段利润。

每个样本的分离模型只建一次。n、z 只出现在目标的 beta 里，
约束（盒子、McCormick、椭球）不变，所以之后只改目标再求解。
椭球写成二范数的平方，和二次型 ||L^{-1}(D-mu)||^2 <= r^2 是同一个集合。

一次求解可以带回搜索中碰到的多个最优 pattern。每条都是有效割。
椭球仍写成凸二次不等式。Gurobi 在 NonConvex=0 下会把范数等式看成非凸二次等式，所以这里不用那种写法。

分离在最小化 sum_i (斜率_{piece(i)} * D_i + beta_i)。
每个区用三个 0-1 变量选一段，再用 McCormick 等式把“选中段的 D”写出来。
因为利润对区可加，而区与区只通过椭球连在一起，这个混合整数模型等于
min_D min_pattern (斜率·D + beta)。
"""

from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from instance import local_box
from patterns import PIECES, beta_zone, canonical_pattern, slope


# 键是 (实例对象的 id, 日期类型, 样本编号)。几何数据变了但还是同一个 dict 时，
# 这里会继续用旧模型。【核对】当前脚本不会在同一次求解中改盒子或椭球。
_ORACLES: dict[tuple, dict] = {}


def solve_separation(solution: dict, event: str, sample_idx: int, instance: dict) -> dict:
    """求解一个样本的分离问题。solution 使用 n、z、eta 三个字段。

    返回的 violation = 主问题当前 eta - 分离得到的最坏利润。
    violation 大于容差时，这条部署的 eta 还偏高，需要把对应 pattern 加进主问题。
    分离没有证明最优时，certified_optimal 为 False。调用方不能把这种情况记成收敛。
    """
    oracle = _oracle(instance, event, sample_idx)
    zones = oracle["zones"]
    fixed_counts = [float(solution["n"][zone]) for zone in zones]
    mobile_counts = [float(solution["z"][event][zone]) for zone in zones]
    current_eta = float(solution["eta"][event, sample_idx])
    model = oracle["model"]
    piece = oracle["piece"]
    product = oracle["product"]
    # beta 用当前解的 n、z 算成数字。换一个部署时只改目标，不改建模型。
    profit = gp.quicksum(
        slope(piece_id, instance) * product[zone, piece_id]
        + beta_zone(piece_id, fixed_counts[zone_id], mobile_counts[zone_id], instance) * piece[zone, piece_id]
        for zone_id, zone in enumerate(zones)
        for piece_id in PIECES
    )
    model.setObjective(profit, GRB.MINIMIZE)
    pool = int(instance.get("separation_pool_solutions") or 1)
    model.Params.PoolSolutions = max(pool, 1)
    # 1：保留这棵搜索树里已经找到的解，不再为了凑满解池继续搜索。
    # 漏掉的并列 pattern 仍然是有效割，下一轮还可以被发现。不影响割的正确性。
    model.Params.PoolSearchMode = 1 if pool > 1 else 0
    if instance.get("separation_time_limit") is not None:
        model.Params.TimeLimit = max(float(instance["separation_time_limit"]), 1e-3)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return {
            "status": "NOT_CERTIFIED",
            "certified_optimal": False,
            "native_status": int(model.Status),
            "patterns": [],
        }
    tolerance = float(instance["separation_tolerance"])
    worst_profit = float(model.ObjVal)
    patterns = []
    scale = 1.0 + abs(worst_profit)
    for number in range(int(model.SolCount)):
        model.Params.SolutionNumber = number
        if abs(float(model.PoolObjVal) - worst_profit) > 1e-6 * scale:
            continue
        # 每个区取等于 1 的那段。g=0 时随后会把第 1 段收成第 0 段。
        pattern = canonical_pattern(tuple(
            int(max(PIECES, key=lambda piece_id: piece[zone, piece_id].Xn))
            for zone in zones
        ), instance)
        if pattern not in patterns:
            patterns.append(pattern)
    violation = current_eta - worst_profit
    violated = violation > tolerance
    return {
        "status": "FOUND_VIOLATION" if violated else "CERTIFIED_NO_VIOLATION",
        "certified_optimal": True,
        "pattern": None if not patterns else patterns[0],
        "patterns": patterns if violated else [],
        "worst_profit": worst_profit,
        "violation": violation,
        "violated": violated,
    }


def _oracle(instance: dict, event: str, sample_idx: int) -> dict:
    key = (id(instance), event, sample_idx)
    oracle = _ORACLES.get(key)
    if oracle is None:
        oracle = _build_oracle(instance, event, sample_idx)
        _ORACLES[key] = oracle
    return oracle


def _build_oracle(instance: dict, event: str, sample_idx: int) -> dict:
    """为一个样本建立分离模型。需求盒和椭球在这里固定，n、z 不进入约束。"""
    zones = tuple(instance["zones"])
    lower, upper = local_box(instance, event, sample_idx)
    mean = np.asarray(instance["mean"], dtype=float)
    covariance = np.asarray(instance["covariance"], dtype=float)
    cholesky = np.linalg.cholesky(covariance)
    inverse_cholesky = np.linalg.solve(cholesky, np.eye(len(zones)))
    radius = float(instance["radius"])
    model = gp.Model(f"local_separation_{event}_{sample_idx}")
    model.Params.OutputFlag = 0
    # MIPGap 保持 Gurobi 默认 1e-4。椭球约束是凸二次，不打开非凸模式。
    # 与支持函数使用同一个 BarQCPConvTol，避免两边各停在默认精度的不同位置。
    model.Params.BarQCPConvTol = 1e-8
    model.Params.NonConvex = 0
    model.Params.Threads = 1
    demand, piece, product = {}, {}, {}
    for zone_id, zone in enumerate(zones):
        demand[zone] = model.addVar(lb=float(lower[zone_id]), ub=float(upper[zone_id]), name=f"D[{zone}]")
        for piece_id in PIECES:
            piece[zone, piece_id] = model.addVar(vtype=GRB.BINARY, name=f"y[{zone},{piece_id}]")
            # product = piece * D。四个不等式是 [下界, 上界] 上的 McCormick 包络。
            # piece 是 0-1 时，包络收成 product = D（选中）或 product = 0（没选中）。
            product[zone, piece_id] = model.addVar(lb=-GRB.INFINITY, name=f"w[{zone},{piece_id}]")
            chosen = piece[zone, piece_id]
            demand_var = demand[zone]
            low, high = float(lower[zone_id]), float(upper[zone_id])
            model.addConstr(product[zone, piece_id] >= low * chosen)
            model.addConstr(product[zone, piece_id] <= high * chosen)
            model.addConstr(product[zone, piece_id] >= demand_var - high * (1 - chosen))
            model.addConstr(product[zone, piece_id] <= demand_var - low * (1 - chosen))
        model.addConstr(gp.quicksum(piece[zone, piece_id] for piece_id in PIECES) == 1)
    whitened = model.addVars(len(zones), lb=-GRB.INFINITY, name="whitened")
    for row in range(len(zones)):
        model.addConstr(whitened[row] == gp.quicksum(
            float(inverse_cholesky[row, column]) * (demand[zones[column]] - float(mean[column]))
            for column in range(len(zones))
        ))
    model.addQConstr(
        gp.quicksum(whitened[row] * whitened[row] for row in range(len(zones))) <= radius ** 2,
        name="SEP_ELLIPSOID",
    )
    return {"model": model, "zones": zones, "piece": piece, "product": product}

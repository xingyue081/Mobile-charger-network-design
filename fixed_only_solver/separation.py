"""固定桩容量已知时，计算一个局部歧义集上的最坏运营利润。

每个区只有两段利润。容量 C_i = tau * n_i。

    D_i <= C_i 时，利润是 rho * D_i。
    D_i > C_i 时，利润是 (rho + b) * C_i - b * D_i。

两段取较小值，就是

    rho * min(D_i, C_i) - b * (D_i - C_i)^+。

没有移动桩，所以没有第三段，也没有运输成本。
pattern a_i = 1 表示第 i 区使用缺电那一段，a_i = 0 表示使用全部满足那一段。

局部集合与另外两套求解器相同：该样本的 l_infinity 盒子，再与全体样本
共用的椭球 (D - mu)^T Sigma_inv (D - mu) <= r^2 取交集。
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from data import local_box


def piecewise_profit(demand: np.ndarray, capacity: np.ndarray, rho: float, penalty: float) -> float:
    """按两段公式直接计算利润，用来核对分离模型的目标值。"""
    demand = np.asarray(demand, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    served = np.minimum(demand, capacity)
    shortage = np.maximum(demand - capacity, 0.0)
    return float(rho * served.sum() - penalty * shortage.sum())


def support_value(pattern: tuple[int, ...], lower: np.ndarray, upper: np.ndarray, instance: dict) -> float:
    """h(a) = max_D sum_i ((rho + b) a_i - rho) D_i，D 在盒子与椭球的交集上。

    h 只依赖 pattern 和这个样本的局部集合，不依赖 n。
    主问题里的割是 eta <= (rho + b) * tau * sum_i a_i n_i - h(a)。
    """
    rho, penalty = float(instance["rho"]), float(instance["penalty"])
    direction = np.array(
        [(rho + penalty) * int(piece) - rho for piece in pattern],
        dtype=float,
    )
    model = gp.Model("fixed_only_support")
    model.Params.OutputFlag = 0
    demand = model.addVars(
        len(pattern), lb=lower.tolist(), ub=upper.tolist(), name="D",
    )
    _add_ellipsoid(model, demand, instance)
    model.setObjective(
        gp.quicksum(float(direction[i]) * demand[i] for i in range(len(pattern))),
        GRB.MAXIMIZE,
    )
    model.optimize()
    # 福田 35 维椭球上，默认精度会偶发停在 SUBOPTIMAL。
    # 提高 NumericFocus 后仍要求 Gurobi 证明最优，两种精度都失败才报错。
    if model.Status != GRB.OPTIMAL:
        for focus in (2, 3):
            model.Params.NumericFocus = focus
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                break
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"支持函数没有证明最优，状态 {model.Status}")
    return float(model.ObjVal)


def worst_operating_profit(capacity: np.ndarray, event: str, sample_idx: int, instance: dict,
                           time_limit: float | None = None) -> dict:
    """在一个样本的局部集合上最小化两段利润。

    二元变量 a 选择每个区的利润段，v 表示 a 与 D 的乘积。
    目标对固定的 a 是 D 的线性函数，椭球是凸二次约束。
    返回的 profit 是 Gurobi 在默认相对间隙内接受的最优值。
    """
    lower, upper = local_box(instance, event, sample_idx)
    zone_count = len(capacity)
    rho, penalty = float(instance["rho"]), float(instance["penalty"])
    model = gp.Model("fixed_only_separation")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    demand = model.addVars(zone_count, lb=lower.tolist(), ub=upper.tolist(), name="D")
    shortage = model.addVars(zone_count, vtype=GRB.BINARY, name="a")
    product = model.addVars(zone_count, lb=-GRB.INFINITY, name="v")
    for zone in range(zone_count):
        # v = a * D 的 McCormick 包络。a 是 0-1 时，包络取到乘积本身。
        model.addConstr(product[zone] >= float(lower[zone]) * shortage[zone])
        model.addConstr(product[zone] <= float(upper[zone]) * shortage[zone])
        model.addConstr(product[zone] >= demand[zone] - float(upper[zone]) * (1 - shortage[zone]))
        model.addConstr(product[zone] <= demand[zone] - float(lower[zone]) * (1 - shortage[zone]))
    _add_ellipsoid(model, demand, instance)

    # rho * D - (rho + b) * a * D + (rho + b) * C * a。
    # a 选 0 时这一区是 rho * D；a 选 1 时是 (rho + b) * C - b * D。
    model.setObjective(
        gp.quicksum(
            rho * demand[zone]
            - (rho + penalty) * product[zone]
            + (rho + penalty) * float(capacity[zone]) * shortage[zone]
            for zone in range(zone_count)
        ),
        GRB.MINIMIZE,
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"分离没有证明最优：status={model.Status}, runtime={model.Runtime}, "
            f"gap={model.MIPGap}, sol_count={model.SolCount}"
        )
    pattern = tuple(int(round(shortage[zone].X)) for zone in range(zone_count))
    demand_value = np.array([float(demand[zone].X) for zone in range(zone_count)])
    return {
        "profit": float(model.ObjVal),
        "pattern": pattern,
        "demand": demand_value,
    }


def pattern_cut_value(pattern: tuple[int, ...], capacity: np.ndarray, support: float,
                      instance: dict) -> float:
    """当前容量下，一条 pattern 割的右边：(rho + b) * sum a_i C_i - h(a)。"""
    rho, penalty = float(instance["rho"]), float(instance["penalty"])
    covered = sum(
        int(piece) * float(capacity[zone]) for zone, piece in enumerate(pattern)
    )
    return (rho + penalty) * covered - float(support)


def _add_ellipsoid(model: gp.Model, demand, instance: dict) -> None:
    """加入 (D - mu)^T Sigma_inv (D - mu) <= r^2。

    Sigma_inv 和 r 都来自数据包。二次型把矩阵的上下三角都写进去，
    结果等于 D^T Sigma_inv D，不把非对角项再乘一次 2。
    """
    mean = instance["mean"]
    precision = instance["precision"]
    zone_count = len(mean)
    diff = [demand[zone] - float(mean[zone]) for zone in range(zone_count)]
    quadratic = gp.quicksum(
        float(precision[left, right]) * diff[left] * diff[right]
        for left in range(zone_count)
        for right in range(zone_count)
    )
    model.addQConstr(quadratic <= float(instance["radius"]) ** 2, name="ellipsoid")

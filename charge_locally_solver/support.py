"""在椭球与截断盒子的交集上计算支持函数。

这是原始凸问题：变量是需求 D，约束是集合本身，目标是线性函数的最大值。
h 只依赖数据和 pattern 的方向，不依赖 n、z，所以可以在主问题之外预先算成常数。

局部集合是三部分的交集：
- 样本周围的 l_infinity 盒子，下界已截到 0（instance.local_box）；
- 全体样本共用的 Mahalanobis 椭球 (D-mu)^T Sigma^{-1} (D-mu) <= r^2。

Sigma = L L^T 时，椭球写成 ||L^{-1}(D-mu)||^2 <= r^2。
h(pattern) = max (-斜率 · D)，最大值就在这个交集上取。
"""

from __future__ import annotations

import os

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instance import local_box, sample_keys
from patterns import all_patterns, support_direction


def _precision_and_whitening(instance: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = np.asarray(instance["covariance"], dtype=float)
    precision = np.linalg.inv(covariance)
    cholesky = np.linalg.cholesky(covariance)
    # Sigma = L L^T 时，椭球约束写成 ||L^{-1}(D-mu)||^2 <= r^2。
    inverse_cholesky = np.linalg.solve(cholesky, np.eye(cholesky.shape[0]))
    return precision, cholesky, inverse_cholesky


def _optimize_support(model: gp.Model) -> float | None:
    model.Params.OutputFlag = 0
    # 默认 1e-6 时，目标约 5e4 的支持函数可以少算约 0.01，仍被报告为最优。
    # 分离用绝对差 1e-4 验收，所以先收到 1e-8。福田 35 维上这一档偶发失败，
    # 失败后再放宽精度重解；两种坐标都失败才由调用方报错。
    for tolerance, focus in ((1e-8, 0), (1e-6, 2), (1e-5, 3)):
        model.Params.BarQCPConvTol = tolerance
        model.Params.NumericFocus = focus
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return float(model.ObjVal)
    return None


def support_value(direction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  instance: dict) -> float:
    """求解 max direction·D，D 位于局部盒子和 Mahalanobis 椭球的交集。

    direction 已经是斜率的相反数。同一次精度下用两种等价坐标各算一次，
    取更大的最优值。某一种停在次优时只用另一种。两种都失败才抛错。
    """
    mean = np.asarray(instance["mean"], dtype=float)
    _, cholesky, inverse_cholesky = _precision_and_whitening(instance)
    radius = float(instance["radius"])
    demand_model = gp.Model("local_support")
    threads = os.environ.get("CG_THREADS")
    if threads:
        demand_model.Params.Threads = int(threads)
    demand = demand_model.addMVar(len(direction), lb=lower, ub=upper, name="D")
    whitened = inverse_cholesky @ (demand - mean)
    demand_model.addConstr(whitened @ whitened <= radius ** 2, name="SEP_ELLIPSOID")
    demand_model.setObjective(direction @ demand, GRB.MAXIMIZE)
    values = [value for value in (_optimize_support(demand_model),) if value is not None]

    whitened_model = gp.Model("local_support_whitened")
    if threads:
        whitened_model.Params.Threads = int(threads)
    coefficient = whitened_model.addMVar(len(direction), lb=-GRB.INFINITY, name="u")
    whitened_model.addConstr(coefficient @ coefficient <= radius ** 2, name="SEP_ELLIPSOID")
    demand_expr = mean + cholesky @ coefficient
    whitened_model.addConstr(demand_expr >= lower, name="box_lower")
    whitened_model.addConstr(demand_expr <= upper, name="box_upper")
    whitened_model.setObjective(direction @ demand_expr, GRB.MAXIMIZE)
    whitened_value = _optimize_support(whitened_model)
    if whitened_value is not None:
        values.append(whitened_value)
    if not values:
        raise RuntimeError("支持函数求解失败，两种坐标都没有得到最优解")
    return max(values)


class SupportCache:
    """按需要计算 h。20 个 zone 不能预先枚举 3^20 个 pattern。

    主问题每次真正加入一条割，才为那个 (日期类型, 样本, pattern) 算一次 h。
    同一个 key 再出现时直接读表。
    """

    def __init__(self, instance: dict):
        self.instance = instance
        self.table: dict[tuple, float] = {}

    def __getitem__(self, key: tuple) -> float:
        if key not in self.table:
            event, sample_idx, pattern = key
            lower, upper = local_box(self.instance, event, sample_idx)
            direction = support_direction(pattern, self.instance)
            self.table[key] = support_value(direction, lower, upper, self.instance)
        return self.table[key]


def precompute_supports(instance: dict) -> dict[tuple, float]:
    """为每个样本、每个 pattern 计算 h。穷举和约束生成必须共用这张表。

    【核对】validate.py 的“目标差为 0”用的就是这张共用表。
    它证明割的循环和“全部 pattern 的 beta-h”一致，
    不能单独证明 h 的几何和另一套直接利润计算一致。直接利润对照在 9 月的公式审计里。
    """
    table = {}
    patterns = all_patterns(len(instance["zones"]))
    for event, sample_idx in sample_keys(instance):
        lower, upper = local_box(instance, event, sample_idx)
        center = np.asarray(instance["samples"][event][sample_idx], dtype=float)
        precision, _, _ = _precision_and_whitening(instance)
        diff = center - np.asarray(instance["mean"], dtype=float)
        if float(diff @ precision @ diff) > float(instance["radius"]) ** 2 + 1e-8:
            raise ValueError(f"样本 {(event, sample_idx)} 不在椭球内")
        for pattern in patterns:
            direction = support_direction(pattern, instance)
            table[event, sample_idx, pattern] = support_value(direction, lower, upper, instance)
    return table

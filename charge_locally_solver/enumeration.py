"""对很小的整数可行域做穷举，作为约束生成的独立对照。

每个 (n, m, z) 都选择满足流平衡的最低单程运输成本，
再用全部 pattern 的 beta-h 计算最坏利润。
这里不调用分离 MIP，也不使用主问题里已经加入的部分割。

【核对】利润部分和约束生成共用同一张支持函数表，以及同一套 beta。
两边目标一致，说明割的循环没有漏掉这个小例子里的 pattern；
它不另算一遍“直接对需求最小化原始利润”。

【核对】运输穷举只写了 2 个 zone、2 个日期类型。
日期类型按实例里的顺序配成一来一回两条转移，不能拿去跑 20 区。
"""

from __future__ import annotations

import numpy as np

from patterns import all_patterns, beta_pattern


def _deployments(fleet: int, zone_count: int) -> list[tuple[int, ...]]:
    if zone_count != 2:
        raise ValueError("当前穷举只实现 2 个 zone，避免无意中展开大规模组合")
    return [(left, fleet - left) for left in range(fleet + 1)]


def _transport_plans(supply: tuple[int, int], demand: tuple[int, int]) -> list[np.ndarray]:
    """列出从 supply 部署改到 demand 部署的全部整数运输矩阵。

    行是出发区，列是到达区。两个区时，第一个区留下的车辆数就能决定整张矩阵。
    """
    plans = []
    low = max(0, supply[0] - demand[1])
    high = min(supply[0], demand[0])
    for stay in range(low, high + 1):
        move = supply[0] - stay
        into_first = demand[0] - stay
        stay_second = supply[1] - into_first
        if min(move, into_first, stay_second) < 0:
            continue
        if move + stay_second != demand[1]:
            continue
        plans.append(np.array([[stay, move], [into_first, stay_second]], dtype=float))
    if not plans:
        raise RuntimeError(f"没有可行运输方案: {supply} -> {demand}")
    return plans


def _reposition_cost(plan_forward: np.ndarray, plan_back: np.ndarray, instance: dict) -> float:
    """两条转移的期望单程成本：p_kl * gamma * sum d_ii' r_ii'kl。"""
    events = instance["date_types"]
    forward = (events[0], events[1])
    backward = (events[1], events[0])
    distance = instance["distance"]
    gamma = instance["gamma"]
    probability = instance["transition_probability"]
    return float(
        probability[forward] * gamma * np.sum(distance * plan_forward)
        + probability[backward] * gamma * np.sum(distance * plan_back)
    )


def _worst_operating_profit(fixed_counts: np.ndarray, deployment: dict, supports: dict, instance: dict) -> float:
    """按 q_l/N_{l,Omega} 把每个样本的最坏 pattern 利润加总。

    一个样本的最坏利润 = min_pattern (beta(pattern, n, z) - h(pattern))。
    这里的 min 走完 3^{区数} 个 pattern，所以只用于 2 区。
    """
    total = 0.0
    patterns = all_patterns(len(instance["zones"]))
    for event in instance["date_types"]:
        mobile_counts = np.asarray(deployment[event], dtype=float)
        for sample_idx in range(len(instance["samples"][event])):
            worst = min(
                beta_pattern(pattern, fixed_counts, mobile_counts, instance)
                - supports[event, sample_idx, pattern]
                for pattern in patterns
            )
            total += instance["sample_weight"][event] * worst
    return total


def enumerate_optima(instance: dict, supports: dict) -> dict:
    """穷举 n、m 和两种日期类型的部署，返回最优目标及全部并列最优解。"""
    best = -np.inf
    optima = []
    events = instance["date_types"]
    for n0 in range(instance["n_bar"] + 1):
        for n1 in range(instance["n_bar"] + 1):
            fixed_counts = np.array([n0, n1], dtype=float)
            for fleet in range(instance["m_bar"] + 1):
                for first in _deployments(fleet, 2):
                    for second in _deployments(fleet, 2):
                        cost = min(
                            _reposition_cost(forward, back, instance)
                            for forward in _transport_plans(first, second)
                            for back in _transport_plans(second, first)
                        )
                        deployment = {events[0]: first, events[1]: second}
                        value = (
                            -instance["fixed_cost"] * float(fixed_counts.sum())
                            - instance["mobile_cost"] * fleet
                            - cost
                            + _worst_operating_profit(fixed_counts, deployment, supports, instance)
                        )
                        decision = {"n": [n0, n1], "m": fleet, "z": deployment}
                        if value > best + 1e-8:
                            best = value
                            optima = [(decision, value)]
                        elif abs(value - best) <= 1e-8:
                            optima.append((decision, value))
    return {"objective": float(best), "decisions": [item[0] for item in optima]}

"""合成的 2-zone 本地补能实例。

这个模块只构造手造数据，不读取深圳 CSV，也不估计 support 半径。
数值和 reformulation/validate_charge_locally_nonnegative.py 使用同一组几何与成本，
因此约束生成的结果可以和那边的穷举目标对照。

读这份文件时看三件事：
1. 每个样本的权重是 q_l / N_{l,Omega}，不是 1/N，也不按筛选后的天数重算 q_l。
2. 局部盒子是样本中心加减 epsilon_l，下界再截到 0。盒子再和共享椭球相交，交集在 support.py 里实现。
3. p_kl 只保留 k != l。同一天类型不发生 repositioning，所以对角项不进目标。
"""

from __future__ import annotations

import numpy as np


def synthetic_two_zone() -> dict:
    """返回约束生成、分离和穷举共同使用的一份实例。

    q 来自假想的完整训练期天数 (6, 4)，不是 support 内样本数 (2, 1)。
    每个样本的权重是 q_l / N_{l,Omega}。
    p_kl 只保留 k != l 的联合概率，对角项不进入 repositioning。

    【核对】这里的 rho、b、g、容量和距离都是为了公式对照手造的，不能拿去解释深圳结果。
    g=1，并且 0 <= 1 <= 4+3，三段下包络的斜率条件成立。
    """
    rho, penalty, local_g = 4.0, 3.0, 1.0
    if not 0.0 <= local_g <= rho + penalty:
        raise ValueError("三仿射下包络要求 0 <= g <= rho+b")
    samples = {
        "workday": (np.array([4.2, 5.1]), np.array([3.8, 4.8])),
        "weekend": (np.array([4.5, 5.2]),),
    }
    # epsilon_l 是该日期类型的 l_infinity Wasserstein 半径，不是椭球半径。
    # 椭球半径在下面的 radius。两个半径管的是两个不同的集合。
    epsilon = {"workday": 0.35, "weekend": 0.30}
    q = {"workday": 0.6, "weekend": 0.4}
    counts = {event: len(rows) for event, rows in samples.items()}
    return {
        "zones": (0, 1),
        "date_types": ("workday", "weekend"),
        "samples": samples,
        "epsilon": epsilon,
        "mean": np.array([4.0, 5.0]),
        "covariance": np.array([[1.0, 0.2], [0.2, 1.0]]),
        "radius": 2.0,
        "rho": rho,
        "penalty": penalty,
        "local_g": local_g,
        "tau": 2.0,
        "kappa": 3.0,
        "fixed_cost": 1.5,
        "mobile_cost": 2.0,
        "gamma": 0.4,
        "distance": np.array([[0.0, 3.0], [3.0, 0.0]]),
        "q": q,
        # 一个 workday 样本的权重是 0.6/2，一个 weekend 样本的权重是 0.4/1。
        # 全部样本权重加总等于 q 的总和 1，所以后面的加权和是期望，不是样本总和。
        "sample_weight": {event: q[event] / counts[event] for event in q},
        "transition_probability": {
            ("workday", "weekend"): 0.12,
            ("weekend", "workday"): 0.12,
        },
        "n_bar": 1,
        "m_bar": 2,
        "separation_tolerance": 1e-4,
    }


def sample_keys(instance: dict) -> list[tuple[str, int]]:
    """按 (日期类型, 该类型下的样本编号) 列出全部分离子问题。

    样本编号从 0 起，只在该日期类型内部编号。
    20 区实例因此有 81+28+10=119 个分离子问题。
    """
    return [
        (event, sample_idx)
        for event in instance["date_types"]
        for sample_idx in range(len(instance["samples"][event]))
    ]


def local_box(instance: dict, event: str, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """样本周围的 l_infinity 盒子，下界截到 0，用来表达需求非负。

    第 i 个坐标的盒子是
    [max(0, Dhat_i - epsilon_l), Dhat_i + epsilon_l]。
    这只是局部集合的盒子部分。共享椭球还要再裁一次，裁剪在 support.py 和 separation.py。

    样本向量的第 i 列必须对应 zones 的第 i 个区。当前合成实例和 20 区实例都把
    zones 设成 0,1,2,...，所以列号和区号相同。
    """
    center = np.asarray(instance["samples"][event][sample_idx], dtype=float)
    epsilon = float(instance["epsilon"][event])
    lower = np.maximum(0.0, center - epsilon)
    upper = center + epsilon
    if np.any(upper < lower):
        raise ValueError(f"空的局部盒子: {(event, sample_idx)}")
    return lower, upper

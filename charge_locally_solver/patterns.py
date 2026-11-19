"""三段利润的 pattern、斜率和不乘需求的常数项。

pattern 是每个 zone 选中的段号，取值 0、1、2。它是仿射片的索引，
不是规划者决定的部署变量。

一个区的运营利润，在 fixed 先服务、本地增量成本只对 mobile 实际服务量收取时，是

    rho * D - g * (D - tau*n)^+ - (rho + b - g) * (D - tau*n - kappa*z)^+。

令 a = tau*n，K = kappa*z。0 <= g <= rho+b 时，它等于下面三段仿射函数的最小值。

第 0 段，D <= a：
    斜率 rho，常数 0。
第 1 段，a < D <= a+K：
    斜率 rho-g，常数 g*a。
第 2 段，D > a+K：
    斜率 -b，常数 (rho+b)*a + (rho+b-g)*K。

最坏利润先对 pattern 取最小，再对局部需求集合取最小。因为常数项不含 D，
两步可以交换：每个 pattern 先在局部集合上算出 min (斜率·D + 常数)，再对 pattern 取最小。
约束生成加的就是这些 pattern 的上界割。
"""

from __future__ import annotations

from itertools import product

import numpy as np

PIECES = (0, 1, 2)


def all_patterns(zone_count: int) -> list[tuple[int, ...]]:
    """2 个 zone 有 9 个 pattern，3 个 zone 有 27 个。

    20 个 zone 有 3^20 个，不能全部枚举。穷举对照只允许 2 个 zone。
    """
    return list(product(PIECES, repeat=zone_count))


def slope(piece: int, instance: dict) -> float:
    """三段斜率：rho、rho-g、-b。"""
    rho, penalty, local_g = instance["rho"], instance["penalty"], instance["local_g"]
    return (rho, rho - local_g, -penalty)[piece]


def beta_zone(piece: int, fixed_count: float, mobile_count: float, instance: dict) -> float:
    """一个 zone 在指定段上、不乘 D 的常数。

    第 0 段没有容量常数。
    第 1 段只含 fixed charger 容量，等于 g*tau*n。
    第 2 段同时含 fixed 与 mobile 容量，等于
    (rho+b)*tau*n + (rho+b-g)*kappa*z。

    fixed_count、mobile_count 可以是普通数，也可以是主问题里的 Gurobi 变量。
    传变量时，返回值是一条线性表达式，直接放进割 eta <= beta - h。
    """
    rho, penalty, local_g = instance["rho"], instance["penalty"], instance["local_g"]
    tau, kappa = instance["tau"], instance["kappa"]
    if piece == 0:
        return 0.0
    if piece == 1:
        return local_g * tau * fixed_count
    if piece == 2:
        return (rho + penalty) * tau * fixed_count + (rho + penalty - local_g) * kappa * mobile_count
    raise ValueError(f"未知的利润段 {piece}")


def beta_pattern(pattern: tuple[int, ...], fixed_counts, mobile_counts, instance: dict) -> float:
    """把各 zone 的 beta 相加。fixed_counts 与 mobile_counts 可以是数或 Gurobi 变量。

    pattern[i] 对应 zones 里的第 i 个区，也对应需求向量的第 i 列。
    当前调用方都按 0,1,2,... 的顺序传入，所以位置和区号一致。
    """
    total = 0.0
    for zone, piece in enumerate(pattern):
        total += beta_zone(piece, fixed_counts[zone], mobile_counts[zone], instance)
    return total


def canonical_pattern(pattern: tuple[int, ...], instance: dict) -> tuple[int, ...]:
    """g=0 时第 0 段和第 1 段是同一条仿射函数，写成同一种 pattern。

    斜率都是 rho，beta 都是 0。换成同一种写法不会改变割。
    g>0 时两段不同，pattern 原样返回。

    【核对】20 区试跑把 g 设成 0，所以会走这个合并。合成 2 区对照的 g 是 1，不合并。
    """
    if float(instance["local_g"]) != 0.0:
        return tuple(pattern)
    return tuple(0 if piece == 1 else int(piece) for piece in pattern)


def support_direction(pattern: tuple[int, ...], instance: dict) -> np.ndarray:
    """支持函数的方向是斜率的相反数。最小化 alpha·D 等于 -max(-alpha·D)。

    因此主问题里的割写成 eta <= beta - h，其中 h = max (-斜率 · D)。
    """
    return np.array([-slope(piece, instance) for piece in pattern], dtype=float)

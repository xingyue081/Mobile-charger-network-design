"""fixed-only 割平面。

每个样本先放两条割：所有区都满足，以及所有区都缺电。
随后用分离寻找当前 n 下真正最坏的分段。若 eta 高于该最坏利润，
就把对应 pattern 的线性上界加进主问题。

新增的割对以后所有整数 n 都有效。h 在局部集合上取支持函数，
不把产生这条割时的 n 写进常数项。

加速（与 charge_centrally_solver_1030 对齐）：
- 上一轮 n 作为下一轮 MIP start；
- 主问题未证明到间隙时，分离只核算，不写新割；
- 可选 adaptive_gap：加割轮次用较松间隙，无新割后再收紧。
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from data import local_box, sample_keys
from master import solve_master
from separation import pattern_cut_value, piecewise_profit, support_value, worst_operating_profit


def _acceptance_gap(value: float) -> float:
    """绝对差允许到 Gurobi 默认相对间隙 1e-4 的量级。

    不修改求解器的 MIPGap。大需求下，0-1 容差会让分离目标与两段利润
    出现远小于这个相对间隙的绝对差。
    """
    return 1e-4 * (1.0 + abs(float(value)))


def solve_fixed_only(instance: dict, max_iterations: int = 100,
                     max_seconds: float | None = None, cut_tolerance: float = 1e-4,
                     mip_gap: float | None = None, mip_gap_abs: float | None = None,
                     master_time_limit: float | None = None,
                     adaptive_gap: bool = False) -> dict:
    """求解固定桩模型。到时或达到迭代上限时，返回最后一组可行 n，不把它当成收敛。"""
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须为正")
    if cut_tolerance < 0:
        raise ValueError("cut_tolerance 不能为负")

    active = {}
    supports = {}
    for key in sample_keys(instance):
        zone_count = len(instance["zones"])
        active[key] = [tuple(0 for _ in range(zone_count)), tuple(1 for _ in range(zone_count))]
        for pattern in active[key]:
            _store_support(supports, instance, key, pattern)

    started = perf_counter()
    history = []
    solution = None
    last_solution = None
    reason = "达到最大迭代次数"
    converged = False
    certifying = False
    loose_gap = 1e-3
    tight_gap = 1e-4 if mip_gap is None else float(mip_gap)
    tight_time = 1800.0 if adaptive_gap else None

    for iteration in range(1, max_iterations + 1):
        elapsed = perf_counter() - started
        if max_seconds is not None and elapsed >= max_seconds:
            reason = "达到总时间上限"
            break
        master_limit = None
        if max_seconds is not None:
            master_limit = max_seconds - (perf_counter() - started)
            if master_limit <= 0:
                reason = "达到总时间上限"
                break
        if adaptive_gap:
            round_mip_gap = tight_gap if certifying else loose_gap
            round_master_time = tight_time if certifying else master_time_limit
        else:
            round_mip_gap = mip_gap
            round_master_time = master_time_limit
        print(f"开始第 {iteration} 轮主问题", flush=True)
        # 单轮时限和总时限同时存在时，用更短的那个。
        round_limit = master_limit
        if round_master_time is not None and (round_limit is None or round_master_time < round_limit):
            round_limit = round_master_time
        solution = solve_master(
            instance, active, supports,
            {
                "mip_gap": round_mip_gap,
                "mip_gap_abs": mip_gap_abs,
                "time_limit": round_limit,
            },
            start_solution=last_solution,
        )
        if not solution["solvable"]:
            reason = "主问题没有可行解"
            solution = None
            break
        last_solution = solution
        print(
            f"主问题 {solution['status']}，目标 {solution['objective']:.6g}，"
            f"上界 {solution['bound']:.6g}，固定桩 {sum(solution['n'])} 个，"
            f"MIP gap 目标 {round_mip_gap}，用时 {solution['runtime']:.1f} 秒",
            flush=True,
        )

        # 主问题没有证明到间隙时，分离只核算这个方案，不把模式写进下一轮。
        write_cuts = solution["status"] == "OPTIMAL"
        added, residual, finished, stop_reason, accounted = _separate_round(
            instance, solution, active, supports, started, max_seconds, cut_tolerance,
            write_cuts=write_cuts,
        )
        history.append({
            "iteration": iteration,
            "master_status": solution["status"],
            "objective": solution["objective"],
            "bound": solution["bound"],
            "master_mip_gap_target": round_mip_gap,
            "cuts_written": write_cuts,
            "accounted_violations_not_written": accounted,
            "added": added,
            "elapsed_seconds": perf_counter() - started,
        })
        print(f"第 {iteration} 轮新增 {added} 条割", flush=True)
        if not write_cuts and finished:
            reason = "主问题未证明到间隙，分离只核算当前方案，不加割"
            print(f"{reason}。本轮若写入会新增 {accounted} 条割。", flush=True)
            break
        if not finished:
            reason = stop_reason
            break
        if (
            adaptive_gap
            and added == 0
            and finished
            and write_cuts
            and not residual
            and not certifying
        ):
            certifying = True
            print(
                f"没有新的割。下一轮把主问题间隙从 {loose_gap} 收到 {tight_gap}。",
                flush=True,
            )
            continue
        if adaptive_gap and added > 0 and certifying:
            certifying = False
            print("收紧间隙后出现新割，后面的轮次回到较松的间隙。", flush=True)
        # 本轮加进了新割时，先用新割再解主问题。
        # 只有没有新割、旧割仍被违反时，才不能宣布收敛。
        if added == 0 and residual:
            reason = "已有割仍被违反"
            break
        if added == 0:
            if solution["status"] == "OPTIMAL":
                converged = True
                reason = "没有新的割，主问题相对间隙已达到 Gurobi 接受的范围"
            else:
                reason = "没有新的割，但主问题只是提前停止"
            break

    evaluated = None
    if converged:
        # 用独立的分离再算一次目标。主问题的 eta 最多比最坏利润高 cut_tolerance。
        evaluated = evaluate_plan(instance, solution["n"])
        gap = abs(evaluated["objective"] - solution["objective"])
        if gap > cut_tolerance + _acceptance_gap(solution["objective"]):
            raise RuntimeError(
                f"收敛后的主问题目标与逐样本最坏利润相差 {gap:.6g}"
            )
    return _result(
        instance, solution, active, history, converged, reason,
        perf_counter() - started, cut_tolerance, evaluated, mip_gap, adaptive_gap,
    )


def evaluate_plan(instance: dict, counts: list[int]) -> dict:
    """给定一组固定桩数量，逐个样本求最坏利润并汇总目标。"""
    capacity = float(instance["tau"]) * np.asarray(counts, dtype=float)
    operating = 0.0
    by_event = {event: 0.0 for event in instance["date_types"]}
    for event, sample_idx in sample_keys(instance):
        separated = worst_operating_profit(capacity, event, sample_idx, instance)
        direct = piecewise_profit(
            separated["demand"], capacity, instance["rho"], instance["penalty"],
        )
        if abs(direct - separated["profit"]) > _acceptance_gap(separated["profit"]):
            raise RuntimeError(
                f"分离目标与两段利润不一致: {(event, sample_idx)} "
                f"{separated['profit']} vs {direct}"
            )
        weight = float(instance["sample_weight"][event])
        operating += weight * separated["profit"]
        by_event[event] += weight * separated["profit"]
    investment = float(instance["fixed_cost"]) * sum(counts)
    return {
        "objective": -investment + operating,
        "investment_cost": investment,
        "operating_profit": operating,
        "operating_profit_by_date_type": by_event,
    }


def _separate_round(instance, solution, active, supports, started, max_seconds, cut_tolerance,
                    write_cuts: bool = True):
    added = 0
    accounted = 0
    residual = False
    capacity = float(instance["tau"]) * np.asarray(solution["n"], dtype=float)
    rho, penalty = float(instance["rho"]), float(instance["penalty"])
    for event, sample_idx in sample_keys(instance):
        remaining = None
        if max_seconds is not None:
            remaining = max_seconds - (perf_counter() - started)
            # 剩余太短时不要再开分离：Gurobi 时限内未证最优会直接报错。
            if remaining <= 5.0:
                return added, residual, False, "达到总时间上限", accounted
        separated = worst_operating_profit(capacity, event, sample_idx, instance, remaining)
        direct = piecewise_profit(separated["demand"], capacity, rho, penalty)
        if abs(direct - separated["profit"]) > _acceptance_gap(separated["profit"]):
            raise RuntimeError(
                f"分离目标与两段利润不一致: {(event, sample_idx)} "
                f"separation={separated['profit']}, piecewise={direct}"
            )
        worst_profit = separated["profit"]
        # 支持函数和分离各解一次，数值上会有大约 1e-6 的差。
        # 默认容差 1e-4，与 charge centrally、charge locally 的加割容差一致。
        if solution["eta"][event, sample_idx] <= worst_profit + cut_tolerance:
            continue
        pattern = separated["pattern"]
        if pattern not in active[event, sample_idx]:
            if not write_cuts:
                accounted += 1
                print(
                    f"核算到违背，不写入 {event} 样本 {sample_idx}: {pattern}",
                    flush=True,
                )
                continue
            support = _store_support(supports, instance, (event, sample_idx), pattern)
            cut_value = pattern_cut_value(pattern, capacity, support, instance)
            if abs(cut_value - worst_profit) > _acceptance_gap(worst_profit):
                raise RuntimeError(
                    f"新割在当前容量处不等于分离利润: {(event, sample_idx)} "
                    f"cut={cut_value}, separation={worst_profit}"
                )
            active[event, sample_idx].append(pattern)
            added += 1
            print(f"新增割 {event} 样本 {sample_idx}: {pattern}", flush=True)
        else:
            excess = solution["eta"][event, sample_idx] - worst_profit
            # 分离保持 Gurobi 默认相对间隙 1e-4。主问题再收紧后，eta 会贴到割上，
            # 分离利润可以比这条割低零点零几。差不超过分离自己的验收范围时，
            # 与上面“新割必须等于分离利润”用的是同一个范围，不把它当成未收敛。
            if excess <= _acceptance_gap(worst_profit):
                print(
                    f"已有割的数值差 {event} 样本 {sample_idx}，超出 {excess:.6g}，"
                    "在分离间隙内",
                    flush=True,
                )
                continue
            residual = True
            print(f"已有割仍被违反 {event} 样本 {sample_idx}，超出 {excess:.6g}", flush=True)
    return added, residual, True, None, accounted


def _store_support(supports, instance, key, pattern) -> float:
    event, sample_idx = key
    label = (event, sample_idx, pattern)
    if label not in supports:
        lower, upper = local_box(instance, event, sample_idx)
        supports[label] = support_value(pattern, lower, upper, instance)
    return supports[label]


def _result(instance, solution, active, history, converged, reason, elapsed,
            cut_tolerance, evaluated, mip_gap=None, adaptive_gap=False) -> dict:
    decision = None
    if solution is not None:
        counts = solution["n"]
        decision = {
            "taz_ids": list(instance["taz_ids"]),
            "n": counts,
            "capacity": [float(instance["tau"]) * count for count in counts],
            "mobile_chargers": 0,
            "investment_cost": solution["investment_cost"],
            "operating_profit": solution["operating_profit"],
        }
        if evaluated is not None:
            decision["evaluated_operating_profit"] = evaluated["operating_profit"]
            decision["operating_profit_by_date_type"] = evaluated["operating_profit_by_date_type"]
    return {
        "model": "fixed-only",
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "converged": converged,
        "reason": reason,
        "iterations": len(history),
        "objective": None if solution is None else solution["objective"],
        "bound": None if solution is None else solution["bound"],
        "mip_gap": None if solution is None else solution["mip_gap"],
        "elapsed_seconds": elapsed,
        "decision": decision,
        "pattern_count": sum(len(patterns) for patterns in active.values()),
        "history": history,
        "parameters": {
            "rho": instance["rho"],
            "shortage_penalty": instance["penalty"],
            "fixed_capacity": instance["tau"],
            "fixed_unit_cost": instance["fixed_cost"],
            "n_bar": list(instance["n_bar"]),
            "cut_tolerance": cut_tolerance,
            "master_mip_gap": mip_gap,
            "adaptive_gap": adaptive_gap,
            "acceleration": (
                "capacity lifted; unproven master does not add cuts; "
                "previous integer n is the next MIP start"
                + (
                    "; adaptive master gap 1e-3 until no new cut, then "
                    + str(1e-4 if mip_gap is None else mip_gap)
                    if adaptive_gap else ""
                )
            ),
        },
    }

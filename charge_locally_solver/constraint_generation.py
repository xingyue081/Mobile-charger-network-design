"""本地补能的约束生成循环。

循环和集中补能 constraint_generation.py 相同：
解受限主问题，逐个样本调用分离，发现违反就把 pattern 加进主问题。
只有每个样本都被证明没有违反时才算收敛。
分离失败不能记成收敛。

一轮的实际做法：
1. 解当前主问题。上界连续两轮不动时，再取几个同值解一起分离。
2. 从上次停下来的样本继续往后看。
3. 一旦已经发现违反，这一轮最多再看 samples_per_iteration 个样本，
   并且最多准备 cut_cap 条新割。没有发现违反时，这个上限不起作用，
   会把剩下的样本全部看完。
4. 有新割就加进主问题，进入下一轮。
5. 一整圈都没有新割、主问题是 OPTIMAL、也没有数值残差，才记收敛。

因此 samples_per_iteration=8 不是“全程只检查 8 个样本”。
它只限制已经找到违反之后，这一轮还要再扫多远。

加速（与 charge_centrally_solver_1030 对齐）：
- 主问题未证明到间隙时，分离只核算，不写新割；
- 可选 adaptive_gap：加割轮次用较松间隙，无新割后再收紧；
- 热启动与容量提成在 master.py。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from instance import sample_keys
from master import LocalMaster
from patterns import canonical_pattern
from separation import solve_separation


def solve_constraint_generation(instance: dict, supports: dict, *, max_iterations: int = 30) -> dict:
    """从每个样本的第 0 段 pattern 出发，生成被违反的三段割。

    第 0 段是 D 不超过 fixed 容量时的仿射片，斜率 rho、常数 0。
    它一定是利润的一个上界，所以可以作为初始割。
    调用方如果给了 resume_patterns，就从那些已经加入的割接着算。
    """
    if max_iterations < 1:
        raise ValueError("max_iterations 必须是正整数")
    keys = sample_keys(instance)
    zone_count = len(instance["zones"])
    initial = instance.get("resume_patterns") or {key: {(0,) * zone_count} for key in keys}
    started = time.monotonic()
    deadline = None
    if instance.get("wall_clock_seconds") is not None:
        deadline = started + float(instance["wall_clock_seconds"])
    master = LocalMaster(instance, supports, initial)
    history = []
    solution = None
    cursor = 0
    adaptive_gap = bool(instance.get("adaptive_gap"))
    loose_gap = 1e-3
    tight_gap = 1e-4 if instance.get("master_mip_gap") is None else float(instance["master_mip_gap"])
    base_master_time = instance.get("master_time_limit")
    tight_time = 1800.0 if adaptive_gap else None
    certifying = False
    iteration = 0
    try:
        while True:
            iteration += 1
            if iteration > max_iterations:
                max_iterations *= 2
                print(f"达到迭代上限，上限加长到 {max_iterations} 后继续。", flush=True)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return _finish("TIME_LIMIT", False, history, solution, master, started, instance)
            if adaptive_gap:
                round_gap = tight_gap if certifying else loose_gap
                master.set_mip_gap(round_gap)
                round_time = tight_time if certifying else base_master_time
            else:
                round_gap = instance.get("master_mip_gap")
                round_time = base_master_time
            if remaining is not None and round_time is not None:
                instance["master_time_limit"] = min(float(round_time), remaining)
            elif remaining is not None:
                instance["master_time_limit"] = remaining
            elif round_time is not None:
                instance["master_time_limit"] = float(round_time)
            else:
                instance.pop("master_time_limit", None)

            recent = [row.get("objective") for row in history[-2:]]
            stalled = (
                len(recent) == 2
                and None not in recent
                and abs(recent[0] - recent[1]) <= 1e-8 * (1.0 + abs(recent[1]))
            )
            # 连续两轮上界不动时，再找几个同值的主问题解，把这些解一起拿去分离。
            # 这些额外解产生的割同样对所有部署有效。
            instance["master_pool_solutions"] = int(instance.get("stall_pool_solutions") or 1) if stalled else 1
            solution = master.optimize()
            while solution["status"] == "FEASIBLE":
                # 时限内有可行解但没证明到间隙。加长本轮时限再解，不把这个方案当成收敛。
                round_time = 1800.0 if round_time is None else float(round_time) * 2.0
                if certifying:
                    tight_time = round_time
                else:
                    base_master_time = round_time
                print(
                    f"主问题在时限内未证明最优，时限加长到 {round_time:.0f} 秒后重解本轮。",
                    flush=True,
                )
                instance["master_time_limit"] = float(round_time)
                solution = master.optimize()
            row = {
                "iteration": iteration,
                "master_status": solution["status"],
                "objective": solution.get("objective"),
                "bound": solution.get("bound"),
                "master_mip_gap_target": round_gap,
                "added": [],
                "accounted_violations_not_written": 0,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            if instance.get("checkpoint_path"):
                print(json.dumps({
                    "iteration": iteration,
                    "master_status": solution["status"],
                    "objective": solution.get("objective"),
                    "bound": solution.get("bound"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "master_seconds": solution.get("runtime"),
                    "node_count": solution.get("node_count"),
                    "stalled": stalled,
                    "master_mip_gap_target": round_gap,
                }, ensure_ascii=False), flush=True)
            _checkpoint(instance, _finish("RUNNING", False, history, solution, master, started, instance))
            if solution["status"] not in ("OPTIMAL", "FEASIBLE"):
                return _finish("MASTER_NOT_OPTIMAL", False, history, solution, master, started, instance)

            # 主问题没有证明到间隙时，分离只核算这个方案，不把模式写进下一轮。
            write_cuts = solution["status"] == "OPTIMAL"
            points = [solution] if write_cuts else [solution]
            if write_cuts:
                points = [solution, *solution.get("alternates", [])]
            batch = int(instance.get("samples_per_iteration") or len(keys))
            cut_cap = int(instance.get("cut_cap") or len(keys))
            order = keys[cursor:] + keys[:cursor]
            pending = []
            accounted = 0
            examined = 0
            active_residual = False
            for event, sample_idx in order:
                # 已经有待加入的割时，看满 batch 个样本就停。还没有割时继续往下看。
                if examined >= batch and pending:
                    break
                if len(pending) >= cut_cap:
                    break
                examined += 1
                for point in points:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return _finish("TIME_LIMIT", False, history, solution, master, started, instance)
                    if remaining is not None:
                        instance["separation_time_limit"] = remaining
                    oracle = solve_separation(point, event, sample_idx, instance)
                    primary = point is solution
                    if primary:
                        row.setdefault("separations", {})[event, sample_idx] = {
                            "status": oracle["status"],
                            "violation": oracle.get("violation"),
                            "pattern": oracle.get("pattern"),
                        }
                    if not oracle.get("certified_optimal"):
                        # 当前最优解的分离没证完，整次运行停止。并列解没证完则跳过该解。
                        if primary:
                            return _finish("SEPARATION_NOT_CERTIFIED", False, history, solution, master, started, instance)
                        continue
                    if oracle["status"] != "FOUND_VIOLATION":
                        continue
                    tolerance = float(instance["separation_tolerance"])
                    for pattern in oracle.get("patterns") or []:
                        pattern = canonical_pattern(pattern, instance)
                        if pattern in master.active[event, sample_idx] or any(
                            item[2] == pattern and item[0] == event and item[1] == sample_idx for item in pending
                        ):
                            if float(oracle["violation"]) <= tolerance:
                                continue
                            # 分离保持默认相对间隙 1e-4。主问题再收紧后，已有割可以
                            # 看起来差零点零几。不超过分离自己的验收范围时继续。
                            room = 1e-4 * (1.0 + abs(float(oracle.get("worst_profit") or 0.0)))
                            if float(oracle["violation"]) <= room:
                                continue
                            # 这条割已经在主问题里，但分离值仍明显低于 eta。
                            # 这是支持函数和分离模型之间的数值差，不是一条可以再加的新割。
                            # 先继续看别的样本；整圈结束仍有残差时，下面会拒绝宣布收敛。
                            if primary:
                                active_residual = True
                            continue
                        if not write_cuts:
                            accounted += 1
                            if primary:
                                print(
                                    f"核算到违背，不写入 {event} 样本 {sample_idx}: {pattern}",
                                    flush=True,
                                )
                            continue
                        pending.append((event, sample_idx, pattern, oracle["violation"]))
                        if len(pending) >= cut_cap:
                            break
            row["accounted_violations_not_written"] = accounted
            cursor = (cursor + examined) % len(keys)
            if not write_cuts:
                return _finish(
                    "MASTER_NOT_OPTIMAL_NO_CUTS",
                    False, history, solution, master, started, instance,
                )
            if pending:
                for event, sample_idx, pattern, _violation in pending:
                    if pattern in master.active[event, sample_idx]:
                        continue
                    if not master.add_pattern(event, sample_idx, pattern):
                        return _finish("PATTERN_ADDITION_FAILURE", False, history, solution, master, started, instance)
                    row["added"].append((event, sample_idx, pattern))
                row["elapsed_seconds"] = time.monotonic() - started
                if adaptive_gap and certifying:
                    certifying = False
                    print("收紧间隙后出现新割，后面的轮次回到较松的间隙。", flush=True)
                _checkpoint(instance, _finish("RUNNING", False, history, solution, master, started, instance))
                continue
            if examined < len(keys):
                # 这一轮因样本上限或割数量上限提前停下，没有看完全部样本，不能宣布收敛。
                continue
            if active_residual:
                return _finish("ACTIVE_PATTERN_RESIDUAL", False, history, solution, master, started, instance)
            if solution["status"] != "OPTIMAL":
                return _finish("MASTER_NOT_OPTIMAL", False, history, solution, master, started, instance)
            if adaptive_gap and not certifying:
                certifying = True
                print(
                    f"没有新的割。下一轮把主问题间隙从 {loose_gap} 收到 {tight_gap}。",
                    flush=True,
                )
                continue
            return _finish("ALL_SEPARATIONS_WITHIN_TOLERANCE", True, history, solution, master, started, instance)
    finally:
        master.dispose()


def _checkpoint(instance: dict, payload: dict) -> None:
    """每轮把当前割和目标写到磁盘。到时停止后，下一次可以从这些割接着算。"""
    path = instance.get("checkpoint_path")
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list), encoding="utf-8")


def _finish(reason: str, converged: bool, history: list, solution: dict | None,
            master: LocalMaster, started: float, instance: dict | None = None) -> dict:
    """整理返回值。converged 只有 reason 为全部样本都在容差内时才是 True。

    reason 的含义：
    - ALL_SEPARATIONS_WITHIN_TOLERANCE：每个样本都证过，没有新的违反。
    - TIME_LIMIT：墙钟用完。目标仍是当前割下的上界。
    - SEPARATION_NOT_CERTIFIED：某个样本的分离没有证明最优。
    - ACTIVE_PATTERN_RESIDUAL：最坏 pattern 已经在主问题里，但 eta 和分离值还有差距。
    - MASTER_NOT_OPTIMAL：主问题自己没有证明最优。
    - MASTER_NOT_OPTIMAL_NO_CUTS：主问题未证明到间隙，分离只核算，不加割。
    - MAXIMUM_ITERATIONS：到达轮数上限。
    - PATTERN_ADDITION_FAILURE：试图加入一条已经存在的割。
    """
    active = {
        f"{event}:{sample_idx}": [list(pattern) for pattern in sorted(patterns)]
        for (event, sample_idx), patterns in master.active.items()
    }
    adaptive_gap = bool((instance or {}).get("adaptive_gap"))
    mip_gap = None if instance is None else instance.get("master_mip_gap")
    return {
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "converged": converged,
        "reason": reason,
        "iterations": len(history),
        "objective": None if solution is None else solution.get("objective"),
        "bound": None if solution is None else solution.get("bound"),
        "elapsed_seconds": time.monotonic() - started,
        "decision": None if solution is None else {
            "n": solution.get("n"),
            "m": solution.get("m"),
            "z": solution.get("z"),
            "flow": solution.get("flow"),
        },
        "active_patterns": active,
        "pattern_count": sum(len(value) for value in active.values()),
        "acceleration": (
            "fixed/mobile capacity lifted; unproven master does not add cuts; "
            "previous integer solution is the next MIP start"
            + (
                "; adaptive master gap 1e-3 until no new cut, then "
                + str(1e-4 if mip_gap is None else mip_gap)
                if adaptive_gap else ""
            )
        ),
        "history": [
            {
                "iteration": row["iteration"],
                "master_status": row["master_status"],
                "objective": row["objective"],
                "bound": row.get("bound"),
                "master_mip_gap_target": row.get("master_mip_gap_target"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "added": row["added"],
                "accounted_violations_not_written": row.get("accounted_violations_not_written", 0),
            }
            for row in history
        ],
    }

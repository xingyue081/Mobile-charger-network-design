"""用本地补能割平面求解一份已校准数据包。

算法在 constraint_generation.py。换数据只换 --data 指向的目录，
目录里需要 futian_parameters.npz 和 futian_metadata.json。
不带 --solve 时只打印数据口径，不调用求解器。

从头求解：

    python run_20zone.py --g 0 --output outputs/futian_local.json --seconds 14400 --solve

接着上一次已经加入的割：

    python run_20zone.py --g 0 --output outputs/futian_local.json --seconds 14400 \\
        --resume outputs/futian_continued.json --solve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gurobipy as gp

from constraint_generation import solve_constraint_generation
from instance import sample_keys
from real_instance import FUTIAN_DIR, JOINT_SAMPLES, load_zone_dataset
from support import SupportCache


def main() -> None:
    parser = argparse.ArgumentParser(description="本地补能割平面：读取数据包并求解")
    parser.add_argument("--data", type=Path, default=FUTIAN_DIR,
                        help="含 futian_parameters.npz 的目录")
    parser.add_argument("--joint-samples", type=Path, default=JOINT_SAMPLES,
                        help="用来计算 p_kl 的历史日期类型序列")
    parser.add_argument("--g", type=float, required=True, help="本地增量补能成本，需满足 0 <= g <= rho+b")
    parser.add_argument("--rho", type=float, default=None, help="覆盖 business.json 的 rho，只作用于本次")
    parser.add_argument("--shortage-penalty", type=float, default=None, help="覆盖 business.json 的 shortage_penalty，只作用于本次")
    parser.add_argument("--tau", type=float, default=None, help="覆盖 business.json 的 fixed_capacity，只作用于本次")
    parser.add_argument("--kappa", type=float, default=None, help="覆盖 business.json 的 mobile_capacity，只作用于本次")
    parser.add_argument("--fixed-cost", type=float, default=None, help="覆盖 business.json 的 fixed_unit_cost，只作用于本次")
    parser.add_argument("--mobile-cost", type=float, default=None, help="覆盖 business.json 的 mobile_unit_cost，只作用于本次")
    parser.add_argument("--gamma", type=float, default=None, help="覆盖 business.json 的 transport_cost_per_distance，只作用于本次。本地使用单程")
    parser.add_argument("--output", type=Path, required=True, help="结果 JSON 的写出路径")
    parser.add_argument("--seconds", type=float, default=None, help="墙钟上限。不给则一直跑到收敛或迭代上限")
    parser.add_argument("--resume", type=Path, default=None, help="上一次结果 JSON。不给就从第 0 段 pattern 开始")
    parser.add_argument(
        "--master-mip-gap", type=float, default=None,
        help="主问题相对间隙。不给则用 Gurobi 默认 1e-4。只改主问题，不改分离",
    )
    parser.add_argument(
        "--master-mip-gap-abs", type=float, default=None,
        help="主问题绝对空隙。达到后 Gurobi 停。不给则用 Gurobi 默认 1e-10",
    )
    parser.add_argument(
        "--master-time-limit", type=float, default=None,
        help="每一轮主问题最多多少秒。不给则这一轮不单独限时",
    )
    parser.add_argument(
        "--adaptive-gap", action="store_true",
        help="加割的轮次把主问题间隙放宽到 1e-3；没有新割后，下一轮用 master-mip-gap（默认 1e-4），这一轮时限 1800 秒",
    )
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--samples-per-iteration", type=int, default=8)
    parser.add_argument("--cut-cap", type=int, default=24)
    parser.add_argument("--solve", action="store_true", help="真正求解。不加这个参数就只核对数据口径")
    parser.add_argument("--master-log", action="store_true",
                        help="打开主问题的 Gurobi 节点日志。不写则主问题不打印节点")
    args = parser.parse_args()
    if args.master_mip_gap is not None and args.master_mip_gap < 0:
        parser.error("master-mip-gap 不能为负")
    if args.master_mip_gap_abs is not None and args.master_mip_gap_abs < 0:
        parser.error("master-mip-gap-abs 不能为负")
    if args.master_time_limit is not None and args.master_time_limit <= 0:
        parser.error("master-time-limit 必须为正")
    if args.adaptive_gap and args.master_mip_gap is not None and args.master_mip_gap > 1e-3:
        parser.error("adaptive-gap 的松间隙是 1e-3，master-mip-gap 必须更紧或等于 1e-3")

    instance = load_zone_dataset(args.data, args.g, args.joint_samples, business_overrides={
        "rho": args.rho,
        "shortage_penalty": args.shortage_penalty,
        "fixed_capacity": args.tau,
        "mobile_capacity": args.kappa,
        "fixed_unit_cost": args.fixed_cost,
        "mobile_unit_cost": args.mobile_cost,
        "transport_cost_per_distance": args.gamma,
    })
    summary = {
        "data": str(Path(args.data).resolve()),
        "artifact": instance["provenance"]["artifact"],
        "taz_ids": list(instance["taz_ids"]),
        "in_support_counts": instance["provenance"]["in_support_counts"],
        "q_l": instance["q"],
        "epsilon_l": instance["epsilon"],
        "radius": instance["radius"],
        "local_g": instance["local_g"],
        "rho": instance["rho"],
        "penalty": instance["penalty"],
        "tau": instance["tau"],
        "kappa": instance["kappa"],
        "fixed_cost": instance["fixed_cost"],
        "mobile_cost": instance["mobile_cost"],
        "gamma": instance["gamma"],
        "business_overrides": instance["provenance"]["business_overrides"],
        "m_bar": instance["m_bar"],
        "transition_probability": instance["provenance"]["transition_probability"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not args.solve:
        print("未加 --solve，没有启动割平面。")
        return

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.master_mip_gap is not None:
        instance["master_mip_gap"] = args.master_mip_gap
    if args.master_mip_gap_abs is not None:
        instance["master_mip_gap_abs"] = args.master_mip_gap_abs
    if args.master_time_limit is not None:
        instance["master_time_limit"] = args.master_time_limit
    instance["adaptive_gap"] = args.adaptive_gap
    instance["master_log"] = args.master_log
    instance["wall_clock_seconds"] = args.seconds
    instance["checkpoint_path"] = str(output)
    instance["resume_patterns"] = _resume_patterns(instance, args.resume)
    instance["samples_per_iteration"] = args.samples_per_iteration
    instance["cut_cap"] = args.cut_cap
    result = solve_constraint_generation(
        instance, SupportCache(instance), max_iterations=args.max_iterations
    )
    result["gurobi"] = list(gp.gurobi.version())
    result["provenance"] = instance["provenance"]
    result["local_g"] = instance["local_g"]
    result["parameters"] = {
        "rho": instance["rho"],
        "shortage_penalty": instance["penalty"],
        "fixed_unit_cost": instance["fixed_cost"],
        "mobile_unit_cost": instance["mobile_cost"],
        "gamma": instance["gamma"],
        "g": instance["local_g"],
    }
    result["master_mip_gap"] = instance.get("master_mip_gap")
    result["adaptive_gap"] = args.adaptive_gap
    result["scope"] = (
        "本地补能割平面。数据来自调用时的 --data。"
        "到时停止或达到迭代上限都不是最优性证书。"
    )
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "reason": result["reason"],
        "iterations": result["iterations"],
        "objective": result["objective"],
        "bound": result["bound"],
        "elapsed_seconds": result["elapsed_seconds"],
        "pattern_count": result["pattern_count"],
        "output": str(output),
    }, indent=2, ensure_ascii=False, default=list), flush=True)


def _resume_patterns(instance: dict, resume_path: Path | None) -> dict:
    """没有 resume 文件时，每个样本只放第 0 段，由约束生成自己作为初始割。"""
    zone_count = len(instance["zones"])
    if resume_path is None:
        return {key: {(0,) * zone_count} for key in sample_keys(instance)}
    saved = json.loads(Path(resume_path).read_text(encoding="utf-8"))["active_patterns"]
    patterns = {}
    for event, sample_idx in sample_keys(instance):
        rows = saved.get(f"{event}:{sample_idx}")
        if not rows:
            patterns[event, sample_idx] = {(0,) * zone_count}
            continue
        converted = {tuple(int(piece) for piece in pattern) for pattern in rows}
        if any(len(pattern) != zone_count for pattern in converted):
            raise ValueError(f"{event}:{sample_idx} 的 pattern 长度不是 {zone_count}")
        patterns[event, sample_idx] = converted
    return patterns


if __name__ == "__main__":
    main()

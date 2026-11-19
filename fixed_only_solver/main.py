"""求解 fixed-only 对照：只有固定桩，没有移动桩。

在本目录运行。不带 --solve 时只核对数据包，不调用割平面。

    python main.py --self-check
    python main.py
    python main.py --solve --output outputs/futian_fixed_only.json

数据默认是 /Users/xingyueyi/Desktop/CG_solver/futian，与 charge centrally、
charge locally 使用同一份 npz 和 business.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np

# 允许用绝对路径启动，而不必先 cd 到本目录。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import FUTIAN_DIR, load_dataset
from self_check import run_self_check
from solve import solve_fixed_only


def main() -> None:
    parser = argparse.ArgumentParser(description="fixed-only 对照模型：只决定固定桩数量")
    parser.add_argument("--data", type=Path, default=FUTIAN_DIR,
                        help="含 *_parameters.npz 和 business.json 的目录。福田文件名仍是 futian_parameters.npz")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果 JSON。只在 --solve 时写出，默认写到本目录 outputs/")
    parser.add_argument("--seconds", type=float, default=None, help="墙钟上限，秒。不给则跑到收敛或迭代上限")
    parser.add_argument("--max-iterations", type=int, default=100, help="割平面最多迭代多少轮，默认 100")
    parser.add_argument("--cut-tolerance", type=float, default=1e-4,
                        help="eta 比分离出的最坏利润高出多少才加割，默认 1e-4")
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
    parser.add_argument("--rho", type=float, default=None, help="覆盖 business.json 的 rho，只作用于本次")
    parser.add_argument("--shortage-penalty", type=float, default=None,
                        help="覆盖 business.json 的 shortage_penalty，只作用于本次")
    parser.add_argument("--tau", type=float, default=None, help="覆盖 business.json 的 fixed_capacity，只作用于本次")
    parser.add_argument("--fixed-cost", type=float, default=None,
                        help="覆盖 business.json 的 fixed_unit_cost，只作用于本次")
    parser.add_argument("--solve", action="store_true", help="求解。不加这个参数就只核对数据口径")
    parser.add_argument("--self-check", action="store_true",
                        help="运行解析例子、两区全枚举和福田数据包核对，不求解福田")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.max_iterations <= 0:
        parser.error("max-iterations 必须为正")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("seconds 必须为正")
    if args.cut_tolerance < 0:
        parser.error("cut-tolerance 不能为负")
    if args.master_mip_gap is not None and args.master_mip_gap < 0:
        parser.error("master-mip-gap 不能为负")
    if args.master_mip_gap_abs is not None and args.master_mip_gap_abs < 0:
        parser.error("master-mip-gap-abs 不能为负")
    if args.master_time_limit is not None and args.master_time_limit <= 0:
        parser.error("master-time-limit 必须为正")
    if args.adaptive_gap and args.master_mip_gap is not None and args.master_mip_gap > 1e-3:
        parser.error("adaptive-gap 的松间隙是 1e-3，master-mip-gap 必须更紧或等于 1e-3")

    instance = load_dataset(args.data, business_overrides={
        "rho": args.rho,
        "shortage_penalty": args.shortage_penalty,
        "fixed_capacity": args.tau,
        "fixed_unit_cost": args.fixed_cost,
    })
    summary = {
        "model": "fixed-only",
        "mobile_chargers": 0,
        "data": instance["provenance"]["data_dir"],
        "taz_ids": list(instance["taz_ids"]),
        "sample_count": instance["provenance"]["sample_count"],
        "q_l": instance["probability"],
        "epsilon_l": instance["epsilon"],
        "radius": instance["radius"],
        "rho": instance["rho"],
        "shortage_penalty": instance["penalty"],
        "fixed_capacity": instance["tau"],
        "fixed_unit_cost": instance["fixed_cost"],
        "business_overrides": instance["provenance"]["business_overrides"],
        "excluded_business_fields": instance["provenance"]["excluded_business_fields"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not args.solve:
        print("未加 --solve，没有启动割平面。")
        return

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / "outputs" / "futian_fixed_only.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = solve_fixed_only(
        instance,
        max_iterations=args.max_iterations,
        max_seconds=args.seconds,
        cut_tolerance=args.cut_tolerance,
        mip_gap=args.master_mip_gap,
        mip_gap_abs=args.master_mip_gap_abs,
        master_time_limit=args.master_time_limit,
        adaptive_gap=args.adaptive_gap,
    )
    parameter_name = Path(instance["provenance"]["parameters"]).name
    result["dataset"] = parameter_name.removesuffix("_parameters.npz")
    result["gurobi"] = list(gp.gurobi.version())
    result["parameters"] = {
        "rho": instance["rho"],
        "shortage_penalty": instance["penalty"],
        "fixed_unit_cost": instance["fixed_cost"],
        "fixed_capacity": instance["tau"],
    }
    result["mobile_chargers"] = 0
    result["provenance"] = instance["provenance"]
    result["ambiguity_set"] = {
        "sample_box": "[max(0, d_i - epsilon_l), d_i + epsilon_l]",
        "ellipsoid": "(D - mu)^T Sigma_inv (D - mu) <= r^2",
        "radius": instance["radius"],
        "epsilon_l": instance["epsilon"],
        "q_l": instance["probability"],
        "sample_count": instance["provenance"]["sample_count"],
        "sample_weight": "q_l / N_l,Omega，N_l,Omega 是 npz 中该日期类型的样本行数",
    }
    result["scope"] = (
        "fixed-only 对照。只有固定桩数量 n_i，容量为 tau*n_i。"
        "没有 hub、移动桩、调度和运输成本。"
        "歧义集与数据包里的盒子和共享椭球一致。"
        "到时停止或达到迭代上限都不是最优性证书。"
    )
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "reason": result["reason"],
        "iterations": result["iterations"],
        "objective": result["objective"],
        "bound": result["bound"],
        "elapsed_seconds": result["elapsed_seconds"],
        "fixed_chargers": None if result["decision"] is None else sum(result["decision"]["n"]),
        "mobile_chargers": 0,
        "output": str(output),
    }, indent=2, ensure_ascii=False), flush=True)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不能写入 JSON 的类型 {type(value)}")


if __name__ == "__main__":
    main()

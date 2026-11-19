"""求解福田 charge-centrally 的 SAA。

    python main.py --self-check
    python main.py --instance futian --dry-run
    python main.py --instance futian --output outputs/futian_all.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import format_summary, load_futian


def _jsonable_decision(decision: dict) -> dict:
    return {
        "x": {str(key): int(value) for key, value in decision["x"].items()},
        "m": {str(key): int(value) for key, value in decision["m"].items()},
        "n": {str(key): int(value) for key, value in decision["n"].items()},
        "y": {f"{zone},{hub}": int(value) for (zone, hub), value in decision["y"].items()},
        "z": {
            f"{zone},{hub},{date_index}": int(value)
            for (zone, hub, date_index), value in decision["z"].items()
        },
    }


def _result_document(data: dict, solved: dict) -> dict:
    decision = solved.get("decision") or {}
    opened = decision.get("x") or {}
    fleet = decision.get("m") or {}
    installed = decision.get("n") or {}
    open_hubs = [hub for hub, value in opened.items() if int(value) == 1]
    names = data["date_names"]
    return {
        "model": "charge_centrally_saa",
        "dataset": data["name"],
        "sample_set": data["sample_set"],
        "objective": solved.get("objective"),
        "bound": solved.get("bound"),
        "mip_gap": solved.get("mip_gap"),
        "runtime": solved.get("runtime"),
        "status_name": solved.get("status_name"),
        "node_count": solved.get("node_count"),
        "investment_cost": solved.get("investment_cost"),
        "sample_average_profit": solved.get("sample_average_profit"),
        "fixed_chargers": None if not installed else int(sum(installed.values())),
        "mobile_chargers": None if not fleet else int(sum(fleet.values())),
        "hub_count": len(open_hubs),
        "hub_indexes": open_hubs,
        "zone_ids": [int(value) for value in data["zone_ids"]],
        "date_names": list(names),
        "q_l": {name: float(value) for name, value in zip(names, data["q_l"])},
        "sample_counts": {
            names[index]: int(data["samples"][index].shape[0]) for index in range(data["S_num"])
        },
        "sample_dates": data["sample_dates"],
        "rho": data["rho"],
        "shortage_penalty": data["b"],
        "fixed_capacity": data["tau"],
        "mobile_capacity": data["kappa"],
        "fixed_unit_cost": data["c_f"],
        "mobile_unit_cost": data["c_m"],
        "record": data["record"],
        "decision": None if not decision else _jsonable_decision(decision),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="求解 charge-centrally 的样本平均近似")
    parser.add_argument("--instance", default="futian", choices=("futian",), help="目前只有福田实例")
    parser.add_argument(
        "--samples",
        default="all",
        choices=("all", "support"),
        help="all=训练窗全部日期；support=与 DRO 相同的椭球内样本",
    )
    parser.add_argument("--mip-gap", type=float, default=None, help="相对 MIP gap。不写则用 Gurobi 默认 1e-4")
    parser.add_argument("--time-limit", type=float, default=None, help="求解秒数上限。不写则不限制")
    parser.add_argument("--log", action="store_true", help="打印 Gurobi 节点日志")
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印读入的数据，不求解")
    parser.add_argument("--self-check", action="store_true", help="用可枚举的小例子核对线性化和目标")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--shortage-penalty", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None, help="覆盖 fixed_capacity")
    parser.add_argument("--kappa", type=float, default=None, help="覆盖 mobile_capacity")
    parser.add_argument("--fixed-cost", type=float, default=None)
    parser.add_argument("--mobile-cost", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="覆盖 transport_cost_per_distance")
    parser.add_argument("--data-dir", type=Path, default=None, help="含 futian_parameters.npz 与 business.json 的目录")
    parser.add_argument("--output", type=Path, default=None, help="把目标和调度决策写入 JSON")
    args = parser.parse_args()
    if args.self_check:
        from model import self_check

        self_check()
        return
    if args.mip_gap is not None and args.mip_gap < 0:
        parser.error("mip-gap 不能为负")
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("time-limit 必须为正")
    overrides = {
        "rho": args.rho,
        "shortage_penalty": args.shortage_penalty,
        "fixed_capacity": args.tau,
        "mobile_capacity": args.kappa,
        "fixed_unit_cost": args.fixed_cost,
        "mobile_unit_cost": args.mobile_cost,
        "transport_cost_per_distance": args.gamma,
    }
    data = load_futian(args.samples, overrides, args.data_dir)
    print(format_summary(data))
    if args.dry_run:
        return
    from model import solve_saa

    solved = solve_saa(data, {
        "mip_gap": args.mip_gap,
        "time_limit": args.time_limit,
        "output_flag": args.log,
    })
    print(
        f"状态 {solved['status_name']}，目标 {solved['objective']}，"
        f"bound {solved['bound']}，MIP gap {solved['mip_gap']}，用时 {solved['runtime']}"
    )
    if solved.get("decision") is None:
        raise SystemExit("SAA 没有可行解")
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_result_document(data, solved), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()

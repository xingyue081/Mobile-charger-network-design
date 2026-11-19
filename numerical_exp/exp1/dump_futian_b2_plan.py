"""从 futian b=2 自适应割日志恢复割，再解一次主问题，导出可绘图的完整决策。"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

ROOT = Path("/Users/xingyueyi/Desktop/CG_solver/charge_centrally_solver_1030")
sys.path.insert(0, str(ROOT))

from main import load_model_data
from master import solve_master_problem

LOG = ROOT / "outputs" / "futian_b2_adaptgap.log"
OUT = Path(__file__).resolve().parent / "inputs" / "futian_charge_centrally_b2_plan.json"
PATTERN = re.compile(r"情形 (\d+) 样本 (\d+): 新增恶劣模式 (\([^)\n]*\))")


def support_height(pattern, lower, upper, mu, sigma_inv, radius, rho, penalty, zone_count):
    model = gp.Model("Compute_h")
    model.Params.OutputFlag = 0
    demand = model.addVars(zone_count, lb=lower, ub=upper, name="D")
    diff = {i: demand[i] - mu[i] for i in range(zone_count)}
    model.addQConstr(
        gp.quicksum(
            sigma_inv[i][j] * diff[i] * diff[j]
            for i in range(zone_count) for j in range(zone_count)
        )
        <= radius ** 2
    )
    model.setObjective(
        gp.quicksum(
            (rho + penalty) * demand[i] * pattern[i] - rho * demand[i]
            for i in range(zone_count)
        ),
        GRB.MAXIMIZE,
    )
    for focus in (0, 2, 3):
        model.Params.NumericFocus = focus
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return float(model.ObjVal)
        if model.SolCount > 0 and model.Status in (getattr(GRB, "SUBOPTIMAL", 13), 13):
            return float(model.ObjVal)
    raise RuntimeError(f"h 求解失败 status={model.Status} pattern={pattern}")


def load_patterns(toy_data):
    zone_count = toy_data["I_num"]
    samples = toy_data["samples"]
    epsilon_l = toy_data["epsilon_l"]
    mu = toy_data["mu"]
    sigma_inv = toy_data["Sigma_inv"]
    radius, rho, penalty = toy_data["r"], toy_data["rho"], toy_data["b"]
    active = {}
    for date_type, rows in samples.items():
        eps = epsilon_l[date_type]
        for sample_idx, demand in enumerate(rows):
            lower = np.maximum(0, demand - eps)
            upper = demand + eps
            zeros = tuple(0 for _ in range(zone_count))
            ones = tuple(1 for _ in range(zone_count))
            active[(date_type, sample_idx)] = [zeros, ones]
            toy_data["h_val"][(date_type, sample_idx, zeros)] = support_height(
                zeros, lower, upper, mu, sigma_inv, radius, rho, penalty, zone_count
            )
            toy_data["h_val"][(date_type, sample_idx, ones)] = support_height(
                ones, lower, upper, mu, sigma_inv, radius, rho, penalty, zone_count
            )
    added = 0
    for line in LOG.read_text(encoding="utf-8").splitlines():
        matched = PATTERN.search(line)
        if matched is None:
            continue
        date_type = int(matched.group(1))
        sample_idx = int(matched.group(2))
        pattern = tuple(ast.literal_eval(matched.group(3)))
        if pattern in active[(date_type, sample_idx)]:
            continue
        demand = samples[date_type][sample_idx]
        eps = epsilon_l[date_type]
        lower = np.maximum(0, demand - eps)
        upper = demand + eps
        toy_data["h_val"][(date_type, sample_idx, pattern)] = support_height(
            pattern, lower, upper, mu, sigma_inv, radius, rho, penalty, zone_count
        )
        active[(date_type, sample_idx)].append(pattern)
        added += 1
        if added % 400 == 0:
            print(f"已恢复 {added} 条割", flush=True)
    print(f"日志中的割已恢复，共 {added} 条", flush=True)
    return active


def main() -> None:
    toy_data = load_model_data("futian", {"shortage_penalty": 2.0})
    active = load_patterns(toy_data)
    solved, _, _, _, decisions = solve_master_problem(
        active,
        toy_data,
        {"mip_gap": 1e-4, "time_limit": 7200.0},
        log_file=Path(__file__).resolve().parent / "inputs" / "futian_central_b2_dump_gurobi.log",
    )
    # 绘图只需完整整数决策；认证目标约 66370.72。
    # 允许 TIME_LIMIT/SUBOPTIMAL，只要已有解且相对间隙不超过 2e-4。
    gap = decisions.get("mip_gap")
    ok = (
        solved
        and decisions.get("x") is not None
        and gap is not None
        and float(gap) <= 2e-4
    )
    if not ok:
        raise SystemExit(
            f"主问题未给出可用解: status={decisions.get('status_name')} gap={gap}"
        )
    zone_ids = [int(value) for value in toy_data["zone_ids"]]
    payload = {
        "dataset": "futian",
        "shortage_penalty": 2.0,
        "objective": float(decisions["obj"]),
        "bound": float(decisions["obj_bound"]),
        "mip_gap": float(decisions["mip_gap"]),
        "status_name": decisions["status_name"],
        "zone_ids": zone_ids,
        "decision": {
            "x": {str(j): int(decisions["x"][j]) for j in range(toy_data["J_num"])},
            "m": {str(j): int(decisions["m"][j]) for j in range(toy_data["J_num"])},
            "n": {str(i): int(decisions["n"][i]) for i in range(toy_data["I_num"])},
            "y": {
                f"{i},{j}": int(decisions["y"][(i, j)])
                for i in range(toy_data["I_num"])
                for j in range(toy_data["J_num"])
            },
            "z": {
                f"{i},{j},{l}": int(decisions["z"][(i, j, l)])
                for i in range(toy_data["I_num"])
                for j in range(toy_data["J_num"])
                for l in range(toy_data["S_num"])
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    open_hubs = [j for j, opened in decisions["x"].items() if int(opened) == 1]
    print(
        f"已写入 {OUT}: obj={payload['objective']:.6g} "
        f"fixed={sum(payload['decision']['n'].values())} "
        f"mobile={sum(payload['decision']['m'].values())} hubs={open_hubs}",
        flush=True,
    )


if __name__ == "__main__":
    main()

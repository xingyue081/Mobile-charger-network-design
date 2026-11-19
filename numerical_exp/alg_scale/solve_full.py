"""全量 pattern 主问题。先算全部 h，再用剩余墙钟交给 Gurobi。"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "charge_centrally_solver_1030"))

from main import load_model_data
from master import solve_master_problem
from oracle import compute_h_val

from deployment import attach_deployment
from settings import BUSINESS, DATA_ROOT, DATE_TYPES, FULL_MODEL_COUNTS, LOCAL_G, OUTPUT_ROOT, WALL_SECONDS


def _patterns(zone_count: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=zone_count))


def _support_value(pattern, lower, upper, data: dict) -> float:
    """Gurobi 对个别凸二次约束给出可行但未标成最优的解时，间隙足够小就采用该可行值。"""
    try:
        return compute_h_val(
            pattern, lower, upper, data["mu"], data["Sigma_inv"],
            data["r"], data["rho"], data["b"], data["I_num"],
        )
    except RuntimeError as error:
        if "status=13" not in str(error):
            raise
    zone_count = data["I_num"]
    model = gp.Model("Compute_h_suboptimal")
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 1)
    demand = model.addVars(zone_count, lb=lower, ub=upper)
    centered = {zone: demand[zone] - data["mu"][zone] for zone in range(zone_count)}
    precision = data["Sigma_inv"]
    model.addQConstr(gp.quicksum(
        precision[left][right] * centered[left] * centered[right]
        for left in range(zone_count) for right in range(zone_count)
    ) <= data["r"] ** 2)
    model.setObjective(gp.quicksum(
        (data["rho"] + data["b"]) * demand[zone] * pattern[zone] - data["rho"] * demand[zone]
        for zone in range(zone_count)
    ), GRB.MAXIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"compute_h_val 没有可行解：status={model.Status}, pattern={pattern}")
    gap = abs(model.ObjBound - model.ObjVal) / max(1.0, abs(model.ObjVal))
    if gap > 1e-6:
        raise RuntimeError(
            f"compute_h_val 间隙过大：status={model.Status}, gap={gap}, pattern={pattern}"
        )
    return float(model.ObjVal)


def _fill(data: dict, patterns: list[tuple[int, ...]], deadline: float) -> tuple[dict, int, bool]:
    active = {}
    done = 0
    total = sum(len(data["samples"][date_type]) for date_type in range(data["S_num"])) * len(patterns)
    for date_type in range(data["S_num"]):
        epsilon = data["epsilon_l"][date_type]
        for sample_index, demand in enumerate(data["samples"][date_type]):
            lower = np.maximum(0.0, np.asarray(demand, dtype=float) - epsilon)
            upper = np.asarray(demand, dtype=float) + epsilon
            active[date_type, sample_index] = patterns
            for pattern in patterns:
                if time.monotonic() >= deadline:
                    return active, done, False
                data["h_val"][date_type, sample_index, pattern] = _support_value(
                    pattern, lower, upper, data,
                )
                done += 1
                if done % 500 == 0 or done == total:
                    print(f"k={data['I_num']} h {done}/{total}", flush=True)
    return active, done, True


def solve_full(k: int, wall_seconds: float = WALL_SECONDS) -> dict:
    started = time.monotonic()
    deadline = started + wall_seconds
    data = load_model_data("futian", data_dir=DATA_ROOT / f"k{k}")
    patterns = _patterns(data["I_num"])
    os.environ["CG_THREADS"] = "1"
    active, built, finished = _fill(data, patterns, deadline)
    os.environ["CG_THREADS"] = "4"
    h_seconds = time.monotonic() - started
    output_dir = OUTPUT_ROOT / f"k{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "full_pattern_milp",
        "k": k,
        "dataset": "futian",
        "parameters": {
            "rho": data["rho"],
            "shortage_penalty": data["b"],
            "fixed_capacity": data["tau"],
            "mobile_capacity": data["kappa"],
            "fixed_unit_cost": data["c_f"],
            "mobile_unit_cost": data["c_m"],
            "gamma": BUSINESS["transport_cost_per_distance"],
            "hub_open_cost": data["f"][0],
            "local_g_not_used": LOCAL_G,
        },
        "pattern_count_per_sample": len(patterns),
        "heights_built": built,
        "h_seconds": h_seconds,
        "zone_ids": [int(value) for value in data["zone_ids"]],
        "sample_counts": {
            DATE_TYPES[date_type]: len(data["samples"][date_type])
            for date_type in range(data["S_num"])
        },
    }
    if not finished:
        payload.update({
            "status_name": "MODEL_NOT_BUILT",
            "stop_reason": "模型未能在时限内建立",
            "converged": False,
            "objective": None,
            "bound": None,
            "gap": None,
            "solve_seconds": 0.0,
            "decision": None,
            "deployment": None,
        })
    else:
        remaining = max(deadline - time.monotonic(), 1.0)
        solved, _, _, _, decisions = solve_master_problem(
            active, data, {"mip_gap": 1e-4, "time_limit": remaining},
        )
        payload["solve_seconds"] = None if not solved else float(decisions["runtime"])
        payload["status_name"] = None if not solved else decisions["status_name"]
        payload["converged"] = bool(solved and decisions["status_name"] == "OPTIMAL")
        payload["stop_reason"] = None if solved else "主问题没有可行解"
        if solved:
            upper = float(decisions["obj_bound"])
            lower = float(decisions["obj"])
            payload["objective"] = lower
            payload["bound"] = upper
            payload["gap"] = None if upper == 0.0 else (upper - lower) / abs(upper)
            payload["mip_gap"] = float(decisions["mip_gap"])
            attach_deployment(payload, data["zone_ids"], decisions)
        else:
            payload["objective"] = None
            payload["bound"] = None
            payload["gap"] = None
            payload["decision"] = None
            payload["deployment"] = None
    payload["elapsed_seconds"] = time.monotonic() - started
    destination = output_dir / "full_milp.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"k={k} full status={payload['status_name']} obj={payload['objective']} "
        f"h_s={h_seconds:.1f}",
        flush=True,
    )
    return payload


def main() -> None:
    for k in FULL_MODEL_COUNTS:
        solve_full(k)


if __name__ == "__main__":
    main()

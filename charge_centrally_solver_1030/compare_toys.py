"""三个小例子：全模式稠密主问题、全模式稀疏主问题、割平面，三者目标应一致。

稠密主问题把每条割里的运输成本按 z 直接展开，用来检查稀疏改写没有改可行域。
全模式是把每个样本的全部 2^|I| 个缺电模式放进同一次主问题。
"""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from main import build_toy_data, run_cutting_plane_algorithm
from master import solve_master_problem
from oracle import compute_h_val

TIGHT = {
    "master_mip_gap": 0.0,
    "master_mip_gap_abs": 0.0,
    "optimality_tol": 1e-9,
    "master_time_limit": 600.0,
    "max_iterations": 100,
    "max_total_seconds": None,
    "cut_tolerance": 0.0,
    "separation_mip_gap": None,
}
MASTER_LIMITS = {
    "mip_gap": 0.0,
    "mip_gap_abs": 0.0,
    "optimality_tol": 1e-9,
    "time_limit": 600.0,
}


def build_toy2_data():
    """2 区、2 个日期类型、每个日期类型 2 个样本。手写，不是福田数据。"""
    demand = np.array([
        [20.0, 8.0],
        [18.0, 9.0],
        [6.0, 22.0],
        [7.0, 21.0],
    ])
    mu = demand.mean(axis=0)
    sigma = np.diag([4.0, 4.0])
    distance = np.array([[0.0, 5.0], [5.0, 0.0]])
    return {
        "name": "toy2",
        "zone_ids": (0, 1),
        "hub_ids": (0, 1),
        "date_names": ("workday", "weekend"),
        "I_num": 2,
        "J_num": 2,
        "S_num": 2,
        "samples": {0: [demand[0], demand[1]], 1: [demand[2], demand[3]]},
        "q_l": [0.5, 0.5],
        "epsilon_l": [1.0, 1.2],
        "mu": mu,
        "Sigma_inv": np.linalg.inv(sigma),
        "r": 8.0,
        "f": [10.0, 12.0],
        "c_m": 5.0,
        "c_f": 4.0,
        "rho": 2.0,
        "b": 1.0,
        "tau": 10.0,
        "kappa": 8.0,
        "M_m": [2, 2],
        "n_bar": [3, 3],
        "M_z": np.full((2, 2), 2.0),
        "t_ij": 2.0 * 0.05 * distance,
        "h_val": {},
    }


def build_zone4_data():
    """合成 4 区、2 类日期、每类 2 个样本。不是真实 TAZ 数据。"""
    demand_data = np.array([
        [30.0, 12.0, 18.0, 9.0],
        [28.0, 14.0, 16.0, 11.0],
        [10.0, 32.0, 14.0, 20.0],
        [12.0, 30.0, 12.0, 22.0],
    ])
    mu = np.mean(demand_data, axis=0)
    sigma = np.diag([25.0, 25.0, 25.0, 25.0])
    distance = np.array([
        [0.0, 6.0, 9.0, 12.0],
        [6.0, 0.0, 5.0, 8.0],
        [9.0, 5.0, 0.0, 4.0],
        [12.0, 8.0, 4.0, 0.0],
    ])
    return {
        "name": "synthetic_4zone",
        "zone_ids": (0, 1, 2, 3),
        "hub_ids": (0, 1, 2, 3),
        "date_names": ("workday", "weekend"),
        "I_num": 4,
        "J_num": 4,
        "S_num": 2,
        "samples": {0: [demand_data[0], demand_data[1]], 1: [demand_data[2], demand_data[3]]},
        "q_l": [0.5, 0.5],
        "epsilon_l": [1.0, 1.5],
        "mu": mu,
        "Sigma_inv": np.linalg.inv(sigma),
        "r": 4.0,
        "f": [30.0, 30.0, 30.0, 30.0],
        "c_m": 41.0,
        "c_f": 40.0,
        "rho": 2.0,
        "b": 2.0,
        "tau": 20.0,
        "kappa": 15.0,
        "M_m": [5, 5, 5, 5],
        "n_bar": [8, 8, 8, 8],
        "M_z": np.full((4, 4), 5.0),
        "t_ij": 2 * 0.01 * distance,
        "h_val": {},
    }


def with_all_patterns(model_data):
    """每个样本放入全部缺电模式，并预先算好 h。"""
    data = deepcopy(model_data)
    data["h_val"] = {}
    zones = data["I_num"]
    patterns = list(product((0, 1), repeat=zones))
    active = {}
    for date_type in range(data["S_num"]):
        epsilon = data["epsilon_l"][date_type]
        for sample_index, demand in enumerate(data["samples"][date_type]):
            lower = np.maximum(0, np.asarray(demand, dtype=float) - epsilon)
            upper = np.asarray(demand, dtype=float) + epsilon
            active[date_type, sample_index] = patterns
            for pattern in patterns:
                data["h_val"][date_type, sample_index, pattern] = compute_h_val(
                    pattern, lower, upper, data["mu"], data["Sigma_inv"],
                    data["r"], data["rho"], data["b"], zones,
                )
    return data, active


def solve_dense_full(data, active):
    """旧写法：每条割直接展开 sum t_ij z_ijl，不引入运输变量。"""
    zones = range(data["I_num"])
    hubs = range(data["J_num"])
    dates = range(data["S_num"])
    model = gp.Model("dense_full_patterns")
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 0.0)
    model.setParam("MIPGapAbs", 0.0)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("TimeLimit", 600.0)
    x = model.addVars(hubs, vtype=GRB.BINARY, name="x")
    y = model.addVars(zones, hubs, vtype=GRB.BINARY, name="y")
    m = model.addVars(hubs, vtype=GRB.INTEGER, name="m")
    n = model.addVars(zones, vtype=GRB.INTEGER, name="n")
    z = model.addVars(zones, hubs, dates, vtype=GRB.INTEGER, name="z")
    eta = {}
    for (date_type, sample_index), patterns in active.items():
        eta[date_type, sample_index] = model.addVar(lb=-GRB.INFINITY, name=f"eta[{date_type},{sample_index}]")
        for pattern in patterns:
            revenue = gp.quicksum(
                (data["rho"] + data["b"]) * pattern[i] * (
                    data["tau"] * n[i] + data["kappa"] * gp.quicksum(z[i, j, date_type] for j in hubs)
                )
                for i in zones
            )
            transport = gp.quicksum(
                data["t_ij"][i, j] * z[i, j, date_type] for i in zones for j in hubs
            )
            h_value = data["h_val"][date_type, sample_index, pattern]
            model.addConstr(eta[date_type, sample_index] <= -transport + revenue - h_value)
    for j in hubs:
        for date_type in dates:
            model.addConstr(gp.quicksum(z[i, j, date_type] for i in zones) <= m[j])
        model.addConstr(m[j] <= data["M_m"][j] * x[j])
    for i in zones:
        model.addConstr(gp.quicksum(y[i, j] for j in hubs) <= 1)
        model.addConstr(n[i] <= data["n_bar"][i])
        for j in hubs:
            model.addConstr(y[i, j] <= x[j])
            for date_type in dates:
                model.addConstr(z[i, j, date_type] <= data["M_z"][i, j] * y[i, j])
    model.setObjective(
        - gp.quicksum(data["f"][j] * x[j] for j in hubs)
        - data["c_m"] * gp.quicksum(m[j] for j in hubs)
        - data["c_f"] * gp.quicksum(n[i] for i in zones)
        + gp.quicksum(
            (data["q_l"][date_type] / len(data["samples"][date_type])) * eta[date_type, sample_index]
            for date_type in dates
            for sample_index in range(len(data["samples"][date_type]))
        ),
        GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"稠密全枚举未证明最优: status={model.Status}")
    return float(model.ObjVal)


def compare_one(name, source):
    full_data, active = with_all_patterns(source)
    dense = solve_dense_full(full_data, active)
    solved, _, _, _, decisions = solve_master_problem(active, full_data, MASTER_LIMITS)
    if not solved or decisions.get("status_name") != "OPTIMAL":
        raise RuntimeError(f"稀疏全枚举未证明最优: {decisions.get('status_name')}")
    sparse = float(decisions["obj"])
    cg = run_cutting_plane_algorithm(deepcopy(source), TIGHT)
    cg_obj = None if cg["objective"] is None else float(cg["objective"])
    dense_gap = abs(sparse - dense)
    cg_gap = None if cg_obj is None else abs(cg_obj - dense)
    passed = (
        cg["status_name"] == "OPTIMAL"
        and cg["stop_reason"] == "没有新的割，主问题相对间隙已达到 Gurobi 接受的范围"
        and dense_gap < 1e-5
        and cg_gap is not None
        and cg_gap < 1e-5
    )
    row = {
        "name": name,
        "zones": source["I_num"],
        "dense_full": dense,
        "sparse_full": sparse,
        "cutting_plane": cg_obj,
        "sparse_minus_dense": sparse - dense,
        "cg_minus_dense": None if cg_obj is None else cg_obj - dense,
        "iterations": cg["iterations"],
        "status": cg["status_name"],
        "stop_reason": cg["stop_reason"],
        "passed": passed,
    }
    print(
        f"[{name}] dense={dense:.10g} sparse={sparse:.10g} cg={cg_obj} "
        f"|cg-dense|={cg_gap} iterations={cg['iterations']} PASS={passed}",
        flush=True,
    )
    return row


def main():
    rows = [
        compare_one("toy2", build_toy2_data()),
        compare_one("toy", build_toy_data()),
        compare_one("synthetic_4zone", build_zone4_data()),
    ]
    report = {"passed": all(row["passed"] for row in rows), "instances": rows}
    destination = Path(__file__).resolve().parent / "outputs" / "toy_compare.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ALL_PASS={report['passed']} report={destination}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

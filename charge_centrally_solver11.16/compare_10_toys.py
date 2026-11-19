"""10 个小例子：全模式稠密枚举 vs 加速后的割平面，目标应一致。

暴力侧把每个样本的全部 2^|I| 个 pattern 放进同一次主问题（稠密写法）。
算法侧跑本目录的切割平面（启发式分离 + 固定 a_i + 可并行）。

至少若干例子的最优解必须实际使用 mobile charger（sum m > 0）。
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
from oracle import compute_h_val

# Exactness check: no cut/sample throttling, prove master to zero gap.
TIGHT = {
    "master_mip_gap": 0.0,
    "master_mip_gap_abs": 0.0,
    "optimality_tol": 1e-9,
    "master_time_limit": 600.0,
    "max_iterations": 200,
    "max_total_seconds": None,
    "cut_tolerance": 0.0,
    "separation_mip_gap": None,
    "separation_workers": 1,
    "cut_cap": 10**9,
    "samples_per_iteration": 10**9,
    "allow_cuts_on_timelimit": True,
    "adaptive_gap": False,
}
OBJ_TOL = 1e-5


def _base_two_zone(
    name,
    demand_rows,
    date_types,
    *,
    epsilon_l,
    q_l,
    f,
    c_m,
    c_f,
    rho,
    b,
    tau,
    kappa,
    M_m,
    n_bar,
    r,
    distance,
    gamma=0.05,
    require_mobile=False,
):
    demand = np.asarray(demand_rows, dtype=float)
    mu = demand.mean(axis=0)
    # Diagonal covariance keeps the ellipsoid simple but nonempty.
    sigma = np.diag(np.maximum(np.var(demand, axis=0), 1.0))
    samples = {k: [] for k in sorted(set(date_types))}
    for row, date_type in zip(demand, date_types):
        samples[date_type].append(row.copy())
    I_num = demand.shape[1]
    J_num = I_num
    return {
        "name": name,
        "require_mobile": require_mobile,
        "zone_ids": tuple(range(I_num)),
        "hub_ids": tuple(range(J_num)),
        "date_names": tuple(f"type{k}" for k in range(len(samples))),
        "I_num": I_num,
        "J_num": J_num,
        "S_num": len(samples),
        "samples": samples,
        "q_l": list(q_l),
        "epsilon_l": list(epsilon_l),
        "mu": mu,
        "Sigma_inv": np.linalg.inv(sigma),
        "r": float(r),
        "f": list(f),
        "c_m": float(c_m),
        "c_f": float(c_f),
        "rho": float(rho),
        "b": float(b),
        "tau": float(tau),
        "kappa": float(kappa),
        "M_m": list(M_m),
        "n_bar": list(n_bar),
        "M_z": np.full((I_num, J_num), max(M_m), dtype=float),
        "t_ij": 2.0 * float(gamma) * np.asarray(distance, dtype=float),
        "h_val": {},
    }


def build_instances():
    """Hand-written toys. Several are tuned so the optimum buys mobile units."""
    d2 = np.array([[0.0, 5.0], [5.0, 0.0]])
    d3 = np.array([[0.0, 6.0, 9.0], [6.0, 0.0, 5.0], [9.0, 5.0, 0.0]])

    instances = []

    # 1) Cheap mobile, shifting demand -> mobile should be used.
    instances.append(_base_two_zone(
        "mobile_shift_A",
        [[40.0, 5.0], [38.0, 6.0], [5.0, 42.0], [6.0, 40.0]],
        [0, 0, 1, 1],
        epsilon_l=[1.0, 1.0],
        q_l=[0.5, 0.5],
        f=[3.0, 3.0],
        c_m=2.0,
        c_f=12.0,
        rho=3.0,
        b=1.0,
        tau=8.0,
        kappa=10.0,
        M_m=[4, 4],
        n_bar=[6, 6],
        r=6.0,
        distance=d2,
        require_mobile=True,
    ))

    # 2) Same structure, stronger shortage penalty.
    instances.append(_base_two_zone(
        "mobile_shift_B",
        [[35.0, 4.0], [33.0, 5.0], [4.0, 36.0], [5.0, 34.0]],
        [0, 0, 1, 1],
        epsilon_l=[0.8, 1.1],
        q_l=[0.6, 0.4],
        f=[2.0, 4.0],
        c_m=1.5,
        c_f=15.0,
        rho=2.5,
        b=2.0,
        tau=7.0,
        kappa=9.0,
        M_m=[5, 5],
        n_bar=[5, 5],
        r=5.0,
        distance=d2,
        require_mobile=True,
    ))

    # 3) Three date types, mobile bridges peak zone changes.
    instances.append(_base_two_zone(
        "mobile_three_dates",
        [
            [30.0, 6.0], [28.0, 7.0],
            [8.0, 29.0], [7.0, 31.0],
            [18.0, 18.0], [17.0, 19.0],
        ],
        [0, 0, 1, 1, 2, 2],
        epsilon_l=[1.0, 1.0, 1.2],
        q_l=[1 / 3, 1 / 3, 1 / 3],
        f=[2.5, 2.5],
        c_m=1.0,
        c_f=14.0,
        rho=3.0,
        b=0.5,
        tau=6.0,
        kappa=8.0,
        M_m=[4, 4],
        n_bar=[6, 6],
        r=5.5,
        distance=d2,
        require_mobile=True,
    ))

    # 4) Fixed-only friendly: mobile expensive.
    instances.append(_base_two_zone(
        "fixed_preferred",
        [[20.0, 18.0], [19.0, 17.0], [18.0, 20.0], [17.0, 19.0]],
        [0, 0, 1, 1],
        epsilon_l=[0.5, 0.5],
        q_l=[0.5, 0.5],
        f=[20.0, 20.0],
        c_m=30.0,
        c_f=3.0,
        rho=2.0,
        b=0.0,
        tau=10.0,
        kappa=8.0,
        M_m=[3, 3],
        n_bar=[5, 5],
        r=4.0,
        distance=d2,
        require_mobile=False,
    ))

    # 5) b=0, still want mobile via cost gap.
    instances.append(_base_two_zone(
        "mobile_b0",
        [[45.0, 3.0], [44.0, 4.0], [3.0, 46.0], [4.0, 44.0]],
        [0, 0, 1, 1],
        epsilon_l=[1.0, 1.0],
        q_l=[0.55, 0.45],
        f=[1.0, 1.0],
        c_m=1.0,
        c_f=20.0,
        rho=4.0,
        b=0.0,
        tau=5.0,
        kappa=12.0,
        M_m=[4, 4],
        n_bar=[4, 4],
        r=6.0,
        distance=d2,
        require_mobile=True,
    ))

    # 6) Uneven hubs: only hub 0 is cheap to open.
    instances.append(_base_two_zone(
        "mobile_one_cheap_hub",
        [[32.0, 8.0], [30.0, 9.0], [9.0, 33.0], [8.0, 31.0]],
        [0, 0, 1, 1],
        epsilon_l=[1.2, 0.9],
        q_l=[0.5, 0.5],
        f=[1.0, 25.0],
        c_m=2.0,
        c_f=11.0,
        rho=3.0,
        b=1.5,
        tau=8.0,
        kappa=10.0,
        M_m=[5, 2],
        n_bar=[5, 5],
        r=5.0,
        distance=d2,
        require_mobile=True,
    ))

    # 7) Tiny epsilon, near-SAA.
    instances.append(_base_two_zone(
        "mobile_tiny_epsilon",
        [[26.0, 10.0], [25.0, 11.0], [10.0, 27.0], [11.0, 25.0]],
        [0, 0, 1, 1],
        epsilon_l=[0.1, 0.1],
        q_l=[0.5, 0.5],
        f=[2.0, 2.0],
        c_m=1.5,
        c_f=10.0,
        rho=2.5,
        b=1.0,
        tau=7.0,
        kappa=9.0,
        M_m=[3, 3],
        n_bar=[5, 5],
        r=4.5,
        distance=d2,
        require_mobile=True,
    ))

    # 8) Larger local boxes.
    instances.append(_base_two_zone(
        "mobile_wide_box",
        [[28.0, 7.0], [27.0, 8.0], [7.0, 29.0], [8.0, 27.0]],
        [0, 0, 1, 1],
        epsilon_l=[3.0, 3.5],
        q_l=[0.5, 0.5],
        f=[2.0, 2.0],
        c_m=1.8,
        c_f=12.0,
        rho=3.0,
        b=0.8,
        tau=6.0,
        kappa=10.0,
        M_m=[4, 4],
        n_bar=[6, 6],
        r=7.0,
        distance=d2,
        require_mobile=True,
    ))

    # 9) Built-in 3-zone toy from main.py (may or may not use mobile).
    toy = build_toy_data()
    toy["require_mobile"] = False
    instances.append(toy)

    # 10) 3-zone mobile-oriented synthetic.
    demand3 = np.array([
        [40.0, 5.0, 6.0],
        [38.0, 6.0, 5.0],
        [6.0, 42.0, 5.0],
        [5.0, 40.0, 7.0],
        [5.0, 6.0, 44.0],
        [7.0, 5.0, 41.0],
    ])
    instances.append(_base_two_zone(
        "mobile_3zone",
        demand3,
        [0, 0, 1, 1, 2, 2],
        epsilon_l=[1.0, 1.0, 1.0],
        q_l=[1 / 3, 1 / 3, 1 / 3],
        f=[2.0, 2.0, 2.0],
        c_m=1.2,
        c_f=16.0,
        rho=3.0,
        b=1.0,
        tau=6.0,
        kappa=10.0,
        M_m=[4, 4, 4],
        n_bar=[5, 5, 5],
        r=6.0,
        distance=d3,
        require_mobile=True,
    ))

    if len(instances) != 10:
        raise RuntimeError(f"expected 10 toys, got {len(instances)}")
    return instances


def with_all_patterns(model_data):
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
        eta[date_type, sample_index] = model.addVar(
            lb=-GRB.INFINITY, name=f"eta[{date_type},{sample_index}]"
        )
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
        -gp.quicksum(data["f"][j] * x[j] for j in hubs)
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
    mobile = int(round(sum(m[j].X for j in hubs)))
    return float(model.ObjVal), mobile


def compare_one(source):
    name = source["name"]
    require_mobile = bool(source.get("require_mobile"))
    full_data, active = with_all_patterns(source)
    dense_obj, dense_mobile = solve_dense_full(full_data, active)
    cg = run_cutting_plane_algorithm(deepcopy(source), TIGHT)
    cg_obj = None if cg["objective"] is None else float(cg["objective"])
    plan = cg.get("plan") or {}
    cg_mobile = 0 if not plan.get("m") else int(sum(plan["m"].values()))
    gap = None if cg_obj is None else abs(cg_obj - dense_obj)
    mobile_ok = (not require_mobile) or (cg_mobile > 0 and dense_mobile > 0)
    passed = (
        cg["status_name"] == "OPTIMAL"
        and cg["stop_reason"] == "没有新的割，主问题相对间隙已达到 Gurobi 接受的范围"
        and cg_obj is not None
        and gap is not None
        and gap < OBJ_TOL
        and mobile_ok
    )
    row = {
        "name": name,
        "zones": source["I_num"],
        "require_mobile": require_mobile,
        "dense_full": dense_obj,
        "dense_mobile": dense_mobile,
        "cutting_plane": cg_obj,
        "cg_mobile": cg_mobile,
        "abs_gap": gap,
        "iterations": cg["iterations"],
        "status": cg["status_name"],
        "stop_reason": cg["stop_reason"],
        "mobile_ok": mobile_ok,
        "passed": passed,
    }
    print(
        f"[{name}] dense={dense_obj:.10g} cg={cg_obj} |gap|={gap} "
        f"mobile dense/cg={dense_mobile}/{cg_mobile} "
        f"iters={cg['iterations']} PASS={passed}",
        flush=True,
    )
    return row


def main():
    rows = [compare_one(instance) for instance in build_instances()]
    mobile_cases = [row for row in rows if row["require_mobile"]]
    report = {
        "passed": all(row["passed"] for row in rows),
        "n_instances": len(rows),
        "n_require_mobile": len(mobile_cases),
        "n_require_mobile_passed": sum(row["passed"] for row in mobile_cases),
        "instances": rows,
    }
    destination = Path(__file__).resolve().parent / "outputs" / "toy10_compare.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"ALL_PASS={report['passed']} mobile_required_pass="
        f"{report['n_require_mobile_passed']}/{report['n_require_mobile']} "
        f"report={destination}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

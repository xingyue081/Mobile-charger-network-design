"""Compare the enumerated MISOCP with the official cutting-plane code.

The solver files are not edited. The MISOCP constraints are executed from
MISOCP.py. Synthetic 4-zone and 5-zone instances are formula checks only.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from main import build_toy_data, run_cutting_plane_algorithm
from oracle import compute_h_val

MISOCP_PATH = (
    Path(__file__).resolve().parents[1] / "reformulation_validation" / "MISOCP.py"
)
REPORT_DIR = Path(__file__).resolve().parent / "reports" / "precheck"
# Gurobi defaults: stop when the relative gap is at most 1e-4, or the absolute gap is at most 1e-10.
GUROBI_DEFAULT_MIP_GAP = 1e-4
GUROBI_DEFAULT_MIP_GAP_ABS = 1e-10
CG_LIMITS = {
    "master_mip_gap": GUROBI_DEFAULT_MIP_GAP,
    "master_mip_gap_abs": GUROBI_DEFAULT_MIP_GAP_ABS,
    "master_time_limit": 600.0,
    "max_iterations": 100,
    "max_total_seconds": 180.0,
    "cut_tolerance": 1e-4,
    "separation_mip_gap": None,
}


def within_tolerance(left, right):
    difference = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)))
    relative = 0.0 if scale == 0.0 else difference / scale
    accepted = relative <= GUROBI_DEFAULT_MIP_GAP or difference <= GUROBI_DEFAULT_MIP_GAP_ABS
    return accepted, difference


def misocp_source():
    text = MISOCP_PATH.read_text(encoding="utf-8")
    start = text.index("# 第二步：建立模型盒")
    finish = text.index("# 第五步：提取并打印结果")
    model_source = text[start:finish]
    model_source = model_source.replace(
        'model = gp.Model("Charge_Centrally_MISOCP")\n',
        'model = gp.Model("Charge_Centrally_MISOCP")\n'
        'model.setParam("OutputFlag", 0)\n'
        'model.setParam("MIPGap", 1e-4)\n'
        'model.setParam("MIPGapAbs", 1e-10)\n'
        'model.setParam("TimeLimit", 180)\n',
        1,
    )
    return text[:start], model_source


def solve_enumerated_misocp(data):
    import gurobipy as gp
    import itertools
    from gurobipy import GRB

    _, model_source = misocp_source()
    sigma = np.linalg.inv(np.asarray(data["Sigma_inv"], dtype=float))
    namespace = {
        "gp": gp,
        "GRB": GRB,
        "np": np,
        "I_num": data["I_num"],
        "J_num": data["J_num"],
        "S_num": data["S_num"],
        "samples": data["samples"],
        "q_l": list(data["q_l"]),
        "epsilon_l": list(data["epsilon_l"]),
        "r": float(data["r"]),
        "f": list(data["f"]),
        "c_m": data["c_m"],
        "c_f": data["c_f"],
        "rho": data["rho"],
        "b": data["b"],
        "tau": data["tau"],
        "kappa": data["kappa"],
        "M_m": list(data["M_m"]),
        "M_z": np.asarray(data["M_z"], dtype=float),
        "n_bar": list(data["n_bar"]),
        "t_ij": np.asarray(data["t_ij"], dtype=float),
        "mu": np.asarray(data["mu"], dtype=float),
        "S_mat": np.linalg.cholesky(sigma),
        "K_set": list(itertools.product((0, 1), repeat=data["I_num"])),
    }
    exec(model_source, namespace)
    model = namespace["model"]
    payload = {
        "status": int(model.Status),
        "runtime": float(model.Runtime),
        "objective": None if model.SolCount == 0 else float(model.ObjVal),
        "bound": None if model.SolCount == 0 else float(model.ObjBound),
        "mip_gap": None if model.SolCount == 0 else float(model.MIPGap),
        "plan": None,
    }
    if model.Status != GRB.OPTIMAL:
        return payload
    zones = range(data["I_num"])
    hubs = range(data["J_num"])
    dates = range(data["S_num"])
    plan = {
        "n": [int(round(namespace["n_fixed"][i].X)) for i in zones],
        "x": [int(round(namespace["x"][j].X)) for j in hubs],
        "m": [int(round(namespace["m"][j].X)) for j in hubs],
        "z": [
            [
                [int(round(namespace["z"][i, j, l].X)) for j in hubs]
                for i in zones
            ]
            for l in dates
        ],
    }
    payload["plan"] = plan
    return payload


def cg_plan(plan):
    if plan is None:
        return None
    zones = sorted({i for i, _ in plan["y"]})
    hubs = sorted(plan["x"])
    dates = sorted({entry[2] for entry in plan["z"]})
    return {
        "n": [int(plan["n"][i]) for i in zones],
        "x": [int(plan["x"][j]) for j in hubs],
        "m": [int(plan["m"][j]) for j in hubs],
        "z": [
            [
                [int(plan["z"][i, j, l]) for j in hubs]
                for i in zones
            ]
            for l in dates
        ],
    }


def compare_h(data):
    differences = []
    for (date_index, sample_index, pattern), stored in data["h_val"].items():
        sample = np.asarray(data["samples"][date_index][sample_index], dtype=float)
        epsilon = float(data["epsilon_l"][date_index])
        lower = np.maximum(0.0, sample - epsilon)
        upper = sample + epsilon
        direct = compute_h_val(
            pattern, lower, upper, data["mu"], data["Sigma_inv"],
            data["r"], data["rho"], data["b"], data["I_num"],
        )
        accepted, difference = within_tolerance(stored, direct)
        differences.append(difference)
        if not accepted:
            return {
                "passed": False,
                "count": len(differences),
                "max_abs_difference": max(differences),
                "first_failure": {
                    "date": date_index,
                    "sample": sample_index,
                    "pattern": list(pattern),
                    "stored": stored,
                    "compute_h_val": direct,
                    "abs_difference": difference,
                },
            }
    return {
        "passed": True,
        "count": len(differences),
        "max_abs_difference": max(differences) if differences else 0.0,
    }


def toy_file_matches_loader():
    data_source, _ = misocp_source()
    namespace = {"np": np, "itertools": __import__("itertools")}
    exec(data_source, namespace)
    loaded = build_toy_data()
    comparisons = {
        "mu": bool(np.allclose(namespace["mu"], loaded["mu"])),
        "epsilon_l": list(namespace["epsilon_l"]) == list(loaded["epsilon_l"]),
        "r": float(namespace["r"]) == float(loaded["r"]),
        "q_l": list(namespace["q_l"]) == list(loaded["q_l"]),
        "c_m": namespace["c_m"] == loaded["c_m"],
        "c_f": namespace["c_f"] == loaded["c_f"],
        "f": list(namespace["f"]) == list(loaded["f"]),
        "tau": namespace["tau"] == loaded["tau"],
        "kappa": namespace["kappa"] == loaded["kappa"],
        "t_ij": bool(np.allclose(namespace["t_ij"], loaded["t_ij"])),
        "Sigma": bool(np.allclose(namespace["Sigma"], np.linalg.inv(loaded["Sigma_inv"]))),
    }
    return {"passed": all(comparisons.values()), "items": comparisons}


def synthetic_instance(zone_count, seed):
    rng = np.random.default_rng(seed)
    counts = (4, 3, 2)
    samples = {}
    rows = []
    for date_index, count in enumerate(counts):
        demand = rng.uniform(30.0, 90.0, size=(count, zone_count))
        samples[date_index] = [row.copy() for row in demand]
        rows.append(demand)
    stacked = np.vstack(rows)
    mu = stacked.mean(axis=0)
    sigma = np.cov(stacked, rowvar=False) + np.eye(zone_count) * 1e-4
    sigma_inv = np.linalg.inv(sigma)
    centered = stacked - mu
    quadratic = np.einsum("ni,ij,nj->n", centered, sigma_inv, centered)
    radius = float(np.sqrt(np.max(quadratic)) * 1.05)
    total = sum(counts)
    distance = rng.uniform(1.0, 20.0, size=(zone_count, zone_count))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return {
        "name": f"synthetic_I{zone_count}",
        "zone_ids": tuple(range(zone_count)),
        "hub_ids": tuple(range(zone_count)),
        "date_names": ("date0", "date1", "date2"),
        "I_num": zone_count,
        "J_num": zone_count,
        "S_num": 3,
        "samples": samples,
        "q_l": [count / total for count in counts],
        "epsilon_l": [4.0, 5.0, 3.0],
        "mu": mu,
        "Sigma_inv": sigma_inv,
        "r": radius,
        "f": [15.0] * zone_count,
        "c_m": 6.0,
        "c_f": 8.0,
        "rho": 2.0,
        "b": 2.0,
        "tau": 20.0,
        "kappa": 15.0,
        "M_m": [3.0] * zone_count,
        "n_bar": [4.0] * zone_count,
        "M_z": np.full((zone_count, zone_count), 3.0),
        "t_ij": 2.0 * 0.01 * distance,
        "h_val": {},
        "synthetic": True,
        "seed": seed,
    }


def compare_instance(name, data):
    print(f"start {name}", file=sys.stderr, flush=True)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        cg_result = run_cutting_plane_algorithm(data, CG_LIMITS)
    misocp_result = solve_enumerated_misocp(data)
    objective_ok = False
    objective_difference = None
    if cg_result["objective"] is not None and misocp_result["objective"] is not None:
        objective_ok, objective_difference = within_tolerance(
            cg_result["objective"], misocp_result["objective"]
        )
    cg_deployment = cg_plan(cg_result["plan"])
    deployment_equal = cg_deployment == misocp_result["plan"]
    h_report = compare_h(data)
    converged = cg_result["stop_reason"] == "没有新的割，且主问题已证明最优"
    passed = (
        converged
        and misocp_result["status"] == 2
        and objective_ok
        and h_report["passed"]
    )
    return {
        "name": name,
        "synthetic": bool(data.get("synthetic", False)),
        "zones": data["I_num"],
        "passed": passed,
        "cg_stop_reason": cg_result["stop_reason"],
        "cg_iterations": cg_result["iterations"],
        "cg_objective": cg_result["objective"],
        "misocp_status": misocp_result["status"],
        "misocp_runtime": misocp_result.get("runtime"),
        "misocp_objective": misocp_result["objective"],
        "misocp_bound": misocp_result.get("bound"),
        "misocp_mip_gap": misocp_result.get("mip_gap"),
        "objective_abs_difference": objective_difference,
        "objective_within_tolerance": objective_ok,
        "deployment_equal": deployment_equal,
        "cg_plan": cg_deployment,
        "misocp_plan": misocp_result["plan"],
        "h_check": h_report,
        "log_tail": "\n".join(log.getvalue().splitlines()[-8:]),
    }


def main():
    toy_match = toy_file_matches_loader()
    print(f"toy_source_match={toy_match['passed']}", file=sys.stderr, flush=True)
    instances = [
        ("toy_I3", build_toy_data()),
        ("synthetic_I4", synthetic_instance(4, 20261024)),
        ("synthetic_I5", synthetic_instance(5, 20261025)),
    ]
    results = []
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / "small_equivalence.json"
    for name, data in instances:
        results.append(compare_instance(name, data))
        print(
            f"done {name} passed={results[-1]['passed']} "
            f"cg={results[-1]['cg_objective']} misocp={results[-1]['misocp_objective']}",
            file=sys.stderr,
            flush=True,
        )
        destination.write_text(json.dumps({"instances": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "passed": toy_match["passed"] and all(item["passed"] for item in results),
        "tolerance": {
            "mip_gap": GUROBI_DEFAULT_MIP_GAP,
            "mip_gap_abs": GUROBI_DEFAULT_MIP_GAP_ABS,
            "rule": "Gurobi default: relative gap <= 1e-4 or absolute gap <= 1e-10",
        },
        "misocp_time_limit_seconds": 180,
        "toy_source_matches_build_toy_data": toy_match,
        "instances": results,
    }
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"passed={report['passed']} report={destination}")
    print(f"  toy_source_match={toy_match['passed']}")
    for item in results:
        print(
            f"  {item['passed']} {item['name']} "
            f"cg={item['cg_objective']} misocp={item['misocp_objective']} "
            f"deployment_equal={item['deployment_equal']} "
            f"h_max_abs={item['h_check'].get('max_abs_difference')}"
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""比较 model:finite-milp 的全枚举最优值与当前割平面。

4 区实例没有现成数据，这里按 toy 的同一字段另造一个合成例子。
比较时主问题 gap 和割容差都收到 0，否则主问题默认相对间隙 1e-4、割容差 1e-4 本身就会超过 1e-7。
"""
from copy import deepcopy
from itertools import product

import numpy as np

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
}


def build_zone4_data():
    """合成 4 区、2 类日期、每类 2 个样本。不是真实 TAZ 数据。"""
    demand_data = np.array([
        [30.0, 12.0, 18.0, 9.0],
        [28.0, 14.0, 16.0, 11.0],
        [10.0, 32.0, 14.0, 20.0],
        [12.0, 30.0, 12.0, 22.0],
    ])
    date_types = [0, 0, 1, 1]
    mu = np.mean(demand_data, axis=0)
    # 固定对角协方差，避免 4 个样本把样本协方差弄成病态、导致支持函数 status=13。
    sigma = np.diag([25.0, 25.0, 25.0, 25.0])
    samples = {0: [], 1: []}
    for row, date_type in zip(demand_data, date_types):
        samples[date_type].append(row)
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
        "date_names": ("workday", "weekend", "holiday"),
        "I_num": 4, "J_num": 4, "S_num": 2,
        "samples": samples,
        "q_l": [0.5, 0.5],
        "epsilon_l": [1.0, 1.5],
        "mu": mu,
        "Sigma_inv": np.linalg.inv(sigma),
        "r": 4.0,
        "f": [30.0, 30.0, 30.0, 30.0],
        "c_m": 41.0, "c_f": 40.0, "rho": 2.0, "b": 2.0,
        "tau": 20.0, "kappa": 15.0,
        "M_m": [5, 5, 5, 5],
        "n_bar": [8, 8, 8, 8],
        "M_z": np.full((4, 4), 5.0),
        "t_ij": 2 * 0.01 * distance,
        "h_val": {},
    }


def solve_finite_milp(model_data):
    """式 model:finite-milp：每个样本放入全部 2^|I| 个 pattern。"""
    data = deepcopy(model_data)
    data["h_val"] = {}
    num_zones = data["I_num"]
    patterns = list(product((0, 1), repeat=num_zones))
    active = {}
    for date_type in range(data["S_num"]):
        epsilon = data["epsilon_l"][date_type]
        for sample_index, demand in enumerate(data["samples"][date_type]):
            lower = np.maximum(0, demand - epsilon)
            upper = demand + epsilon
            active[date_type, sample_index] = patterns
            for pattern in patterns:
                data["h_val"][date_type, sample_index, pattern] = compute_h_val(
                    pattern, lower, upper, data["mu"], data["Sigma_inv"],
                    data["r"], data["rho"], data["b"], num_zones,
                )
    solved, _, _, _, decisions = solve_master_problem(active, data, {
        "mip_gap": 0.0,
        "mip_gap_abs": 0.0,
        "optimality_tol": 1e-9,
        "time_limit": 600.0,
    })
    if not solved or decisions.get("status_name") != "OPTIMAL":
        raise RuntimeError(f"全枚举未证明最优: {decisions.get('status_name')}")
    return decisions["obj"]


def compare(name, model_data):
    full_objective = solve_finite_milp(model_data)
    cg = run_cutting_plane_algorithm(deepcopy(model_data), TIGHT)
    gap = None if cg["objective"] is None else abs(cg["objective"] - full_objective)
    iteration_limit = 2 * model_data["I_num"]
    passed = (
        cg["status_name"] == "OPTIMAL"
        and cg["stop_reason"] == "没有新的割，且主问题已证明最优"
        and gap is not None
        and gap < 1e-7
        and cg["iterations"] <= iteration_limit
    )
    print(
        f"\n[{name}] Pi*={full_objective:.12g}  Pi_CG={cg['objective']}  "
        f"|diff|={gap}  iterations={cg['iterations']}  "
        f"limit={iteration_limit}  status={cg['status_name']}  "
        f"stop={cg['stop_reason']}  PASS={passed}"
    )
    return passed


if __name__ == "__main__":
    toy_ok = compare("toy", build_toy_data())
    zone4_ok = compare("synthetic_4zone", build_zone4_data())
    print("ALL_PASS" if toy_ok and zone4_ok else "NOT_PASS")

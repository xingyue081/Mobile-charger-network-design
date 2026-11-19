"""2 区小例子：暴力枚举整数方案，核对全量 pattern 的 Gurobi 主问题。"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "charge_centrally_solver_1030"))

from master import solve_master_problem
from oracle import compute_h_val

from settings import BUSINESS, OUTPUT_ROOT


def toy_data() -> dict:
    """整数上界都是 1，两个日期类型各一个样本，方便枚举全部可行部署。"""
    samples = {
        0: [np.array([100.0, 80.0])],
        1: [np.array([40.0, 150.0])],
    }
    rows = np.vstack([samples[0][0], samples[1][0], np.array([90.0, 110.0])])
    mu = rows.mean(axis=0)
    sigma = np.cov(rows, rowvar=False, ddof=1)
    distance = np.array([[0.0, 10.0], [10.0, 0.0]])
    return {
        "name": "toy2_brute",
        "zone_ids": (0, 1),
        "hub_ids": (0, 1),
        "date_names": ("workday", "weekend"),
        "I_num": 2,
        "J_num": 2,
        "S_num": 2,
        "samples": samples,
        "q_l": [0.7, 0.3],
        "epsilon_l": [10.0, 10.0],
        "mu": mu,
        "Sigma_inv": np.linalg.inv(sigma),
        "r": 2.0,
        "f": [BUSINESS["hub_open_cost"], BUSINESS["hub_open_cost"]],
        "c_m": BUSINESS["mobile_unit_cost"],
        "c_f": BUSINESS["fixed_unit_cost"],
        "rho": BUSINESS["rho"],
        "b": BUSINESS["shortage_penalty"],
        "tau": BUSINESS["fixed_capacity"],
        "kappa": BUSINESS["mobile_capacity"],
        "M_m": [1, 1],
        "n_bar": [1, 1],
        "M_z": np.ones((2, 2)),
        "t_ij": 2.0 * BUSINESS["transport_cost_per_distance"] * distance,
        "h_val": {},
    }


def all_patterns(zone_count: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=zone_count))


def fill_heights(data: dict, patterns: list[tuple[int, ...]]) -> dict:
    active = {}
    for date_type in range(data["S_num"]):
        epsilon = data["epsilon_l"][date_type]
        for sample_index, demand in enumerate(data["samples"][date_type]):
            lower = np.maximum(0.0, np.asarray(demand, dtype=float) - epsilon)
            upper = np.asarray(demand, dtype=float) + epsilon
            active[date_type, sample_index] = patterns
            for pattern in patterns:
                data["h_val"][date_type, sample_index, pattern] = compute_h_val(
                    pattern, lower, upper, data["mu"], data["Sigma_inv"],
                    data["r"], data["rho"], data["b"], data["I_num"],
                )
    return active


def _bounded_sums(remaining: int, caps: tuple[int, ...]):
    if not caps:
        yield ()
        return
    for value in range(min(caps[0], remaining) + 1):
        for rest in _bounded_sums(remaining - value, caps[1:]):
            yield (value,) + rest


def _hub_dispatch(data: dict, assignment: tuple[int | None, ...], fleet: tuple[int, ...]):
    """一个日期类型上，各枢纽把车派到已归属区域。"""
    per_hub = []
    for hub, count in enumerate(fleet):
        caps = tuple(
            int(data["M_z"][zone, hub]) if assignment[zone] == hub else 0
            for zone in range(data["I_num"])
        )
        per_hub.append(list(_bounded_sums(count, caps)))
    for combo in product(*per_hub):
        yield combo


def enumerate_plans(data: dict):
    zones = range(data["I_num"])
    hubs = range(data["J_num"])
    fixed_ranges = [range(int(data["n_bar"][zone]) + 1) for zone in zones]
    for opened in product((0, 1), repeat=data["J_num"]):
        fleet_ranges = [
            range(int(data["M_m"][hub]) + 1) if opened[hub] else (0,)
            for hub in hubs
        ]
        choices = []
        for _zone in zones:
            choices.append((None,) + tuple(hub for hub in hubs if opened[hub]))
        for assignment in product(*choices):
            for fleet in product(*fleet_ranges):
                for fixed in product(*fixed_ranges):
                    date_options = [
                        list(_hub_dispatch(data, assignment, fleet))
                        for _date in range(data["S_num"])
                    ]
                    for dated in product(*date_options):
                        yield opened, assignment, fleet, fixed, dated


def score_plan(data: dict, patterns: list[tuple[int, ...]], plan: tuple) -> float:
    opened, _assignment, fleet, fixed, dated = plan
    investment = (
        sum(data["f"][hub] * opened[hub] for hub in range(data["J_num"]))
        + data["c_m"] * sum(fleet)
        + data["c_f"] * sum(fixed)
    )
    operating = 0.0
    for date_type in range(data["S_num"]):
        dispatch = dated[date_type]
        capacity = []
        for zone in range(data["I_num"]):
            incoming = sum(dispatch[hub][zone] for hub in range(data["J_num"]))
            capacity.append(data["tau"] * fixed[zone] + data["kappa"] * incoming)
        transport = sum(
            data["t_ij"][zone, hub] * dispatch[hub][zone]
            for hub in range(data["J_num"])
            for zone in range(data["I_num"])
        )
        weight = data["q_l"][date_type] / len(data["samples"][date_type])
        for sample_index in range(len(data["samples"][date_type])):
            sample_worst = min(
                sum((data["rho"] + data["b"]) * pattern[zone] * capacity[zone] for zone in range(data["I_num"]))
                - data["h_val"][date_type, sample_index, pattern]
                - transport
                for pattern in patterns
            )
            operating += weight * sample_worst
    return -investment + operating


def score_decision(data: dict, patterns: list[tuple[int, ...]], decisions: dict) -> float:
    opened = tuple(int(decisions["x"][hub]) for hub in range(data["J_num"]))
    assignment = []
    for zone in range(data["I_num"]):
        chosen = [hub for hub in range(data["J_num"]) if int(decisions["y"][zone, hub]) == 1]
        assignment.append(None if not chosen else chosen[0])
    fleet = tuple(int(decisions["m"][hub]) for hub in range(data["J_num"]))
    fixed = tuple(int(decisions["n"][zone]) for zone in range(data["I_num"]))
    dated = []
    for date_type in range(data["S_num"]):
        dated.append(tuple(
            tuple(int(decisions["z"][zone, hub, date_type]) for zone in range(data["I_num"]))
            for hub in range(data["J_num"])
        ))
    return score_plan(data, patterns, (opened, tuple(assignment), fleet, fixed, tuple(dated)))


def main() -> None:
    data = toy_data()
    patterns = all_patterns(data["I_num"])
    active = fill_heights(data, patterns)
    best = None
    count = 0
    for plan in enumerate_plans(data):
        count += 1
        value = score_plan(data, patterns, plan)
        if best is None or value > best:
            best = value
    solved, _, _, _, decisions = solve_master_problem(
        active, data, {"mip_gap": 0.0, "mip_gap_abs": 0.0, "time_limit": 120.0},
    )
    if not solved or decisions.get("status_name") != "OPTIMAL":
        raise SystemExit(f"全量主问题未证明最优: {decisions.get('status_name')}")
    rescored = score_decision(data, patterns, decisions)
    gap = abs(float(decisions["obj"]) - best)
    rescore_gap = abs(rescored - best)
    passed = gap < 1e-5 and rescore_gap < 1e-5
    report = {
        "passed": bool(passed),
        "enumerated_plans": int(count),
        "brute_objective": float(best),
        "gurobi_objective": float(decisions["obj"]),
        "gurobi_rescored": float(rescored),
        "gurobi_minus_brute": float(decisions["obj"]) - float(best),
        "rescore_minus_brute": float(rescored) - float(best),
        "status_name": decisions["status_name"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / "toy_brute_check.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

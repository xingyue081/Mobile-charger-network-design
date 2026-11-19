"""Separation oracle for charge-centrally cutting planes.

Accelerations kept exact:
- (b) Fix a_i when capacity lies outside the local box.
- (a) Alternating ascent finds a violated pattern cheaply; MIQCP certifies
  only when the heuristic does not already prove a violation.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class SeparationBound(Exception):
    """Minimization stopped with a finite lower bound, not a proven optimum."""

    def __init__(self, profit_lower: float, pattern=None):
        self.profit_lower = float(profit_lower)
        self.pattern = pattern
        super().__init__(
            f"分离只有下界 {self.profit_lower}，没有证明最优"
        )


def _limit_threads(model, threads: int | None = None) -> None:
    if threads is not None:
        model.setParam("Threads", int(threads))
        return
    env_threads = os.environ.get("CG_THREADS")
    if env_threads:
        model.setParam("Threads", int(env_threads))


def classify_pattern_bits(C_star, L, U, tol: float = 1e-9):
    """Fix pattern bits that the local box forces.

    If C_i <= L_i then every feasible D has D_i >= C_i, so a_i = 1.
    If C_i >= U_i then every feasible D has D_i <= C_i, so a_i = 0.
    Remaining zones stay free. This is the only part of Claude's (c) that
    we keep: the three-way zone state collapses to these fixes.
    """
    fixed = {}
    free = []
    for i, (capacity, lower, upper) in enumerate(zip(C_star, L, U)):
        if capacity <= lower + tol:
            fixed[i] = 1
        elif capacity >= upper - tol:
            fixed[i] = 0
        else:
            free.append(i)
    return fixed, free


def _apply_fixed_bits(a_list, fixed):
    for i, value in fixed.items():
        a_list[i] = int(value)
    return a_list


def compute_h_val(
    a_vec, L, U, mu, Sigma_inv, r, rho, b, I_num,
    threads: int | None = None,
    return_demand: bool = False,
):
    """h(a) = max_D {(rho + b) D·a - rho D} over the local ellipsoid∩box."""
    model = gp.Model("Compute_h")
    _limit_threads(model, threads)
    model.setParam("OutputFlag", 0)

    D = model.addVars(I_num, lb=L, ub=U, vtype=GRB.CONTINUOUS, name="D")
    diff = {i: D[i] - mu[i] for i in range(I_num)}
    quad_expr = gp.quicksum(
        Sigma_inv[i][j] * diff[i] * diff[j]
        for i in range(I_num) for j in range(I_num)
    )
    model.addQConstr(quad_expr <= r**2, name="Ellipsoid")
    obj = gp.quicksum((rho + b) * D[i] * a_vec[i] - rho * D[i] for i in range(I_num))
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        model.setParam("NumericFocus", 2)
        model.setParam("BarHomogeneous", 1)
        model.setParam("TimeLimit", 30)
        model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"compute_h_val 求解失败：status={model.Status}, "
            f"runtime={model.Runtime}, sol_count={model.SolCount}, pattern={a_vec}"
        )
    if return_demand:
        demand = tuple(float(D[i].X) for i in range(I_num))
        return model.ObjVal, demand
    return model.ObjVal


def _profit_from_h(a_vec, C_star, h_val, rho, b):
    return sum((rho + b) * C_star[i] * a_vec[i] for i in range(len(C_star))) - h_val


def _pattern_from_demand(demand, C_star, fixed):
    a = [1 if demand[i] > C_star[i] else 0 for i in range(len(C_star))]
    return tuple(_apply_fixed_bits(a, fixed))


def _seed_patterns(C_star, L, U, fixed, free, sample_demand=None):
    I = len(C_star)
    seeds = []

    def close(bits_for_free):
        a = [0] * I
        for i, bit in zip(free, bits_for_free):
            a[i] = int(bit)
        return tuple(_apply_fixed_bits(a, fixed))

    seeds.append(tuple(_apply_fixed_bits([0] * I, fixed)))
    seeds.append(tuple(_apply_fixed_bits([1] * I, fixed)))
    if free:
        seeds.append(close([0] * len(free)))
        seeds.append(close([1] * len(free)))
        mid = 0.5 * (np.asarray(L, dtype=float) + np.asarray(U, dtype=float))
        seeds.append(_pattern_from_demand(mid, C_star, fixed))
        if sample_demand is not None:
            seeds.append(_pattern_from_demand(sample_demand, C_star, fixed))
        # A few deterministic flips of the midpoint pattern on free zones.
        base = list(_pattern_from_demand(mid, C_star, fixed))
        for offset in range(min(3, len(free))):
            trial = base[:]
            idx = free[offset]
            trial[idx] = 1 - trial[idx]
            seeds.append(tuple(_apply_fixed_bits(trial, fixed)))
    # Deduplicate while keeping order.
    seen = set()
    ordered = []
    for pattern in seeds:
        if pattern not in seen:
            seen.add(pattern)
            ordered.append(pattern)
    return ordered


def alternating_worst_pattern(
    C_star,
    L,
    U,
    mu,
    Sigma_inv,
    r,
    rho,
    b,
    sample_demand=None,
    max_passes: int = 20,
    threads: int | None = None,
):
    """Heuristic local search over patterns.

    Fix a, solve the SOCP for worst D; fix D, set a_i = 1{D_i > C_i}.
    Returns the best pattern found and its exact pattern-wise profit.
    This value is an upper bound on the true worst profit.
    """
    fixed, free = classify_pattern_bits(C_star, L, U)
    I = len(C_star)
    # If the box already fixes every bit, one SOCP finishes the job exactly.
    if not free:
        pattern = tuple(_apply_fixed_bits([0] * I, fixed))
        h_val, demand = compute_h_val(
            pattern, L, U, mu, Sigma_inv, r, rho, b, I,
            threads=threads, return_demand=True,
        )
        profit = _profit_from_h(pattern, C_star, h_val, rho, b)
        return pattern, profit, demand, True

    best_pattern = None
    best_profit = None
    best_demand = None
    for seed in _seed_patterns(C_star, L, U, fixed, free, sample_demand):
        pattern = seed
        demand = None
        for _ in range(max_passes):
            h_val, demand = compute_h_val(
                pattern, L, U, mu, Sigma_inv, r, rho, b, I,
                threads=threads, return_demand=True,
            )
            profit = _profit_from_h(pattern, C_star, h_val, rho, b)
            if best_profit is None or profit < best_profit - 1e-12:
                best_profit = profit
                best_pattern = pattern
                best_demand = demand
            nxt = _pattern_from_demand(demand, C_star, fixed)
            if nxt == pattern:
                break
            pattern = nxt
    return best_pattern, best_profit, best_demand, False


def find_worst_pattern_miqcp(
    C_star,
    L,
    U,
    mu,
    Sigma_inv,
    r,
    rho,
    b,
    return_demand=False,
    mip_gap=None,
    time_limit=None,
    threads: int | None = None,
):
    """Exact MIQCP separation with forced pattern bits from the local box."""
    I = len(C_star)
    fixed, free = classify_pattern_bits(C_star, L, U)
    model = gp.Model("Separation_Oracle")
    _limit_threads(model, threads)
    model.setParam("OutputFlag", 0)
    if mip_gap is not None:
        model.setParam("MIPGap", float(mip_gap))
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))

    D = model.addVars(I, lb=L, ub=U, vtype=GRB.CONTINUOUS, name="D")
    a = {}
    for i in range(I):
        if i in fixed:
            a[i] = model.addVar(lb=fixed[i], ub=fixed[i], vtype=GRB.CONTINUOUS, name=f"a[{i}]")
        else:
            a[i] = model.addVar(vtype=GRB.BINARY, name=f"a[{i}]")
    v = model.addVars(I, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")

    obj = gp.quicksum(
        rho * D[i] - (rho + b) * v[i] + (rho + b) * C_star[i] * a[i]
        for i in range(I)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    for i in range(I):
        model.addConstr(v[i] >= L[i] * a[i])
        model.addConstr(v[i] <= U[i] * a[i])
        model.addConstr(v[i] >= D[i] - U[i] * (1 - a[i]))
        model.addConstr(v[i] <= D[i] - L[i] * (1 - a[i]))

    diff = {i: D[i] - mu[i] for i in range(I)}
    quad_expr = gp.quicksum(
        Sigma_inv[i][j] * diff[i] * diff[j]
        for i in range(I) for j in range(I)
    )
    model.addQConstr(quad_expr <= r**2, name="Ellipsoid")
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        a_star = tuple(int(round(a[i].X)) for i in range(I))
        worst_profit = model.ObjVal
        if return_demand:
            demand = tuple(float(D[i].X) for i in range(I))
            return a_star, worst_profit, demand
        return a_star, worst_profit

    bound = getattr(model, "ObjBound", None)
    pattern = None
    if model.SolCount:
        pattern = tuple(int(round(a[i].X)) for i in range(I))
    if bound is not None and math.isfinite(float(bound)):
        raise SeparationBound(float(bound), pattern)

    gap = getattr(model, "MIPGap", None)
    raise RuntimeError(
        f"子问题未达到设定精度：status={model.Status}, runtime={model.Runtime}, "
        f"gap={gap}, sol_count={model.SolCount}, free_bits={len(free)}"
    )


def find_worst_pattern(
    C_star,
    L,
    U,
    mu,
    Sigma_inv,
    r,
    rho,
    b,
    return_demand=False,
    mip_gap=None,
    time_limit=None,
    sample_demand=None,
    eta=None,
    transport=0.0,
    cut_tolerance=0.0,
    force_exact=False,
    threads: int | None = None,
):
    """Separate one sample.

    If eta is given and the heuristic already finds a violated pattern, return
    that pattern without MIQCP. Otherwise run the exact MIQCP (with fixed bits).
    """
    heuristic_pattern = None
    heuristic_profit = None
    heuristic_demand = None
    exact_already = False
    try:
        heuristic_pattern, heuristic_profit, heuristic_demand, exact_already = alternating_worst_pattern(
            C_star, L, U, mu, Sigma_inv, r, rho, b,
            sample_demand=sample_demand,
            threads=threads,
        )
    except RuntimeError:
        # A failed convex subproblem does not block the exact MIQCP.
        pass
    used_miqcp = False
    exact = False
    profit_lower = None
    cut_pattern = None
    cut_profit = None
    if exact_already:
        a_star, worst_profit, demand = heuristic_pattern, heuristic_profit, heuristic_demand
        exact = True
        profit_lower = worst_profit
        cut_pattern, cut_profit = a_star, worst_profit
    else:
        heuristic_violates = (
            eta is not None
            and heuristic_profit is not None
            and eta > heuristic_profit - transport + cut_tolerance
        )
        if heuristic_violates and not force_exact:
            a_star, worst_profit, demand = heuristic_pattern, heuristic_profit, heuristic_demand
            profit_lower = None
            cut_pattern, cut_profit = a_star, worst_profit
        else:
            used_miqcp = True
            try:
                a_star, worst_profit, demand = find_worst_pattern_miqcp(
                    C_star, L, U, mu, Sigma_inv, r, rho, b,
                    return_demand=True,
                    mip_gap=mip_gap,
                    time_limit=time_limit,
                    threads=threads,
                )
            except SeparationBound as exc:
                a_star, worst_profit, demand = exc.pattern, exc.profit_lower, None
                profit_lower = exc.profit_lower
                if heuristic_pattern is not None and heuristic_profit is not None:
                    cut_pattern, cut_profit = heuristic_pattern, heuristic_profit
            else:
                exact = True
                profit_lower = worst_profit
                cut_pattern, cut_profit = a_star, worst_profit
    info = {
        "used_miqcp": used_miqcp,
        "exact": exact,
        "exact_by_fixing": exact_already,
        "profit_lower": profit_lower,
        "cut_pattern": cut_pattern,
        "cut_profit": cut_profit,
        "heuristic_pattern": heuristic_pattern,
        "heuristic_profit": heuristic_profit,
    }
    if return_demand:
        return a_star, worst_profit, demand, info
    return a_star, worst_profit, info


def separate_samples_parallel(jobs, workers: int = 1):
    """Run independent sample separations. workers=1 keeps sequential order."""
    if workers is None or workers <= 1 or len(jobs) <= 1:
        results = []
        for job in jobs:
            results.append((job["key"], _run_one_job(job)))
        return results

    worker_count = min(int(workers), len(jobs))
    # Avoid oversubscription: one Gurobi thread per sample worker.
    prepared = []
    for job in jobs:
        item = dict(job)
        item["threads"] = 1
        prepared.append(item)

    results = [None] * len(prepared)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_run_one_job, job): idx
            for idx, job in enumerate(prepared)
        }
        for future in as_completed(futures):
            idx = futures[future]
            key = prepared[idx]["key"]
            results[idx] = (key, future.result())
    return results


def _run_one_job(job):
    return find_worst_pattern(
        C_star=job["C_star"],
        L=job["L"],
        U=job["U"],
        mu=job["mu"],
        Sigma_inv=job["Sigma_inv"],
        r=job["r"],
        rho=job["rho"],
        b=job["b"],
        return_demand=False,
        mip_gap=job.get("mip_gap"),
        time_limit=job.get("time_limit"),
        sample_demand=job.get("sample_demand"),
        eta=job.get("eta"),
        transport=job.get("transport", 0.0),
        cut_tolerance=job.get("cut_tolerance", 0.0),
        force_exact=job.get("force_exact", False),
        threads=job.get("threads"),
    )

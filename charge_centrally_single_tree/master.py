"""Single-tree branch-and-cut master for charge-centrally.

One Gurobi MIP tree. When an integer feasible solution is found (MIPSOL),
the callback separates samples and submits violated pattern inequalities as
lazy constraints. Exact MIQCP certification is done outside the callback
after the tree stops, to avoid nested heavy MIQCPs inside the search.
"""

from __future__ import annotations

import os
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from oracle import alternating_worst_pattern, compute_h_val, find_worst_pattern


def _status_name(status):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    return names.get(status, str(status))


class SingleTreeMaster:
    """One MIP tree with lazy pattern cuts at integer solutions."""

    def __init__(self, toy_data, active_patterns, *, cut_tolerance: float = 1e-4, cut_cap: int | None = None):
        self.toy_data = toy_data
        self.active_patterns = active_patterns
        self.cut_tolerance = float(cut_tolerance)
        self.cut_cap = None if cut_cap is None else int(cut_cap)

        self.I_num = toy_data["I_num"]
        self.J_num = toy_data["J_num"]
        self.S_num = toy_data["S_num"]
        self.I = range(self.I_num)
        self.J = range(self.J_num)
        self.S = range(self.S_num)
        self.N_l_Omega = {l: len(toy_data["samples"][l]) for l in self.S}
        self.sample_keys = [
            (l, n_idx) for l in self.S for n_idx in range(self.N_l_Omega[l])
        ]

        self.model = gp.Model("Charge_Centrally_SingleTree")
        threads = os.environ.get("CG_THREADS")
        if threads:
            self.model.setParam("Threads", int(threads))
        self.model.setParam("OutputFlag", 0)
        # Required for cbLazy.
        self.model.setParam("LazyConstraints", 1)

        self.x = self.model.addVars(self.J, vtype=GRB.BINARY, name="x")
        self.y = self.model.addVars(self.I, self.J, vtype=GRB.BINARY, name="y")
        self.m = self.model.addVars(self.J, vtype=GRB.INTEGER, name="m")
        self.n = self.model.addVars(self.I, vtype=GRB.INTEGER, name="n")
        self.z = self.model.addVars(self.I, self.J, self.S, vtype=GRB.INTEGER, name="z")
        self.eta = self.model.addVars(
            [(l, n_idx) for l in self.S for n_idx in range(self.N_l_Omega[l])],
            lb=-GRB.INFINITY,
            name="eta",
        )
        self.transport = self.model.addVars(self.S, lb=-GRB.INFINITY, name="transport")
        self.inflow = self.model.addVars(self.I, self.S, lb=0.0, name="inflow")

        n_bar, M_z, M_m = toy_data["n_bar"], toy_data["M_z"], toy_data["M_m"]
        t_ij = toy_data["t_ij"]
        for j in self.J:
            for l in self.S:
                self.model.addConstr(gp.quicksum(self.z[i, j, l] for i in self.I) <= self.m[j])
            self.model.addConstr(self.m[j] <= M_m[j] * self.x[j])
        for i in self.I:
            self.model.addConstr(gp.quicksum(self.y[i, j] for j in self.J) <= 1)
            self.model.addConstr(self.n[i] <= n_bar[i])
            for j in self.J:
                self.model.addConstr(self.y[i, j] <= self.x[j])
                for l in self.S:
                    self.model.addConstr(self.z[i, j, l] <= M_z[i, j] * self.y[i, j])
        for l in self.S:
            self.model.addConstr(
                self.transport[l] == gp.quicksum(
                    t_ij[i, j] * self.z[i, j, l] for i in self.I for j in self.J
                ),
                name=f"transport_def[{l}]",
            )
            for i in self.I:
                self.model.addConstr(
                    self.inflow[i, l] == gp.quicksum(self.z[i, j, l] for j in self.J),
                    name=f"inflow_def[{i},{l}]",
                )

        for (l, n_idx), patterns in active_patterns.items():
            for a_vec in patterns:
                self._add_user_cut(l, n_idx, a_vec)

        f, c_m, c_f, q_l = toy_data["f"], toy_data["c_m"], toy_data["c_f"], toy_data["q_l"]
        obj = (
            -gp.quicksum(f[j] * self.x[j] for j in self.J)
            - c_m * gp.quicksum(self.m[j] for j in self.J)
            - c_f * gp.quicksum(self.n[i] for i in self.I)
            + gp.quicksum(
                (q_l[l] / self.N_l_Omega[l]) * self.eta[l, n_idx]
                for l in self.S
                for n_idx in range(self.N_l_Omega[l])
            )
        )
        self.model.setObjective(obj, GRB.MAXIMIZE)
        self.model.update()

        self.stats = {
            "callback_calls": 0,
            "lazy_cuts_added": 0,
            "callback_heuristic_cuts": 0,
            "callback_seconds": 0.0,
            "exact_restarts": 0,
            "exact_cuts_added": 0,
        }

    def _cut_rhs_expr(self, l, n_idx, a_vec):
        """Linear expression for the pattern cut RHS (without eta)."""
        toy_data = self.toy_data
        tau, kappa, rho, b = toy_data["tau"], toy_data["kappa"], toy_data["rho"], toy_data["b"]
        a_vec = tuple(a_vec)
        h_current = toy_data["h_val"][l, n_idx, a_vec]
        revenue_expr = gp.quicksum(
            (rho + b) * a_vec[i] * (tau * self.n[i] + kappa * self.inflow[i, l])
            for i in self.I
        )
        return -self.transport[l] + revenue_expr - h_current

    def _add_user_cut(self, l, n_idx, a_vec):
        """Static cut present from the start (or after an exact restart)."""
        self.model.addConstr(self.eta[l, n_idx] <= self._cut_rhs_expr(l, n_idx, a_vec))

    def register_pattern(self, l, n_idx, a_vec, h_val=None):
        """Store pattern constant and mark it active. Returns True if new.

        Prefer computing h by SOCP on the local box, same as the dense
        full-pattern baseline, instead of recovering it from an incumbent.
        """
        a_vec = tuple(a_vec)
        patterns = self.active_patterns[(l, n_idx)]
        if a_vec in patterns:
            return False
        if h_val is None:
            _demand, lower, upper = self._local_box(l, n_idx)
            toy = self.toy_data
            h_val = compute_h_val(
                a_vec, lower, upper, toy["mu"], toy["Sigma_inv"], toy["r"],
                toy["rho"], toy["b"], self.I_num, threads=1,
            )
        patterns.append(a_vec)
        self.toy_data["h_val"][(l, n_idx, a_vec)] = float(h_val)
        return True

    def _local_box(self, l, n_idx):
        eps = self.toy_data["epsilon_l"][l]
        demand = np.asarray(self.toy_data["samples"][l][n_idx], dtype=float)
        lower = np.maximum(0.0, demand - eps)
        upper = demand + eps
        return demand, lower, upper

    def _callback(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return
        started = perf_counter()
        self.stats["callback_calls"] += 1
        toy_data = self.toy_data
        tau, kappa = toy_data["tau"], toy_data["kappa"]
        rho, b = toy_data["rho"], toy_data["b"]
        mu, Sigma_inv, r = toy_data["mu"], toy_data["Sigma_inv"], toy_data["r"]

        n_val = {i: model.cbGetSolution(self.n[i]) for i in self.I}
        z_val = {
            (i, j, l): model.cbGetSolution(self.z[i, j, l])
            for i in self.I for j in self.J for l in self.S
        }
        eta_val = {
            (l, n_idx): model.cbGetSolution(self.eta[l, n_idx])
            for l in self.S for n_idx in range(self.N_l_Omega[l])
        }
        transport_val = {l: model.cbGetSolution(self.transport[l]) for l in self.S}

        candidates = []
        for l, n_idx in self.sample_keys:
            demand, lower, upper = self._local_box(l, n_idx)
            C_star = [
                tau * n_val[i] + kappa * sum(z_val[i, j, l] for j in self.J)
                for i in self.I
            ]
            try:
                pattern, profit, _demand, exact_already = alternating_worst_pattern(
                    C_star, lower, upper, mu, Sigma_inv, r, rho, b,
                    sample_demand=demand,
                    threads=1,
                )
            except Exception:
                # Callback must not kill the master tree on a transient subproblem error.
                continue
            violation = eta_val[l, n_idx] - (profit - transport_val[l])
            if violation <= self.cut_tolerance:
                continue
            if pattern in self.active_patterns[(l, n_idx)]:
                continue
            candidates.append((violation, l, n_idx, pattern, profit, C_star, exact_already))

        if not candidates:
            self.stats["callback_seconds"] += perf_counter() - started
            return

        candidates.sort(reverse=True, key=lambda row: row[0])
        limit = len(candidates) if self.cut_cap is None else min(self.cut_cap, len(candidates))
        for violation, l, n_idx, pattern, profit, C_star, _exact in candidates[:limit]:
            if not self.register_pattern(l, n_idx, pattern):
                continue
            lazy = self.eta[l, n_idx] <= self._cut_rhs_expr(l, n_idx, pattern)
            model.cbLazy(lazy)
            self.stats["lazy_cuts_added"] += 1
            self.stats["callback_heuristic_cuts"] += 1

        self.stats["callback_seconds"] += perf_counter() - started

    def optimize(self, solver_limits=None, start_solution=None, log_file=None):
        limits = {} if solver_limits is None else solver_limits
        if log_file is not None:
            self.model.setParam("OutputFlag", 1)
            self.model.setParam("LogToConsole", 0)
            self.model.setParam("LogFile", str(log_file))
            self.model.setParam("DisplayInterval", 30)
        else:
            self.model.setParam("OutputFlag", 1 if limits.get("output_flag") else 0)
            self.model.setParam("LogFile", "")

        if limits.get("mip_gap") is not None:
            self.model.setParam("MIPGap", float(limits["mip_gap"]))
        else:
            self.model.setParam("MIPGap", 1e-4)
        if limits.get("mip_gap_abs") is not None:
            self.model.setParam("MIPGapAbs", float(limits["mip_gap_abs"]))
        if limits.get("optimality_tol") is not None:
            self.model.setParam("OptimalityTol", float(limits["optimality_tol"]))
        if limits.get("time_limit") is not None:
            self.model.setParam("TimeLimit", float(limits["time_limit"]))
        else:
            self.model.setParam("TimeLimit", GRB.INFINITY)

        if start_solution is not None and "x" in start_solution:
            for j in self.J:
                self.x[j].Start = start_solution["x"][j]
                self.m[j].Start = start_solution["m"][j]
            for i in self.I:
                self.n[i].Start = start_solution["n"][i]
                for j in self.J:
                    self.y[i, j].Start = start_solution["y"][(i, j)]
                    for l in self.S:
                        self.z[i, j, l].Start = start_solution["z"][(i, j, l)]

        self.model.update()
        matrix_size = {
            "num_vars": int(self.model.NumVars),
            "num_constrs": int(self.model.NumConstrs),
            "num_nzs": int(self.model.NumNZs),
            "warm_start": start_solution is not None and "x" in start_solution,
        }
        first_incumbent = {"seconds": None, "obj": None}

        def _record_and_separate(cb_model, where):
            if where == GRB.Callback.MIPSOL and first_incumbent["seconds"] is None:
                first_incumbent["seconds"] = float(cb_model.cbGet(GRB.Callback.RUNTIME))
                first_incumbent["obj"] = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            self._callback(cb_model, where)

        self.model.optimize(_record_and_separate)
        runtime_total = float(self.model.Runtime)

        stopped_with_solution = self.model.Status in (
            GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED,
        )
        if self.model.SolCount == 0 or not stopped_with_solution:
            return False, {}, {}, {}, {
                "status": self.model.Status,
                "status_name": _status_name(self.model.Status),
                "runtime": runtime_total,
                "mip_gap": None,
                "obj_bound": None,
            }

        tau, kappa = self.toy_data["tau"], self.toy_data["kappa"]
        t_ij = self.toy_data["t_ij"]
        C_opt, eta_opt, T_opt = {}, {}, {}
        for l in self.S:
            T_opt[l] = sum(t_ij[i, j] * self.z[i, j, l].X for i in self.I for j in self.J)
            for i in self.I:
                C_opt[l, i] = tau * self.n[i].X + kappa * sum(
                    self.z[i, j, l].X for j in self.J
                )
            for n_idx in range(self.N_l_Omega[l]):
                eta_opt[l, n_idx] = self.eta[l, n_idx].X

        decisions = {
            "obj": self.model.ObjVal,
            "obj_bound": self.model.ObjBound,
            "mip_gap": self.model.MIPGap,
            "runtime": runtime_total,
            "time_limit": float(self.model.Params.TimeLimit),
            "node_count": int(self.model.NodeCount),
            "iter_count": int(self.model.IterCount),
            "status": self.model.Status,
            "status_name": _status_name(self.model.Status),
            "num_vars": matrix_size["num_vars"],
            "num_constrs": int(self.model.NumConstrs),
            "num_nzs": int(self.model.NumNZs),
            "warm_start": matrix_size["warm_start"],
            "first_incumbent_seconds": first_incumbent["seconds"],
            "first_incumbent_obj": first_incumbent["obj"],
            "lazy_cuts_added": self.stats["lazy_cuts_added"],
            "callback_calls": self.stats["callback_calls"],
            "callback_seconds": self.stats["callback_seconds"],
            "x": {j: round(self.x[j].X) for j in self.J},
            "y": {(i, j): round(self.y[i, j].X) for i in self.I for j in self.J},
            "m": {j: round(self.m[j].X) for j in self.J},
            "n": {i: round(self.n[i].X) for i in self.I},
            "z": {
                (i, j, l): round(self.z[i, j, l].X)
                for i in self.I for j in self.J for l in self.S
            },
        }
        return True, C_opt, eta_opt, T_opt, decisions

    def exact_certify(self, C_opt, eta_opt, T_opt, *, mip_gap=None, time_limit=None, cut_cap=None):
        """Exact separation after the tree stops. New cuts are added as user cuts.

        Returns (cuts_added, accounted_operating, details).
        """
        toy_data = self.toy_data
        rho, b = toy_data["rho"], toy_data["b"]
        mu, Sigma_inv, r = toy_data["mu"], toy_data["Sigma_inv"], toy_data["r"]
        cuts_added = 0
        accounted_operating = 0.0
        details = []
        limit = len(self.sample_keys) if cut_cap is None else int(cut_cap)
        pending = []

        for l, n_idx in self.sample_keys:
            demand, lower, upper = self._local_box(l, n_idx)
            C_star = [C_opt[l, i] for i in self.I]
            a_worst, profit, info = find_worst_pattern(
                C_star=C_star,
                L=lower,
                U=upper,
                mu=mu,
                Sigma_inv=Sigma_inv,
                r=r,
                rho=rho,
                b=b,
                mip_gap=mip_gap,
                time_limit=time_limit,
                sample_demand=demand,
                eta=eta_opt[l, n_idx],
                transport=T_opt[l],
                cut_tolerance=self.cut_tolerance,
                force_exact=True,
                threads=1,
            )
            sample_weight = toy_data["q_l"][l] / self.N_l_Omega[l]
            accounted_operating += sample_weight * (profit - T_opt[l])
            violation = eta_opt[l, n_idx] - (profit - T_opt[l])
            if violation > self.cut_tolerance and a_worst not in self.active_patterns[(l, n_idx)]:
                pending.append((violation, l, n_idx, a_worst, profit, C_star, info))

        pending.sort(reverse=True, key=lambda row: row[0])
        for violation, l, n_idx, a_worst, profit, C_star, info in pending[:limit]:
            if not self.register_pattern(l, n_idx, a_worst):
                continue
            self._add_user_cut(l, n_idx, a_worst)
            cuts_added += 1
            self.stats["exact_cuts_added"] += 1
            details.append({
                "sample": (l, n_idx),
                "pattern": a_worst,
                "violation": violation,
                "used_miqcp": info.get("used_miqcp"),
            })

        if cuts_added:
            self.stats["exact_restarts"] += 1
            self.model.update()
        return cuts_added, accounted_operating, details


# Keep thin aliases used by older scripts / compare utilities.
CentralMaster = SingleTreeMaster


def solve_master_problem(active_patterns, toy_data, solver_limits=None, start_solution=None, log_file=None):
    """One-shot solve for full-pattern equivalence scripts."""
    master = SingleTreeMaster(toy_data, active_patterns, cut_tolerance=0.0, cut_cap=None)
    return master.optimize(solver_limits, start_solution=start_solution, log_file=log_file)

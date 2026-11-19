"""Persistent master for charge-centrally cutting planes.

The Gurobi model is built once. Later rounds only add violated pattern cuts.
Transport and inflow are defined once; each cut references those lifts.
"""

from __future__ import annotations

import os

import gurobipy as gp
from gurobipy import GRB


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


class CentralMaster:
    """Keep one master MIP and append pattern cuts in place."""

    def __init__(self, toy_data, active_patterns):
        self.toy_data = toy_data
        self.active_patterns = active_patterns
        self.I_num = toy_data["I_num"]
        self.J_num = toy_data["J_num"]
        self.S_num = toy_data["S_num"]
        self.I = range(self.I_num)
        self.J = range(self.J_num)
        self.S = range(self.S_num)
        self.N_l_Omega = {l: len(toy_data["samples"][l]) for l in self.S}

        self.model = gp.Model("Charge_Centrally_Master")
        threads = os.environ.get("CG_THREADS")
        if threads:
            self.model.setParam("Threads", int(threads))
        self.model.setParam("OutputFlag", 0)

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
                self._add_cut(l, n_idx, a_vec)

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

    def _add_cut(self, l, n_idx, a_vec):
        toy_data = self.toy_data
        tau, kappa, rho, b = toy_data["tau"], toy_data["kappa"], toy_data["rho"], toy_data["b"]
        a_vec = tuple(a_vec)
        h_current = toy_data["h_val"][l, n_idx, a_vec]
        revenue_expr = gp.quicksum(
            (rho + b) * a_vec[i] * (tau * self.n[i] + kappa * self.inflow[i, l])
            for i in self.I
        )
        cut_expr = -self.transport[l] + revenue_expr - h_current
        self.model.addConstr(self.eta[l, n_idx] <= cut_expr)

    def add_pattern(self, l, n_idx, a_vec):
        """Append one pattern cut if it is new for this sample."""
        a_vec = tuple(a_vec)
        patterns = self.active_patterns[(l, n_idx)]
        if a_vec in patterns:
            return False
        patterns.append(a_vec)
        self._add_cut(l, n_idx, a_vec)
        return True

    def optimize(self, solver_limits=None, start_solution=None, log_file=None):
        """Solve the current restricted master once.

        By default the time limit is hard: TIME_LIMIT with a feasible solution
        is returned immediately. Set extend_on_timelimit=True only for a final
        certifying round if you really want the old double-until-OPTIMAL behavior.
        """
        limits = {} if solver_limits is None else solver_limits
        if log_file is not None:
            self.model.setParam("OutputFlag", 1)
            self.model.setParam("LogToConsole", 0)
            self.model.setParam("LogFile", str(log_file))
            self.model.setParam("DisplayInterval", 30)
        else:
            self.model.setParam("OutputFlag", 1 if limits.get("output_flag") else 0)
            self.model.setParam("LogFile", "")

        # Reset per-round limits. None means leave Gurobi default / unlimited.
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
            "num_qconstrs": int(self.model.NumQConstrs),
            "warm_start": start_solution is not None and "x" in start_solution,
        }
        first_incumbent = {"seconds": None, "obj": None}

        def _record_first_incumbent(cb_model, where):
            if where != GRB.Callback.MIPSOL or first_incumbent["seconds"] is not None:
                return
            first_incumbent["seconds"] = float(cb_model.cbGet(GRB.Callback.RUNTIME))
            first_incumbent["obj"] = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))

        runtime_total = 0.0
        extend_on_timelimit = bool(limits.get("extend_on_timelimit"))
        max_extends = int(limits.get("max_timelimit_extends") or 0)
        extends = 0
        while True:
            self.model.optimize(_record_first_incumbent)
            runtime_total += float(self.model.Runtime)
            if (
                self.model.Status != GRB.TIME_LIMIT
                or self.model.SolCount == 0
                or not extend_on_timelimit
                or extends >= max_extends
            ):
                break
            extends += 1
            extended = float(self.model.Params.TimeLimit) * 2.0
            print(
                f"主问题在时限内未证明最优，时限加长到 {extended:.0f} 秒，同一棵分支树接着算"
                f"（认证加长 {extends}/{max_extends}）。",
                flush=True,
            )
            self.model.Params.TimeLimit = extended

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
            T_opt[l] = sum(
                t_ij[i, j] * self.z[i, j, l].X for i in self.I for j in self.J
            )
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
            "num_constrs": matrix_size["num_constrs"],
            "num_nzs": matrix_size["num_nzs"],
            "num_qconstrs": matrix_size["num_qconstrs"],
            "warm_start": matrix_size["warm_start"],
            "first_incumbent_seconds": first_incumbent["seconds"],
            "first_incumbent_obj": first_incumbent["obj"],
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


def solve_master_problem(active_patterns, toy_data, solver_limits=None, start_solution=None, log_file=None):
    """One-shot rebuild wrapper kept for small equivalence scripts."""
    master = CentralMaster(toy_data, active_patterns)
    return master.optimize(solver_limits, start_solution=start_solution, log_file=log_file)

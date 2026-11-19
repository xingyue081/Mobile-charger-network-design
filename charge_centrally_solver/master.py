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

def solve_master_problem(active_patterns, toy_data, solver_limits=None, start_solution=None):
    """
    求解 Charge-Centrally 模式的 MISOCP 主问题 (仅包含 active_patterns 中的割平面)。

    solver_limits 只控制这一轮 Gurobi：
    - mip_gap：相对 MIP gap，达到后 Gurobi 可以停，不必证明到默认精度；
    - time_limit：这一轮主问题最多算多少秒。
    时间到了但已经有可行解时，返回该解，并在 decisions['status_name'] 里写 TIME_LIMIT。
    """
    # 1. 解包全局参数
    I_num, J_num, S_num = toy_data['I_num'], toy_data['J_num'], toy_data['S_num']
    I, J, S = range(I_num), range(J_num), range(S_num)
    N_l_Omega = {l: len(toy_data['samples'][l]) for l in S}
    
    n_bar, M_z, M_m = toy_data['n_bar'], toy_data['M_z'], toy_data['M_m']
    tau, kappa, rho, b = toy_data['tau'], toy_data['kappa'], toy_data['rho'], toy_data['b']
    t_ij = toy_data['t_ij']
    f, c_m, c_f, q_l = toy_data['f'], toy_data['c_m'], toy_data['c_f'], toy_data['q_l']
    h_val = toy_data['h_val'] 
    
    model = gp.Model("Charge_Centrally_Master")
    limits = {} if solver_limits is None else solver_limits
    model.setParam('OutputFlag', 1 if limits.get('output_flag') else 0)
    # 不传时保持 Gurobi 默认：MIPGap=1e-4，TimeLimit 不限。
    if limits.get('mip_gap') is not None:
        model.setParam('MIPGap', float(limits['mip_gap']))
    if limits.get('mip_gap_abs') is not None:
        model.setParam('MIPGapAbs', float(limits['mip_gap_abs']))
    if limits.get('optimality_tol') is not None:
        model.setParam('OptimalityTol', float(limits['optimality_tol']))
    if limits.get('time_limit') is not None:
        model.setParam('TimeLimit', float(limits['time_limit']))
    
    # 2. 定义变量
    x = model.addVars(J, vtype=GRB.BINARY, name="x")
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")
    m = model.addVars(J, vtype=GRB.INTEGER, name="m")
    n = model.addVars(I, vtype=GRB.INTEGER, name="n")
    z = model.addVars(I, J, S, vtype=GRB.INTEGER, name="z")
    
    eta = model.addVars([(l, n_idx) for l in S for n_idx in range(N_l_Omega[l])], 
                        lb=-GRB.INFINITY, name="eta")
    
    # 3. 基础物理约束
    for j in J:
        for l in S:
            model.addConstr(gp.quicksum(z[i, j, l] for i in I) <= m[j])
    for i in I:
        model.addConstr(gp.quicksum(y[i, j] for j in J) <= 1)
        model.addConstr(n[i] <= n_bar[i])
        for j in J:
            model.addConstr(y[i, j] <= x[j])
            for l in S:
                model.addConstr(z[i, j, l] <= M_z[i, j] * y[i, j])
    for j in J:
        model.addConstr(m[j] <= M_m[j] * x[j])

    # 上一轮的整数解只作为起点。Gurobi 仍会重新证明这一轮的间隙。
    if start_solution is not None:
        for j in J:
            x[j].Start = start_solution["x"][j]
            m[j].Start = start_solution["m"][j]
        for i in I:
            n[i].Start = start_solution["n"][i]
            for j in J:
                y[i, j].Start = start_solution["y"][i, j]
                for l in S:
                    z[i, j, l].Start = start_solution["z"][i, j, l]
        
    # 4. 动态生成割平面 (eta <= cut_expr)
    for (l, n_idx), patterns in active_patterns.items():
        T_l = gp.quicksum(t_ij[i, j] * z[i, j, l] for i in I for j in J)
        for a_vec in patterns:
            revenue_expr = gp.quicksum(
                (rho + b) * a_vec[i] * (tau * n[i] + kappa * gp.quicksum(z[i, j, l] for j in J))
                for i in I
            )
            # 从外部的字典中提取提前存好的常数 h
            h_current = h_val[l, n_idx, tuple(a_vec)]
            cut_expr = -T_l + revenue_expr - h_current
            model.addConstr(eta[l, n_idx] <= cut_expr)
            
    # 5. 目标函数
    obj = (
        - gp.quicksum(f[j] * x[j] for j in J)
        - c_m * gp.quicksum(m[j] for j in J)
        - c_f * gp.quicksum(n[i] for i in I)
        + gp.quicksum(
            (q_l[l] / N_l_Omega[l]) * eta[l, n_idx]
            for l in S for n_idx in range(N_l_Omega[l])
        )
    )
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    
    # ==========================================
    # 6. 检查求解状态
    # ==========================================

    # OPTIMAL 是证到了 gap。TIME_LIMIT / SUBOPTIMAL 只表示手头有一个可行解。
    stopped_with_solution = model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED)
    if model.SolCount == 0 or not stopped_with_solution:
        return False, {}, {}, {}, {
            'status': model.Status,
            'status_name': _status_name(model.Status),
            'runtime': model.Runtime,
            'mip_gap': None,
            'obj_bound': None,
        }

    # ==========================================
    # 7. 提取容量和 eta
    # ==========================================

    C_opt = {}
    eta_opt = {}
    T_opt = {}

    for l in S:

        T_opt[l] = sum(
            t_ij[i, j] * z[i, j, l].X
            for i in I for j in J
        )

        for i in I:
            C_opt[l, i] = (
                tau * n[i].X
                + kappa * sum(
                    z[i, j, l].X for j in J
                )
            )

        for n_idx in range(N_l_Omega[l]):
            eta_opt[l, n_idx] = eta[l, n_idx].X

    # ==========================================
    # 8. 提取全部建站与调度决策
    # ==========================================

    decisions = {

        'obj': model.ObjVal,
        'obj_bound': model.ObjBound,
        'mip_gap': model.MIPGap,
        'runtime': model.Runtime,
        'node_count': int(model.NodeCount),
        'status': model.Status,
        'status_name': _status_name(model.Status),

        # 是否建设移动枢纽
        'x': {
            j: round(x[j].X)
            for j in J
        },

        # 区域与枢纽的分配关系
        'y': {
            (i, j): round(y[i, j].X)
            for i in I for j in J
        },

        # 每个枢纽配置的移动充电桩数量
        'm': {
            j: round(m[j].X)
            for j in J
        },

        # 每个区域安装的固定充电桩数量
        'n': {
            i: round(n[i].X)
            for i in I
        },

        # 不同日期类型下的移动充电桩调度
        'z': {
            (i, j, l): round(z[i, j, l].X)
            for i in I for j in J for l in S
        }
    }

    return True, C_opt, eta_opt, T_opt, decisions
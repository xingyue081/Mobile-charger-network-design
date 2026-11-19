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

def solve_master_problem(active_patterns, toy_data, solver_limits=None, start_solution=None, log_file=None):
    """
    求解 Charge-Centrally 模式的 MISOCP 主问题 (仅包含 active_patterns 中的割平面)。

    solver_limits 只控制这一轮 Gurobi：
    - mip_gap：相对 MIP gap，达到后 Gurobi 可以停，不必证明到默认精度；
    - time_limit：这一轮主问题最多算多少秒。
    时间到了但已经有可行解时，返回该解，并在 decisions['status_name'] 里写 TIME_LIMIT。

    运输成本和对每个区的调入量各用一条等式定义一次。割只引用这两个量，
    不再把同一段 z 的线性式子复制进每一条割。调度变量本身仍在模型里搜索。
    start_solution 是上一轮的整数方案，只作为这一轮新分支树的起点。
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
    threads = os.environ.get("CG_THREADS")
    if threads:
        model.setParam("Threads", int(threads))
    limits = {} if solver_limits is None else solver_limits
    if log_file is not None:
        # OutputFlag 关掉时，Gurobi 连日志文件也不写。控制台单独关掉。
        model.setParam('OutputFlag', 1)
        model.setParam('LogToConsole', 0)
        model.setParam('LogFile', str(log_file))
        model.setParam('DisplayInterval', 30)
    else:
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
    # 每个日期类型的运输成本，以及每个区在该日期类型收到的移动桩。
    # 二者都由 z 用等式钉死，分支时 z 一变，它们跟着变。
    transport = model.addVars(S, lb=-GRB.INFINITY, name="transport")
    inflow = model.addVars(I, S, lb=0.0, name="inflow")
    
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
    for l in S:
        model.addConstr(
            transport[l] == gp.quicksum(t_ij[i, j] * z[i, j, l] for i in I for j in J),
            name=f"transport_def[{l}]",
        )
        for i in I:
            model.addConstr(
                inflow[i, l] == gp.quicksum(z[i, j, l] for j in J),
                name=f"inflow_def[{i},{l}]",
            )

    # 4. 动态生成割平面。每条割只碰 transport、inflow 和 n。
    for (l, n_idx), patterns in active_patterns.items():
        for a_vec in patterns:
            revenue_expr = gp.quicksum(
                (rho + b) * a_vec[i] * (tau * n[i] + kappa * inflow[i, l])
                for i in I
            )
            # 从外部的字典中提取提前存好的常数 h
            h_current = h_val[l, n_idx, tuple(a_vec)]
            cut_expr = -transport[l] + revenue_expr - h_current
            model.addConstr(eta[l, n_idx] <= cut_expr)

    # 上一轮的建站和调度。eta 不填：新割可能让旧的 eta 不可行，交给 Gurobi 补。
    if start_solution is not None and 'x' in start_solution:
        for j in J:
            x[j].Start = start_solution['x'][j]
            m[j].Start = start_solution['m'][j]
        for i in I:
            n[i].Start = start_solution['n'][i]
            for j in J:
                y[i, j].Start = start_solution['y'][(i, j)]
                for l in S:
                    z[i, j, l].Start = start_solution['z'][(i, j, l)]

    model.update()
    matrix_size = {
        'num_vars': int(model.NumVars),
        'num_constrs': int(model.NumConstrs),
        'num_nzs': int(model.NumNZs),
        'num_qconstrs': int(model.NumQConstrs),
        'warm_start': start_solution is not None and 'x' in start_solution,
    }
    first_incumbent = {'seconds': None, 'obj': None}

    def _record_first_incumbent(cb_model, where):
        if where != GRB.Callback.MIPSOL or first_incumbent['seconds'] is not None:
            return
        first_incumbent['seconds'] = float(cb_model.cbGet(GRB.Callback.RUNTIME))
        first_incumbent['obj'] = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            
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
    # extend_on_time_limit 为假时，到时限就返回当前可行解，不再把时限翻倍。
    # 加割轮次靠这个停在约 10 分钟。默认仍翻倍，非自适应的单轮求解行为不变。
    # Runtime 是这一次调用的时间，所以累加。NodeCount 记在这个模型上，取最后一次的值。
    extend_on_time_limit = bool(limits.get("extend_on_time_limit", True))
    runtime_total = 0.0
    while True:
        model.optimize(_record_first_incumbent)
        runtime_total += float(model.Runtime)
        if (
            not extend_on_time_limit
            or model.Status != GRB.TIME_LIMIT
            or model.SolCount == 0
        ):
            break
        extended = float(model.Params.TimeLimit) * 2.0
        print(
            f"主问题在时限内未证明最优，时限加长到 {extended:.0f} 秒，同一棵分支树接着算。",
            flush=True,
        )
        model.Params.TimeLimit = extended
    
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
        'runtime': runtime_total,
        'time_limit': float(model.Params.TimeLimit),
        'node_count': int(model.NodeCount),
        'iter_count': int(model.IterCount),
        'status': model.Status,
        'status_name': _status_name(model.Status),
        'num_vars': matrix_size['num_vars'],
        'num_constrs': matrix_size['num_constrs'],
        'num_nzs': matrix_size['num_nzs'],
        'num_qconstrs': matrix_size['num_qconstrs'],
        'warm_start': matrix_size['warm_start'],
        'first_incumbent_seconds': first_incumbent['seconds'],
        'first_incumbent_obj': first_incumbent['obj'],

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
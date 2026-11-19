# 当前割平面求解代码

这是 2026-11-16 从本机代码抽出的算法核心，供对照论文 Solution Approach 里尚未写的 cutting plane 小节。

两套当前求解器：

- Charge-centrally：`CG_solver/charge_centrally_solver/`。入口是 `python main.py --instance futian`。主问题不传 `--master-mip-gap` 时用 Gurobi 默认相对间隙 1e-4。
- Charge-locally：`CG_solver/charge_locally_solver/`。入口是 `run_20zone.py`。

不要用 `charge_centrally_solver_1030/`。那是旧快照。本地补能 `constraint_generation.py` 顶部注释里写的 “与 1030 对齐” 也是旧说明，循环以本文件里的当前源码为准。

下面没有放：福田数据读取、`solver_interface.py`、`real_instance.py`、数值实验脚本、穷举对照 `enumeration.py` / `validate.py` / `precheck_small_equivalence.py`。

论文里已经写完的对应关系：

- centrally 的 pattern 是每个 zone 两段，`a_i in {0,1}`，常数是 `h_lt(a)`。
- locally 的 pattern 是每个 zone 三段，`alpha_i in {0,1,2}`，常数是 `h^L_lt(alpha)`。
- 两边的内层都是在椭球与截断盒的交集上算支撑函数。代码里 centrally 用二次约束，locally 的 `support.py` 同样是这个支撑函数。

请只根据下面的源码说明算法在做什么。代码里的间隙放宽、时间上限、同值解、热启动是数值实现，不要写成算法的数学步骤，除非被明确问到。


# ===== charge_centrally_solver/oracle.py =====

```python
import gurobipy as gp
from gurobipy import GRB

def compute_h_val(a_vec, L, U, mu, Sigma_inv, r, rho, b, I_num):
    """
    辅助函数：计算 h(a) = max_D { (rho + b)*D*a - rho*D }
    用于主程序在第 0 代初始化预置缺电模式时使用。
    """
    model = gp.Model("Compute_h")
    model.setParam('OutputFlag', 0)
    
    D = model.addVars(I_num, lb=L, ub=U, vtype=GRB.CONTINUOUS, name="D")
    
    # 纯椭球约束 (D - mu)^T Sigma_inv (D - mu) <= r^2
    diff = {i: D[i] - mu[i] for i in range(I_num)}
    quad_expr = gp.quicksum(
        Sigma_inv[i][j] * diff[i] * diff[j] 
        for i in range(I_num) for j in range(I_num)
    )
    model.addQConstr(quad_expr <= r**2, name="Ellipsoid")
    
    # 目标函数
    obj = gp.quicksum((rho + b) * D[i] * a_vec[i] - rho * D[i] for i in range(I_num))
    model.setObjective(obj, GRB.MAXIMIZE)
    
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"compute_h_val 求解失败："
            f"status={model.Status}, "
            f"pattern={a_vec}"
        )
    return model.ObjVal

def find_worst_pattern(C_star, L, U, mu, Sigma_inv, r, rho, b, return_demand=False, mip_gap=None, time_limit=None):
    """
    分离机 (Separation Oracle)：寻找导致底层利润最小的恶劣缺电模式 a。
    """
    I = len(C_star)
    model = gp.Model("Separation_Oracle")
    model.setParam('OutputFlag', 0)
    if mip_gap is not None:
        model.setParam('MIPGap', float(mip_gap))
    if time_limit is not None:
        model.setParam('TimeLimit', float(time_limit))
    
    D = model.addVars(I, lb=L, ub=U, vtype=GRB.CONTINUOUS, name="D")
    a = model.addVars(I, vtype=GRB.BINARY, name="a")
    v = model.addVars(I, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")
    
    obj = gp.quicksum(
        rho * D[i] - (rho + b) * v[i] + (rho + b) * C_star[i] * a[i] 
        for i in range(I)
    )
    model.setObjective(obj, GRB.MINIMIZE)
    
    # McCormick 线性化约束
    for i in range(I):
        model.addConstr(v[i] >= L[i] * a[i])
        model.addConstr(v[i] <= U[i] * a[i])
        model.addConstr(v[i] >= D[i] - U[i] * (1 - a[i]))
        model.addConstr(v[i] <= D[i] - L[i] * (1 - a[i]))
        
    # 纯椭球约束
    diff = {i: D[i] - mu[i] for i in range(I)}
    quad_expr = gp.quicksum(
        Sigma_inv[i][j] * diff[i] * diff[j] 
        for i in range(I) for j in range(I)
    )
    model.addQConstr(quad_expr <= r**2, name="Ellipsoid")
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        a_star = tuple(round(a[i].X) for i in range(I))
        worst_profit = model.ObjVal
        if return_demand:
            demand = tuple(float(D[i].X) for i in range(I))
            return a_star, worst_profit, demand
        return a_star, worst_profit
    else:
        gap = getattr(model, "MIPGap", None)
        raise RuntimeError(
            f"子问题未达到设定精度：status={model.Status}, runtime={model.Runtime}, "
            f"gap={gap}, sol_count={model.SolCount}"
        )
```


# ===== charge_centrally_solver/master.py =====

```python
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
```


# ===== charge_centrally_solver/main.py 中的 run_cutting_plane_algorithm（第 272–626 行） =====

```python
from master import solve_master_problem
from oracle import find_worst_pattern, compute_h_val

def run_cutting_plane_algorithm(toy_data, limits):
    # ==========================================
    # 1. 业务与统计数据定义
    # ==========================================
    # toy_data 这个名字是 master.py 的参数名，真实实例也用同一套字段。
    I_num, J_num, S_num = toy_data['I_num'], toy_data['J_num'], toy_data['S_num']
    samples = toy_data['samples']
    epsilon_l = toy_data['epsilon_l']
    mu = toy_data['mu']
    Sigma_inv = toy_data['Sigma_inv']
    r, rho, b = toy_data['r'], toy_data['rho'], toy_data['b']

    # ==========================================
    # 2. 初始化割平面和常数项
    # ==========================================
    active_patterns = {}
    for l in range(S_num):
        eps = epsilon_l[l]
        for n_idx, D_sample in enumerate(samples[l]):
            L = np.maximum(0, D_sample - eps)
            U = D_sample + eps
            
            p0, p1 = tuple([0]*I_num), tuple([1]*I_num)
            active_patterns[(l, n_idx)] = [p0, p1]
            
            # 计算初始模式的常数 h 并存入字典
            toy_data['h_val'][(l, n_idx, p0)] = compute_h_val(p0, L, U, mu, Sigma_inv, r, rho, b, I_num)
            toy_data['h_val'][(l, n_idx, p1)] = compute_h_val(p1, L, U, mu, Sigma_inv, r, rho, b, I_num)
            
    # ==========================================
    # 3. 核心循环 Controller
    # ==========================================
    # limits 来自命令行，四层彼此独立：
    # - master_mip_gap / master_time_limit：只限制每一轮 Gurobi 主问题；
    # - cut_tolerance：eta 比分离值高出这么多才算违反、才加割；
    # - max_iterations / max_total_seconds：限制整个割平面循环。
    # adaptive_gap 只在加割轮次把相对间隙放到 0.02。没有新割后单独再解一轮，
    # 间隙收回 1e-4。0.02 来自福田大车队这次的日志：600 秒结束时间隙仍在 0.01 附近，
    # 放到 1e-3 仍然会每轮打满时限。
    adaptive_gap = bool(limits.get("adaptive_gap"))
    certifying = False
    loose_gap = 2e-2
    tight_gap = 1e-4 if limits.get("master_mip_gap") is None else float(limits["master_mip_gap"])
    started = perf_counter()
    iteration = 1
    master_solves = 0
    last_decisions = None
    stop_reason = None
    pattern_order = []
    pattern_seen = set()
    pattern_rounds = []
    for patterns in active_patterns.values():
        for pattern in patterns:
            if pattern not in pattern_seen:
                pattern_seen.add(pattern)
                pattern_order.append(pattern)
    while True:
        elapsed = perf_counter() - started
        if limits['max_total_seconds'] is not None and elapsed >= limits['max_total_seconds']:
            stop_reason = "达到总时间上限"
            print(f"\n达到总时间上限 {limits['max_total_seconds']} 秒，停止。已用 {elapsed:.1f} 秒。")
            break
        if iteration > limits['max_iterations']:
            stop_reason = "达到最大迭代次数"
            print(f"\n达到最大迭代次数 {limits['max_iterations']}，停止。")
            break

        print(f"\n{'='*40}\n 开始第 {iteration} 轮迭代求解主模型...\n{'='*40}")
        
        # 3.1 求解主模型。这一轮自己的时间上限不能超过剩余总时间。
        if adaptive_gap:
            round_gap = tight_gap if certifying else loose_gap
            master_time_limit = 1800.0 if certifying else limits['master_time_limit']
        else:
            round_gap = limits['master_mip_gap']
            master_time_limit = limits['master_time_limit']
        if limits['max_total_seconds'] is not None:
            remaining = limits['max_total_seconds'] - (perf_counter() - started)
            master_time_limit = remaining if master_time_limit is None else min(master_time_limit, remaining)
        print(
            f"本轮主问题间隙目标 {round_gap}，时限 {master_time_limit} 秒。"
            + (" 本轮是收紧间隙。" if certifying else ""),
            flush=True,
        )
        is_solved, C_opt, eta_opt, T_opt, decisions = \
            solve_master_problem(active_patterns, toy_data, {
                'mip_gap': round_gap,
                'mip_gap_abs': limits.get('master_mip_gap_abs'),
                'optimality_tol': limits.get('optimality_tol'),
                'time_limit': master_time_limit,
                'output_flag': limits.get('master_log'),
            }, start_solution=last_decisions)
        
        if not is_solved:
            stop_reason = "主问题没有可行解"
            print(f"❌ 主模型求解失败或无可行解（{decisions.get('status_name')}），算法终止。")
            break
        print(
            f"主问题状态 {decisions['status_name']}，"
            f"目标 {decisions['obj']:.6g}，bound {decisions['obj_bound']:.6g}，"
            f"MIP gap {decisions['mip_gap']:.3g}，用时 {decisions['runtime']:.1f} 秒，"
            f"节点 {decisions['node_count']}"
        )
        master_solves += 1
        last_decisions = decisions
        if decisions['status_name'] != "OPTIMAL":
            print("这一轮主问题未证明最优。后面的分离用的是当前可行解。")
            
        cuts_added = 0
        separation_finished = True
        active_residual = False
        
        # 3.2 呼叫分离机 (Separation Oracle)
        for l in range(S_num):
            if not separation_finished:
                break
            eps = epsilon_l[l]
            for n_idx, D_sample in enumerate(samples[l]):
                L = np.maximum(0, D_sample - eps)
                U = D_sample + eps
                
                # 提取当前规划出的区域容量 C_star
                C_star = [C_opt[l, i] for i in range(I_num)]
                
                # 计算最坏缺电模式。单次分离不能超过剩余总时间。
                remaining = None
                if limits['max_total_seconds'] is not None:
                    remaining = limits['max_total_seconds'] - (perf_counter() - started)
                    if remaining <= 0:
                        stop_reason = "达到总时间上限"
                        separation_finished = False
                        print("\n分离前已达到总时间上限，停止。")
                        break
                try:
                    a_worst, true_worst_profit = find_worst_pattern(
                        C_star=C_star, L=L, U=U, mu=mu, Sigma_inv=Sigma_inv,
                        r=r, rho=rho, b=b, mip_gap=limits.get('separation_mip_gap'),
                        time_limit=remaining,
                    )
                except RuntimeError as exc:
                    stop_reason = "分离未在剩余时间内达到设定 gap"
                    separation_finished = False
                    print(f"\n{exc}")
                    break
                
                # 3.3 判断是否违反割平面约束。已经在池里的 pattern 不再写入，也不改 h。
                if eta_opt[l, n_idx] > true_worst_profit - T_opt[l] + limits['cut_tolerance']:
                    if a_worst not in active_patterns[(l, n_idx)]:
                        active_patterns[(l, n_idx)].append(a_worst)
                        
                        # 由当前分离值反推 h。h 一旦写入，后面的轮次不再改它。
                        h_val_new = sum((rho+b)*C_star[i]*a_worst[i] for i in range(I_num)) - true_worst_profit
                        toy_data['h_val'][(l, n_idx, a_worst)] = h_val_new
                        
                        cuts_added += 1
                        if a_worst not in pattern_seen:
                            pattern_seen.add(a_worst)
                            pattern_order.append(a_worst)
                        print(f"  [发现违背] 情形 {l} 样本 {n_idx}: 新增恶劣模式 {a_worst}")
                    else:
                        excess = eta_opt[l, n_idx] - (true_worst_profit - T_opt[l])
                        # 分离仍用默认相对间隙 1e-4。主问题收到 1e-5 后，eta 贴到割上，
                        # 分离值可以低零点零几。不超过分离自己的相对间隙时，不当成未收敛。
                        room = 1e-4 * (1.0 + abs(true_worst_profit - T_opt[l]))
                        if excess <= room:
                            print(
                                f"  [已有割的数值差] 情形 {l} 样本 {n_idx}: "
                                f"模式 {a_worst}，超出 {excess:.6g}，在分离间隙内"
                            )
                        else:
                            active_residual = True
                            print(
                                f"  [已有割仍被违反] 情形 {l} 样本 {n_idx}: "
                                f"模式 {a_worst}，超出 {excess:.6g}"
                            )
                        
        print(f"--> 本轮迭代共新增 {cuts_added} 条割平面。")
        pattern_rounds.append({
            "iteration": iteration,
            "cuts_added": cuts_added,
            "unique_patterns": len(pattern_order),
            "master_status": decisions["status_name"],
            "master_obj": decisions["obj"],
            "master_seconds": decisions["runtime"],
        })
        print(
            f"PATTERN_ROUND iteration={iteration} cuts_added={cuts_added} "
            f"unique={len(pattern_order)} master={decisions['status_name']} "
            f"obj={decisions['obj']:.6g} master_s={decisions['runtime']:.4f} "
            f"nodes={decisions['node_count']}",
            flush=True,
        )
        
        # 3.4 收敛检查。分离被时限打断时，不能把“本轮没加完的割”当成收敛。
        if not separation_finished:
            print(f"\n分离未完成：{stop_reason}")
            break
        if (
            adaptive_gap
            and cuts_added == 0
            and not active_residual
            and decisions["status_name"] == "OPTIMAL"
            and not certifying
        ):
            certifying = True
            print(
                f"\n没有新的割。下一轮把主问题间隙从 {loose_gap:g} 收到 {tight_gap:g}。",
                flush=True,
            )
            iteration += 1
            continue
        if adaptive_gap and cuts_added > 0 and certifying:
            certifying = False
            print("\n收紧间隙后出现新割，后面的轮次回到较松的间隙。", flush=True)
        if cuts_added == 0 and active_residual:
            stop_reason = "已有割仍被违反"
            print("\n已有割仍被违反，不宣布收敛。没有改写已经写入的 h。")
            break
        if cuts_added == 0:
            if decisions['status_name'] == "OPTIMAL":
                gap = decisions['mip_gap']
                stop_reason = "没有新的割，主问题相对间隙已达到 Gurobi 接受的范围"
                print(f"\n没有新的割。主问题相对间隙为 {gap:.6g}。")
            else:
                stop_reason = "没有新的割，但主问题只是提前停止"
                print("\n没有新的割，但主问题是提前停止的，不能当成严格收敛。")
            
            # ===================================================
            # 新增：在此处打印最终的最优解
            # ===================================================
            print("\n" + "="*50)
            print(f"总目标利润: {decisions['obj']}")
            print("🏆 最终最优容量规划 (C_opt)")
            print("="*50)
            for l in range(S_num):
                res = [f"区域 {i}: {C_opt[l, i]:.2f}" for i in range(I_num)]
                print(f"▶ 情形 (Scenario) {l}:  " + " | ".join(res))
                
            print("\n📊 恶劣情况下的保底利润 (eta_opt)")
            print("-" * 50)
            for l in range(S_num):
                res = [f"样本 {n_idx}: {eta_opt[l, n_idx]:.2f}" for n_idx in range(len(samples[l]))]
                print(f"▶ 情形 (Scenario) {l}:  " + " | ".join(res))
            print("="*50 + "\n")

                        # ==========================================
            # 1. 输出枢纽建站决策
            # ==========================================

            print("\n" + "=" * 50)
            print("最终建站决策")
            print("=" * 50)

            for j in range(J_num):

                if decisions['x'][j] == 1:

                    print(
                        f"枢纽 {j}: 建站, "
                        f"移动充电桩数量 = {decisions['m'][j]}"
                    )

                else:

                    print(f"枢纽 {j}: 不建站")

            # ==========================================
            # 2. 输出各区域固定桩与归属枢纽
            # ==========================================

            print("\n" + "=" * 50)
            print("各区域固定桩与枢纽分配")
            print("=" * 50)

            for i in range(I_num):

                assigned_hub = [
                    j for j in range(J_num)
                    if decisions['y'][i, j] == 1
                ]

                if assigned_hub:
                    hub_info = f"归属枢纽 {assigned_hub[0]}"
                else:
                    hub_info = "未分配移动枢纽"

                print(
                    f"区域 {i}: "
                    f"固定桩数量 = {decisions['n'][i]}, "
                    f"{hub_info}"
                )

            # ==========================================
            # 3. 输出不同日期类型的移动桩调度
            # ==========================================

            print("\n" + "=" * 50)
            print("不同日期类型下的移动充电桩调度")
            print("=" * 50)

            for l in range(S_num):

                print(f"\n日期类型 {l}:")

                for j in range(J_num):

                    if decisions['x'][j] == 0:
                        continue

                    print(f"  枢纽 {j}:")

                    dispatched = False

                    for i in range(I_num):

                        z_val = decisions['z'][i, j, l]

                        if z_val > 0:

                            print(
                                f"    -> 区域 {i}: "
                                f"调度 {z_val} 个移动充电桩"
                            )

                            dispatched = True

                    if not dispatched:
                        print("    无移动充电桩调度")

            print("\n" + "=" * 50)
            break # 终止循环

        iteration += 1

    plan = None
    if last_decisions is not None and 'x' in last_decisions:
        plan = {
            'x': last_decisions['x'],
            'y': last_decisions['y'],
            'm': last_decisions['m'],
            'n': last_decisions['n'],
            'z': last_decisions['z'],
        }
    return {
        'objective': None if last_decisions is None else last_decisions.get('obj'),
        'bound': None if last_decisions is None else last_decisions.get('obj_bound'),
        'mip_gap': None if last_decisions is None else last_decisions.get('mip_gap'),
        'iterations': master_solves,
        'stop_reason': stop_reason,
        'status_name': None if last_decisions is None else last_decisions.get('status_name'),
        'active_patterns': active_patterns,
        'pattern_order': pattern_order,
        'pattern_rounds': pattern_rounds,
        'plan': plan,
    }
```


# ===== charge_locally_solver/patterns.py =====

```python
"""三段利润的 pattern、斜率和不乘需求的常数项。

pattern 是每个 zone 选中的段号，取值 0、1、2。它是仿射片的索引，
不是规划者决定的部署变量。

一个区的运营利润，在 fixed 先服务、本地增量成本只对 mobile 实际服务量收取时，是

    rho * D - g * (D - tau*n)^+ - (rho + b - g) * (D - tau*n - kappa*z)^+。

令 a = tau*n，K = kappa*z。0 <= g <= rho+b 时，它等于下面三段仿射函数的最小值。

第 0 段，D <= a：
    斜率 rho，常数 0。
第 1 段，a < D <= a+K：
    斜率 rho-g，常数 g*a。
第 2 段，D > a+K：
    斜率 -b，常数 (rho+b)*a + (rho+b-g)*K。

最坏利润先对 pattern 取最小，再对局部需求集合取最小。因为常数项不含 D，
两步可以交换：每个 pattern 先在局部集合上算出 min (斜率·D + 常数)，再对 pattern 取最小。
约束生成加的就是这些 pattern 的上界割。
"""

from __future__ import annotations

from itertools import product

import numpy as np

PIECES = (0, 1, 2)


def all_patterns(zone_count: int) -> list[tuple[int, ...]]:
    """2 个 zone 有 9 个 pattern，3 个 zone 有 27 个。

    20 个 zone 有 3^20 个，不能全部枚举。穷举对照只允许 2 个 zone。
    """
    return list(product(PIECES, repeat=zone_count))


def slope(piece: int, instance: dict) -> float:
    """三段斜率：rho、rho-g、-b。"""
    rho, penalty, local_g = instance["rho"], instance["penalty"], instance["local_g"]
    return (rho, rho - local_g, -penalty)[piece]


def beta_zone(piece: int, fixed_count: float, mobile_count: float, instance: dict) -> float:
    """一个 zone 在指定段上、不乘 D 的常数。

    第 0 段没有容量常数。
    第 1 段只含 fixed charger 容量，等于 g*tau*n。
    第 2 段同时含 fixed 与 mobile 容量，等于
    (rho+b)*tau*n + (rho+b-g)*kappa*z。

    fixed_count、mobile_count 可以是普通数，也可以是主问题里的 Gurobi 变量。
    传变量时，返回值是一条线性表达式，直接放进割 eta <= beta - h。
    """
    rho, penalty, local_g = instance["rho"], instance["penalty"], instance["local_g"]
    tau, kappa = instance["tau"], instance["kappa"]
    if piece == 0:
        return 0.0
    if piece == 1:
        return local_g * tau * fixed_count
    if piece == 2:
        return (rho + penalty) * tau * fixed_count + (rho + penalty - local_g) * kappa * mobile_count
    raise ValueError(f"未知的利润段 {piece}")


def beta_pattern(pattern: tuple[int, ...], fixed_counts, mobile_counts, instance: dict) -> float:
    """把各 zone 的 beta 相加。fixed_counts 与 mobile_counts 可以是数或 Gurobi 变量。

    pattern[i] 对应 zones 里的第 i 个区，也对应需求向量的第 i 列。
    当前调用方都按 0,1,2,... 的顺序传入，所以位置和区号一致。
    """
    total = 0.0
    for zone, piece in enumerate(pattern):
        total += beta_zone(piece, fixed_counts[zone], mobile_counts[zone], instance)
    return total


def canonical_pattern(pattern: tuple[int, ...], instance: dict) -> tuple[int, ...]:
    """g=0 时第 0 段和第 1 段是同一条仿射函数，写成同一种 pattern。

    斜率都是 rho，beta 都是 0。换成同一种写法不会改变割。
    g>0 时两段不同，pattern 原样返回。

    【核对】20 区试跑把 g 设成 0，所以会走这个合并。合成 2 区对照的 g 是 1，不合并。
    """
    if float(instance["local_g"]) != 0.0:
        return tuple(pattern)
    return tuple(0 if piece == 1 else int(piece) for piece in pattern)


def support_direction(pattern: tuple[int, ...], instance: dict) -> np.ndarray:
    """支持函数的方向是斜率的相反数。最小化 alpha·D 等于 -max(-alpha·D)。

    因此主问题里的割写成 eta <= beta - h，其中 h = max (-斜率 · D)。
    """
    return np.array([-slope(piece, instance) for piece in pattern], dtype=float)
```


# ===== charge_locally_solver/support.py =====

```python
"""在椭球与截断盒子的交集上计算支持函数。

这是原始凸问题：变量是需求 D，约束是集合本身，目标是线性函数的最大值。
h 只依赖数据和 pattern 的方向，不依赖 n、z，所以可以在主问题之外预先算成常数。

局部集合是三部分的交集：
- 样本周围的 l_infinity 盒子，下界已截到 0（instance.local_box）；
- 全体样本共用的 Mahalanobis 椭球 (D-mu)^T Sigma^{-1} (D-mu) <= r^2。

Sigma = L L^T 时，椭球写成 ||L^{-1}(D-mu)||^2 <= r^2。
h(pattern) = max (-斜率 · D)，最大值就在这个交集上取。
"""

from __future__ import annotations

import os

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instance import local_box, sample_keys
from patterns import all_patterns, support_direction


def _precision_and_whitening(instance: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = np.asarray(instance["covariance"], dtype=float)
    precision = np.linalg.inv(covariance)
    cholesky = np.linalg.cholesky(covariance)
    # Sigma = L L^T 时，椭球约束写成 ||L^{-1}(D-mu)||^2 <= r^2。
    inverse_cholesky = np.linalg.solve(cholesky, np.eye(cholesky.shape[0]))
    return precision, cholesky, inverse_cholesky


def _optimize_support(model: gp.Model) -> float | None:
    model.Params.OutputFlag = 0
    # 默认 1e-6 时，目标约 5e4 的支持函数可以少算约 0.01，仍被报告为最优。
    # 分离用绝对差 1e-4 验收，所以先收到 1e-8。福田 35 维上这一档偶发失败，
    # 失败后再放宽精度重解；两种坐标都失败才由调用方报错。
    for tolerance, focus in ((1e-8, 0), (1e-6, 2), (1e-5, 3)):
        model.Params.BarQCPConvTol = tolerance
        model.Params.NumericFocus = focus
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return float(model.ObjVal)
    return None


def support_value(direction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  instance: dict) -> float:
    """求解 max direction·D，D 位于局部盒子和 Mahalanobis 椭球的交集。

    direction 已经是斜率的相反数。同一次精度下用两种等价坐标各算一次，
    取更大的最优值。某一种停在次优时只用另一种。两种都失败才抛错。
    """
    mean = np.asarray(instance["mean"], dtype=float)
    _, cholesky, inverse_cholesky = _precision_and_whitening(instance)
    radius = float(instance["radius"])
    demand_model = gp.Model("local_support")
    threads = os.environ.get("CG_THREADS")
    if threads:
        demand_model.Params.Threads = int(threads)
    demand = demand_model.addMVar(len(direction), lb=lower, ub=upper, name="D")
    whitened = inverse_cholesky @ (demand - mean)
    demand_model.addConstr(whitened @ whitened <= radius ** 2, name="SEP_ELLIPSOID")
    demand_model.setObjective(direction @ demand, GRB.MAXIMIZE)
    values = [value for value in (_optimize_support(demand_model),) if value is not None]

    whitened_model = gp.Model("local_support_whitened")
    if threads:
        whitened_model.Params.Threads = int(threads)
    coefficient = whitened_model.addMVar(len(direction), lb=-GRB.INFINITY, name="u")
    whitened_model.addConstr(coefficient @ coefficient <= radius ** 2, name="SEP_ELLIPSOID")
    demand_expr = mean + cholesky @ coefficient
    whitened_model.addConstr(demand_expr >= lower, name="box_lower")
    whitened_model.addConstr(demand_expr <= upper, name="box_upper")
    whitened_model.setObjective(direction @ demand_expr, GRB.MAXIMIZE)
    whitened_value = _optimize_support(whitened_model)
    if whitened_value is not None:
        values.append(whitened_value)
    if not values:
        raise RuntimeError("支持函数求解失败，两种坐标都没有得到最优解")
    return max(values)


class SupportCache:
    """按需要计算 h。20 个 zone 不能预先枚举 3^20 个 pattern。

    主问题每次真正加入一条割，才为那个 (日期类型, 样本, pattern) 算一次 h。
    同一个 key 再出现时直接读表。
    """

    def __init__(self, instance: dict):
        self.instance = instance
        self.table: dict[tuple, float] = {}

    def __getitem__(self, key: tuple) -> float:
        if key not in self.table:
            event, sample_idx, pattern = key
            lower, upper = local_box(self.instance, event, sample_idx)
            direction = support_direction(pattern, self.instance)
            self.table[key] = support_value(direction, lower, upper, self.instance)
        return self.table[key]


def precompute_supports(instance: dict) -> dict[tuple, float]:
    """为每个样本、每个 pattern 计算 h。穷举和约束生成必须共用这张表。

    【核对】validate.py 的“目标差为 0”用的就是这张共用表。
    它证明割的循环和“全部 pattern 的 beta-h”一致，
    不能单独证明 h 的几何和另一套直接利润计算一致。直接利润对照在 9 月的公式审计里。
    """
    table = {}
    patterns = all_patterns(len(instance["zones"]))
    for event, sample_idx in sample_keys(instance):
        lower, upper = local_box(instance, event, sample_idx)
        center = np.asarray(instance["samples"][event][sample_idx], dtype=float)
        precision, _, _ = _precision_and_whitening(instance)
        diff = center - np.asarray(instance["mean"], dtype=float)
        if float(diff @ precision @ diff) > float(instance["radius"]) ** 2 + 1e-8:
            raise ValueError(f"样本 {(event, sample_idx)} 不在椭球内")
        for pattern in patterns:
            direction = support_direction(pattern, instance)
            table[event, sample_idx, pattern] = support_value(direction, lower, upper, instance)
    return table
```


# ===== charge_locally_solver/separation.py =====

```python
"""给定一组 n、z 和当前 eta，求一个样本上的最坏三段利润。

每个样本的分离模型只建一次。n、z 只出现在目标的 beta 里，
约束（盒子、McCormick、椭球）不变，所以之后只改目标再求解。
椭球写成二范数的平方，和二次型 ||L^{-1}(D-mu)||^2 <= r^2 是同一个集合。

一次求解可以带回搜索中碰到的多个最优 pattern。每条都是有效割。
椭球仍写成凸二次不等式。Gurobi 在 NonConvex=0 下会把范数等式看成非凸二次等式，所以这里不用那种写法。

分离在最小化 sum_i (斜率_{piece(i)} * D_i + beta_i)。
每个区用三个 0-1 变量选一段，再用 McCormick 等式把“选中段的 D”写出来。
因为利润对区可加，而区与区只通过椭球连在一起，这个混合整数模型等于
min_D min_pattern (斜率·D + beta)。
"""

from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from instance import local_box
from patterns import PIECES, beta_zone, canonical_pattern, slope


# 键是 (实例对象的 id, 日期类型, 样本编号)。几何数据变了但还是同一个 dict 时，
# 这里会继续用旧模型。【核对】当前脚本不会在同一次求解中改盒子或椭球。
_ORACLES: dict[tuple, dict] = {}


def solve_separation(solution: dict, event: str, sample_idx: int, instance: dict) -> dict:
    """求解一个样本的分离问题。solution 使用 n、z、eta 三个字段。

    返回的 violation = 主问题当前 eta - 分离得到的最坏利润。
    violation 大于容差时，这条部署的 eta 还偏高，需要把对应 pattern 加进主问题。
    分离没有证明最优时，certified_optimal 为 False。调用方不能把这种情况记成收敛。
    """
    oracle = _oracle(instance, event, sample_idx)
    zones = oracle["zones"]
    fixed_counts = [float(solution["n"][zone]) for zone in zones]
    mobile_counts = [float(solution["z"][event][zone]) for zone in zones]
    current_eta = float(solution["eta"][event, sample_idx])
    model = oracle["model"]
    piece = oracle["piece"]
    product = oracle["product"]
    # beta 用当前解的 n、z 算成数字。换一个部署时只改目标，不改建模型。
    profit = gp.quicksum(
        slope(piece_id, instance) * product[zone, piece_id]
        + beta_zone(piece_id, fixed_counts[zone_id], mobile_counts[zone_id], instance) * piece[zone, piece_id]
        for zone_id, zone in enumerate(zones)
        for piece_id in PIECES
    )
    model.setObjective(profit, GRB.MINIMIZE)
    pool = int(instance.get("separation_pool_solutions") or 1)
    model.Params.PoolSolutions = max(pool, 1)
    # 1：保留这棵搜索树里已经找到的解，不再为了凑满解池继续搜索。
    # 漏掉的并列 pattern 仍然是有效割，下一轮还可以被发现。不影响割的正确性。
    model.Params.PoolSearchMode = 1 if pool > 1 else 0
    if instance.get("separation_time_limit") is not None:
        model.Params.TimeLimit = max(float(instance["separation_time_limit"]), 1e-3)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return {
            "status": "NOT_CERTIFIED",
            "certified_optimal": False,
            "native_status": int(model.Status),
            "patterns": [],
        }
    tolerance = float(instance["separation_tolerance"])
    worst_profit = float(model.ObjVal)
    patterns = []
    scale = 1.0 + abs(worst_profit)
    for number in range(int(model.SolCount)):
        model.Params.SolutionNumber = number
        if abs(float(model.PoolObjVal) - worst_profit) > 1e-6 * scale:
            continue
        # 每个区取等于 1 的那段。g=0 时随后会把第 1 段收成第 0 段。
        pattern = canonical_pattern(tuple(
            int(max(PIECES, key=lambda piece_id: piece[zone, piece_id].Xn))
            for zone in zones
        ), instance)
        if pattern not in patterns:
            patterns.append(pattern)
    violation = current_eta - worst_profit
    violated = violation > tolerance
    return {
        "status": "FOUND_VIOLATION" if violated else "CERTIFIED_NO_VIOLATION",
        "certified_optimal": True,
        "pattern": None if not patterns else patterns[0],
        "patterns": patterns if violated else [],
        "worst_profit": worst_profit,
        "violation": violation,
        "violated": violated,
    }


def _oracle(instance: dict, event: str, sample_idx: int) -> dict:
    key = (id(instance), event, sample_idx)
    oracle = _ORACLES.get(key)
    if oracle is None:
        oracle = _build_oracle(instance, event, sample_idx)
        _ORACLES[key] = oracle
    return oracle


def _build_oracle(instance: dict, event: str, sample_idx: int) -> dict:
    """为一个样本建立分离模型。需求盒和椭球在这里固定，n、z 不进入约束。"""
    zones = tuple(instance["zones"])
    lower, upper = local_box(instance, event, sample_idx)
    mean = np.asarray(instance["mean"], dtype=float)
    covariance = np.asarray(instance["covariance"], dtype=float)
    cholesky = np.linalg.cholesky(covariance)
    inverse_cholesky = np.linalg.solve(cholesky, np.eye(len(zones)))
    radius = float(instance["radius"])
    model = gp.Model(f"local_separation_{event}_{sample_idx}")
    model.Params.OutputFlag = 0
    # MIPGap 保持 Gurobi 默认 1e-4。椭球约束是凸二次，不打开非凸模式。
    # 与支持函数使用同一个 BarQCPConvTol，避免两边各停在默认精度的不同位置。
    model.Params.BarQCPConvTol = 1e-8
    model.Params.NonConvex = 0
    model.Params.Threads = 1
    demand, piece, product = {}, {}, {}
    for zone_id, zone in enumerate(zones):
        demand[zone] = model.addVar(lb=float(lower[zone_id]), ub=float(upper[zone_id]), name=f"D[{zone}]")
        for piece_id in PIECES:
            piece[zone, piece_id] = model.addVar(vtype=GRB.BINARY, name=f"y[{zone},{piece_id}]")
            # product = piece * D。四个不等式是 [下界, 上界] 上的 McCormick 包络。
            # piece 是 0-1 时，包络收成 product = D（选中）或 product = 0（没选中）。
            product[zone, piece_id] = model.addVar(lb=-GRB.INFINITY, name=f"w[{zone},{piece_id}]")
            chosen = piece[zone, piece_id]
            demand_var = demand[zone]
            low, high = float(lower[zone_id]), float(upper[zone_id])
            model.addConstr(product[zone, piece_id] >= low * chosen)
            model.addConstr(product[zone, piece_id] <= high * chosen)
            model.addConstr(product[zone, piece_id] >= demand_var - high * (1 - chosen))
            model.addConstr(product[zone, piece_id] <= demand_var - low * (1 - chosen))
        model.addConstr(gp.quicksum(piece[zone, piece_id] for piece_id in PIECES) == 1)
    whitened = model.addVars(len(zones), lb=-GRB.INFINITY, name="whitened")
    for row in range(len(zones)):
        model.addConstr(whitened[row] == gp.quicksum(
            float(inverse_cholesky[row, column]) * (demand[zones[column]] - float(mean[column]))
            for column in range(len(zones))
        ))
    model.addQConstr(
        gp.quicksum(whitened[row] * whitened[row] for row in range(len(zones))) <= radius ** 2,
        name="SEP_ELLIPSOID",
    )
    return {"model": model, "zones": zones, "piece": piece, "product": product}
```


# ===== charge_locally_solver/master.py =====

```python
"""本地补能的受限主问题。

主问题包含投资、全部部署、流平衡和 repositioning。
每个样本的 eta 只受当前已经加入的 pattern 约束。
add_pattern 只加一条线性割，不在这里调用分离。

这是最大化问题。目标是

    - c^f * sum n_i - c^m * m
    - 按 p_kl 加权的单程 repositioning 成本
    + 按 q_l/N_{l,Omega} 加权的各样本 eta。

eta 只有上界割，没有下界。最大化会把 eta 推到已加入割的最低上界。
割还没加全时，这个目标是真最坏利润的上界。objective 和 bound 相等，
只说明当前这个受限主问题自己解完了，不说明分离已经证完。

加速（与 charge_centrally_solver_1030 对齐，只改求解，不改模型含义）：
- 容量提成：fixed_cap_i = tau*n_i，mobile_cap_{l,i} = kappa*z_{il}，割只引用二者；
- 上一轮整数方案作为下一轮 MIP start；
- set_mip_gap 供自适应间隙切换。
"""

from __future__ import annotations

import os

import gurobipy as gp
from gurobipy import GRB

from instance import sample_keys


def _upper_bound(bounds, zone) -> float:
    """n 的上界可以是所有 zone 共用的一个数，也可以按 zone 下标给不同的数。"""
    if isinstance(bounds, (int, float)):
        return float(bounds)
    return float(bounds[zone])


class LocalMaster:
    """保存 Gurobi 模型和已经加入的 pattern。"""

    def __init__(self, instance: dict, supports: dict, initial_patterns: dict[tuple, set[tuple]]):
        self.instance = instance
        self.supports = supports
        self.active = {key: set(patterns) for key, patterns in initial_patterns.items()}
        self.model = gp.Model("local_master")
        threads = os.environ.get("CG_THREADS")
        if threads:
            self.model.Params.Threads = int(threads)
        self.model.Params.OutputFlag = 1 if instance.get("master_log") else 0
        # 不传 master_mip_gap 时保持 Gurobi 默认：相对间隙 1e-4，绝对间隙 1e-10。
        # 传入更小的相对间隙后，OPTIMAL 仍表示“已达到该间隙”，不是间隙一定为 0。
        if self.instance.get("master_mip_gap") is not None:
            self.model.Params.MIPGap = float(self.instance["master_mip_gap"])
        if self.instance.get("master_mip_gap_abs") is not None:
            self.model.Params.MIPGapAbs = float(self.instance["master_mip_gap_abs"])
        zones = instance["zones"]
        events = instance["date_types"]
        # n_i：区 i 的 fixed charger 数量。观察日期和需求之前决定。
        self.n = self.model.addVars(
            zones, vtype=GRB.INTEGER, lb=0,
            ub={zone: _upper_bound(instance["n_bar"], zone) for zone in zones},
            name="n",
        )
        # m：城市级共享车队规模。本地模式没有 hub，也没有 m_j。
        self.fleet = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=instance["m_bar"], name="m")
        # z_{il}：看到日期类型 l 之后、需求实现之前的部署。每个 l 都把 m 辆车全部派出去。
        self.z = self.model.addVars(events, zones, vtype=GRB.INTEGER, lb=0, ub=instance["m_bar"], name="z")
        transitions = list(instance["transition_probability"])
        # r_{i i' k l}：日期类型从 k 到 l 时，从区 i 移到区 i' 的车辆数。
        # Gurobi 会把二元组键展开，所以变量下标是 (k, l, i, i')。
        # 【核对】这里保留整数。z 为整数时，这组流平衡矩阵全幺模，连续 r 的最优值相同；
        # 整数写法更慢，但没有放宽可行域。
        self.flow = self.model.addVars(transitions, zones, zones, vtype=GRB.INTEGER, lb=0,
                                       ub=instance["m_bar"], name="r")
        # 容量提成：割只碰 fixed_cap / mobile_cap，不再把 tau*n、kappa*z 复制进每一条割。
        tau = float(instance["tau"])
        kappa = float(instance["kappa"])
        self.fixed_cap = self.model.addVars(zones, lb=0.0, name="fixed_cap")
        self.mobile_cap = self.model.addVars(events, zones, lb=0.0, name="mobile_cap")
        for zone in zones:
            self.model.addConstr(
                self.fixed_cap[zone] == tau * self.n[zone],
                name=f"fixed_cap_def[{zone}]",
            )
            for event in events:
                self.model.addConstr(
                    self.mobile_cap[event, zone] == kappa * self.z[event, zone],
                    name=f"mobile_cap_def[{event},{zone}]",
                )
        # eta_{l,n}：样本 n 在日期类型 l 下的最坏运营利润。下界放开，靠割往下压。
        self.eta = {
            key: self.model.addVar(lb=-GRB.INFINITY, name=f"eta_{key[0]}_{key[1]}")
            for key in sample_keys(instance)
        }
        # 每种日期类型都部署完全部车队：sum_i z_{il} = m。
        self.model.addConstrs(
            (gp.quicksum(self.z[event, zone] for zone in zones) == self.fleet for event in events),
            name="FLEET",
        )
        for src, dst in transitions:
            # 离开区 origin 的车辆数等于日期类型 src 下部署在 origin 的车辆数。
            self.model.addConstrs(
                (gp.quicksum(self.flow[src, dst, origin, destination] for destination in zones) == self.z[src, origin]
                 for origin in zones),
                name=f"FLOW_OUT_{src}_{dst}",
            )
            # 进入区 destination 的车辆数等于日期类型 dst 下部署在 destination 的车辆数。
            self.model.addConstrs(
                (gp.quicksum(self.flow[src, dst, origin, destination] for origin in zones) == self.z[dst, destination]
                 for destination in zones),
                name=f"FLOW_IN_{src}_{dst}",
            )
        for key, patterns in self.active.items():
            for pattern in patterns:
                self._add_cut(key, pattern)
        # 单程成本 gamma * d_{ii'} * r。不加往返系数 2，也不把这项放进需求的最坏期望。
        # 只对字典里真正出现的 (k,l) 计费。概率为 0 的转移没有变量，也没有成本。
        reposition = gp.quicksum(
            instance["transition_probability"][src, dst]
            * instance["gamma"]
            * instance["distance"][origin, destination]
            * self.flow[src, dst, origin, destination]
            for src, dst in transitions
            for origin in zones
            for destination in zones
        )
        operating = gp.quicksum(
            instance["sample_weight"][event] * variable
            for (event, _), variable in self.eta.items()
        )
        investment = (
            instance["fixed_cost"] * gp.quicksum(self.n[zone] for zone in zones)
            + instance["mobile_cost"] * self.fleet
        )
        self.model.setObjective(-investment - reposition + operating, GRB.MAXIMIZE)
        self._last_start = None

    def set_mip_gap(self, mip_gap: float | None) -> None:
        """自适应间隙：加割轮次用松间隙，无新割后再收紧。"""
        if mip_gap is None:
            self.model.Params.MIPGap = 1e-4
        else:
            self.model.Params.MIPGap = float(mip_gap)

    def _beta_from_caps(self, pattern: tuple, event: str):
        """用提成容量写 beta，数值上等于 patterns.beta_pattern(n, z)。

        piece 0：0
        piece 1：g * fixed_cap
        piece 2：(rho+b)*fixed_cap + (rho+b-g)*mobile_cap
        """
        rho = float(self.instance["rho"])
        penalty = float(self.instance["penalty"])
        local_g = float(self.instance["local_g"])
        total = 0.0
        for zone, piece in enumerate(pattern):
            if piece == 0:
                continue
            if piece == 1:
                total += local_g * self.fixed_cap[zone]
            elif piece == 2:
                total += (
                    (rho + penalty) * self.fixed_cap[zone]
                    + (rho + penalty - local_g) * self.mobile_cap[event, zone]
                )
            else:
                raise ValueError(f"未知的利润段 {piece}")
        return total

    def _add_cut(self, key: tuple, pattern: tuple) -> None:
        """加入 eta <= beta(n,z) - h。

        h 在这里已经是常数。beta 对 n、z 是线性的，所以割对以后所有部署都有效。
        在产生这条割的那个部署上，它等于该 pattern 的局部最坏利润。
        这里用提成后的 fixed_cap / mobile_cap，等价于原来的 tau*n、kappa*z。
        """
        event, sample_idx = key
        height = self.supports[event, sample_idx, pattern]
        self.model.addConstr(
            self.eta[key] <= self._beta_from_caps(pattern, event) - height,
            name=f"CUT_{event}_{sample_idx}_{''.join(map(str, pattern))}",
        )

    def add_pattern(self, event: str, sample_idx: int, pattern: tuple[int, ...]) -> bool:
        """加入一条新割。pattern 已存在时返回 False，调用者应把它当成错误。"""
        key = (event, sample_idx)
        if pattern in self.active[key]:
            return False
        self.active[key].add(pattern)
        self._add_cut(key, pattern)
        return True

    def _apply_warm_start(self, point: dict | None) -> None:
        """上一轮建站与调度。eta 不填：新割可能让旧 eta 不可行。"""
        if point is None or "n" not in point:
            return
        zones = self.instance["zones"]
        events = self.instance["date_types"]
        for zone, value in zip(zones, point["n"]):
            self.n[zone].Start = float(value)
        self.fleet.Start = float(point["m"])
        for event in events:
            for zone, value in zip(zones, point["z"][event]):
                self.z[event, zone].Start = float(value)

    def optimize(self) -> dict:
        """求解当前主问题，并读出规划变量和每个样本的 eta。

        pool > 1 时，Gurobi 在证明最优之后继续找同目标值的其他解。
        这些解只用于多切几条割，不改变“当前最优目标是上界”这件事。
        """
        pool = int(self.instance.get("master_pool_solutions") or 1)
        self.model.Params.PoolSolutions = max(pool, 1)
        # 2 表示在证明最优之后继续找另外几个最优解。1 表示只保留搜索过程中碰到的解。
        self.model.Params.PoolSearchMode = 2 if pool > 1 else 0
        if self.instance.get("master_time_limit") is not None:
            self.model.Params.TimeLimit = max(float(self.instance["master_time_limit"]), 1e-3)
        self._apply_warm_start(self._last_start)
        self.model.optimize()
        status = self.model.Status
        search = {
            "node_count": int(self.model.NodeCount),
            "runtime": float(self.model.Runtime),
            "warm_start": self._last_start is not None,
        }
        if status == GRB.OPTIMAL:
            result = self._read_solution("OPTIMAL") | search
            self._last_start = result
            return result
        # 时限内已有可行解时仍返回该解。循环可以拿它分离，但不能凭它宣布收敛。
        if status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            result = self._read_solution("FEASIBLE") | search
            self._last_start = result
            return result
        return {
            "status": "NOT_OPTIMAL",
            "native_status": int(status),
            "sol_count": int(self.model.SolCount),
        } | search

    def _read_solution(self, status: str) -> dict:
        best = self._read_point(0, status)
        alternates = []
        best_obj = best["objective"]
        scale = 1.0 + abs(best_obj)
        for number in range(1, int(self.model.SolCount)):
            point = self._read_point(number, status)
            if abs(point["objective"] - best_obj) <= 1e-6 * scale:
                alternates.append(point)
        best["alternates"] = alternates
        return best

    def _read_point(self, number: int, status: str) -> dict:
        self.model.Params.SolutionNumber = number
        zones = self.instance["zones"]
        events = self.instance["date_types"]
        transitions = list(self.instance["transition_probability"])
        taz_ids = list(self.instance.get("taz_ids") or zones)
        flows = []
        for src, dst in transitions:
            for origin in zones:
                for destination in zones:
                    vehicles = int(round(self.flow[src, dst, origin, destination].Xn))
                    if vehicles <= 0:
                        continue
                    flows.append({
                        "from_date": src,
                        "to_date": dst,
                        "origin_index": int(origin),
                        "destination_index": int(destination),
                        "origin_taz": int(taz_ids[origin]),
                        "destination_taz": int(taz_ids[destination]),
                        "vehicles": vehicles,
                    })
        return {
            "status": status,
            "native_status": int(self.model.Status),
            "objective": float(self.model.PoolObjVal),
            "bound": float(self.model.ObjBound),
            "n": [self.n[zone].Xn for zone in zones],
            "m": self.fleet.Xn,
            "z": {event: [self.z[event, zone].Xn for zone in zones] for event in events},
            "flow": flows,
            "eta": {key: variable.Xn for key, variable in self.eta.items()},
        }

    def dispose(self) -> None:
        self.model.dispose()
```


# ===== charge_locally_solver/constraint_generation.py =====

```python
"""本地补能的约束生成循环。

循环和集中补能 constraint_generation.py 相同：
解受限主问题，逐个样本调用分离，发现违反就把 pattern 加进主问题。
只有每个样本都被证明没有违反时才算收敛。
分离失败不能记成收敛。

一轮的实际做法：
1. 解当前主问题。上界连续两轮不动时，再取几个同值解一起分离。
2. 从上次停下来的样本继续往后看。
3. 一旦已经发现违反，这一轮最多再看 samples_per_iteration 个样本，
   并且最多准备 cut_cap 条新割。没有发现违反时，这个上限不起作用，
   会把剩下的样本全部看完。
4. 有新割就加进主问题，进入下一轮。
5. 一整圈都没有新割、主问题是 OPTIMAL、也没有数值残差，才记收敛。

因此 samples_per_iteration=8 不是“全程只检查 8 个样本”。
它只限制已经找到违反之后，这一轮还要再扫多远。

加速（与 charge_centrally_solver_1030 对齐）：
- 主问题未证明到间隙时，分离只核算，不写新割；
- 可选 adaptive_gap：加割轮次用较松间隙，无新割后再收紧；
- 热启动与容量提成在 master.py。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from instance import sample_keys
from master import LocalMaster
from patterns import canonical_pattern
from separation import solve_separation


def solve_constraint_generation(instance: dict, supports: dict, *, max_iterations: int = 30) -> dict:
    """从每个样本的第 0 段 pattern 出发，生成被违反的三段割。

    第 0 段是 D 不超过 fixed 容量时的仿射片，斜率 rho、常数 0。
    它一定是利润的一个上界，所以可以作为初始割。
    调用方如果给了 resume_patterns，就从那些已经加入的割接着算。
    """
    if max_iterations < 1:
        raise ValueError("max_iterations 必须是正整数")
    keys = sample_keys(instance)
    zone_count = len(instance["zones"])
    initial = instance.get("resume_patterns") or {key: {(0,) * zone_count} for key in keys}
    started = time.monotonic()
    deadline = None
    if instance.get("wall_clock_seconds") is not None:
        deadline = started + float(instance["wall_clock_seconds"])
    master = LocalMaster(instance, supports, initial)
    history = []
    solution = None
    cursor = 0
    adaptive_gap = bool(instance.get("adaptive_gap"))
    loose_gap = 1e-3
    tight_gap = 1e-4 if instance.get("master_mip_gap") is None else float(instance["master_mip_gap"])
    base_master_time = instance.get("master_time_limit")
    tight_time = 1800.0 if adaptive_gap else None
    certifying = False
    iteration = 0
    try:
        while True:
            iteration += 1
            if iteration > max_iterations:
                max_iterations *= 2
                print(f"达到迭代上限，上限加长到 {max_iterations} 后继续。", flush=True)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return _finish("TIME_LIMIT", False, history, solution, master, started, instance)
            if adaptive_gap:
                round_gap = tight_gap if certifying else loose_gap
                master.set_mip_gap(round_gap)
                round_time = tight_time if certifying else base_master_time
            else:
                round_gap = instance.get("master_mip_gap")
                round_time = base_master_time
            if remaining is not None and round_time is not None:
                instance["master_time_limit"] = min(float(round_time), remaining)
            elif remaining is not None:
                instance["master_time_limit"] = remaining
            elif round_time is not None:
                instance["master_time_limit"] = float(round_time)
            else:
                instance.pop("master_time_limit", None)

            recent = [row.get("objective") for row in history[-2:]]
            stalled = (
                len(recent) == 2
                and None not in recent
                and abs(recent[0] - recent[1]) <= 1e-8 * (1.0 + abs(recent[1]))
            )
            # 连续两轮上界不动时，再找几个同值的主问题解，把这些解一起拿去分离。
            # 这些额外解产生的割同样对所有部署有效。
            instance["master_pool_solutions"] = int(instance.get("stall_pool_solutions") or 1) if stalled else 1
            solution = master.optimize()
            while solution["status"] == "FEASIBLE":
                # 时限内有可行解但没证明到间隙。加长本轮时限再解，不把这个方案当成收敛。
                round_time = 1800.0 if round_time is None else float(round_time) * 2.0
                if certifying:
                    tight_time = round_time
                else:
                    base_master_time = round_time
                print(
                    f"主问题在时限内未证明最优，时限加长到 {round_time:.0f} 秒后重解本轮。",
                    flush=True,
                )
                instance["master_time_limit"] = float(round_time)
                solution = master.optimize()
            row = {
                "iteration": iteration,
                "master_status": solution["status"],
                "objective": solution.get("objective"),
                "bound": solution.get("bound"),
                "master_mip_gap_target": round_gap,
                "added": [],
                "accounted_violations_not_written": 0,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            if instance.get("checkpoint_path"):
                print(json.dumps({
                    "iteration": iteration,
                    "master_status": solution["status"],
                    "objective": solution.get("objective"),
                    "bound": solution.get("bound"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "master_seconds": solution.get("runtime"),
                    "node_count": solution.get("node_count"),
                    "stalled": stalled,
                    "master_mip_gap_target": round_gap,
                }, ensure_ascii=False), flush=True)
            _checkpoint(instance, _finish("RUNNING", False, history, solution, master, started, instance))
            if solution["status"] not in ("OPTIMAL", "FEASIBLE"):
                return _finish("MASTER_NOT_OPTIMAL", False, history, solution, master, started, instance)

            # 主问题没有证明到间隙时，分离只核算这个方案，不把模式写进下一轮。
            write_cuts = solution["status"] == "OPTIMAL"
            points = [solution] if write_cuts else [solution]
            if write_cuts:
                points = [solution, *solution.get("alternates", [])]
            batch = int(instance.get("samples_per_iteration") or len(keys))
            cut_cap = int(instance.get("cut_cap") or len(keys))
            order = keys[cursor:] + keys[:cursor]
            pending = []
            accounted = 0
            examined = 0
            active_residual = False
            for event, sample_idx in order:
                # 已经有待加入的割时，看满 batch 个样本就停。还没有割时继续往下看。
                if examined >= batch and pending:
                    break
                if len(pending) >= cut_cap:
                    break
                examined += 1
                for point in points:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return _finish("TIME_LIMIT", False, history, solution, master, started, instance)
                    if remaining is not None:
                        instance["separation_time_limit"] = remaining
                    oracle = solve_separation(point, event, sample_idx, instance)
                    primary = point is solution
                    if primary:
                        row.setdefault("separations", {})[event, sample_idx] = {
                            "status": oracle["status"],
                            "violation": oracle.get("violation"),
                            "pattern": oracle.get("pattern"),
                        }
                    if not oracle.get("certified_optimal"):
                        # 当前最优解的分离没证完，整次运行停止。并列解没证完则跳过该解。
                        if primary:
                            return _finish("SEPARATION_NOT_CERTIFIED", False, history, solution, master, started, instance)
                        continue
                    if oracle["status"] != "FOUND_VIOLATION":
                        continue
                    tolerance = float(instance["separation_tolerance"])
                    for pattern in oracle.get("patterns") or []:
                        pattern = canonical_pattern(pattern, instance)
                        if pattern in master.active[event, sample_idx] or any(
                            item[2] == pattern and item[0] == event and item[1] == sample_idx for item in pending
                        ):
                            if float(oracle["violation"]) <= tolerance:
                                continue
                            # 分离保持默认相对间隙 1e-4。主问题再收紧后，已有割可以
                            # 看起来差零点零几。不超过分离自己的验收范围时继续。
                            room = 1e-4 * (1.0 + abs(float(oracle.get("worst_profit") or 0.0)))
                            if float(oracle["violation"]) <= room:
                                continue
                            # 这条割已经在主问题里，但分离值仍明显低于 eta。
                            # 这是支持函数和分离模型之间的数值差，不是一条可以再加的新割。
                            # 先继续看别的样本；整圈结束仍有残差时，下面会拒绝宣布收敛。
                            if primary:
                                active_residual = True
                            continue
                        if not write_cuts:
                            accounted += 1
                            if primary:
                                print(
                                    f"核算到违背，不写入 {event} 样本 {sample_idx}: {pattern}",
                                    flush=True,
                                )
                            continue
                        pending.append((event, sample_idx, pattern, oracle["violation"]))
                        if len(pending) >= cut_cap:
                            break
            row["accounted_violations_not_written"] = accounted
            cursor = (cursor + examined) % len(keys)
            if not write_cuts:
                return _finish(
                    "MASTER_NOT_OPTIMAL_NO_CUTS",
                    False, history, solution, master, started, instance,
                )
            if pending:
                for event, sample_idx, pattern, _violation in pending:
                    if pattern in master.active[event, sample_idx]:
                        continue
                    if not master.add_pattern(event, sample_idx, pattern):
                        return _finish("PATTERN_ADDITION_FAILURE", False, history, solution, master, started, instance)
                    row["added"].append((event, sample_idx, pattern))
                row["elapsed_seconds"] = time.monotonic() - started
                if adaptive_gap and certifying:
                    certifying = False
                    print("收紧间隙后出现新割，后面的轮次回到较松的间隙。", flush=True)
                _checkpoint(instance, _finish("RUNNING", False, history, solution, master, started, instance))
                continue
            if examined < len(keys):
                # 这一轮因样本上限或割数量上限提前停下，没有看完全部样本，不能宣布收敛。
                continue
            if active_residual:
                return _finish("ACTIVE_PATTERN_RESIDUAL", False, history, solution, master, started, instance)
            if solution["status"] != "OPTIMAL":
                return _finish("MASTER_NOT_OPTIMAL", False, history, solution, master, started, instance)
            if adaptive_gap and not certifying:
                certifying = True
                print(
                    f"没有新的割。下一轮把主问题间隙从 {loose_gap} 收到 {tight_gap}。",
                    flush=True,
                )
                continue
            return _finish("ALL_SEPARATIONS_WITHIN_TOLERANCE", True, history, solution, master, started, instance)
    finally:
        master.dispose()


def _checkpoint(instance: dict, payload: dict) -> None:
    """每轮把当前割和目标写到磁盘。到时停止后，下一次可以从这些割接着算。"""
    path = instance.get("checkpoint_path")
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list), encoding="utf-8")


def _finish(reason: str, converged: bool, history: list, solution: dict | None,
            master: LocalMaster, started: float, instance: dict | None = None) -> dict:
    """整理返回值。converged 只有 reason 为全部样本都在容差内时才是 True。

    reason 的含义：
    - ALL_SEPARATIONS_WITHIN_TOLERANCE：每个样本都证过，没有新的违反。
    - TIME_LIMIT：墙钟用完。目标仍是当前割下的上界。
    - SEPARATION_NOT_CERTIFIED：某个样本的分离没有证明最优。
    - ACTIVE_PATTERN_RESIDUAL：最坏 pattern 已经在主问题里，但 eta 和分离值还有差距。
    - MASTER_NOT_OPTIMAL：主问题自己没有证明最优。
    - MASTER_NOT_OPTIMAL_NO_CUTS：主问题未证明到间隙，分离只核算，不加割。
    - MAXIMUM_ITERATIONS：到达轮数上限。
    - PATTERN_ADDITION_FAILURE：试图加入一条已经存在的割。
    """
    active = {
        f"{event}:{sample_idx}": [list(pattern) for pattern in sorted(patterns)]
        for (event, sample_idx), patterns in master.active.items()
    }
    adaptive_gap = bool((instance or {}).get("adaptive_gap"))
    mip_gap = None if instance is None else instance.get("master_mip_gap")
    return {
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "converged": converged,
        "reason": reason,
        "iterations": len(history),
        "objective": None if solution is None else solution.get("objective"),
        "bound": None if solution is None else solution.get("bound"),
        "elapsed_seconds": time.monotonic() - started,
        "decision": None if solution is None else {
            "n": solution.get("n"),
            "m": solution.get("m"),
            "z": solution.get("z"),
            "flow": solution.get("flow"),
        },
        "active_patterns": active,
        "pattern_count": sum(len(value) for value in active.values()),
        "acceleration": (
            "fixed/mobile capacity lifted; unproven master does not add cuts; "
            "previous integer solution is the next MIP start"
            + (
                "; adaptive master gap 1e-3 until no new cut, then "
                + str(1e-4 if mip_gap is None else mip_gap)
                if adaptive_gap else ""
            )
        ),
        "history": [
            {
                "iteration": row["iteration"],
                "master_status": row["master_status"],
                "objective": row["objective"],
                "bound": row.get("bound"),
                "master_mip_gap_target": row.get("master_mip_gap_target"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "added": row["added"],
                "accounted_violations_not_written": row.get("accounted_violations_not_written", 0),
            }
            for row in history
        ],
    }
```

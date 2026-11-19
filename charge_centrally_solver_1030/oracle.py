import os

import gurobipy as gp
from gurobipy import GRB


def _limit_threads(model) -> None:
    threads = os.environ.get("CG_THREADS")
    if threads:
        model.setParam("Threads", int(threads))

def compute_h_val(a_vec, L, U, mu, Sigma_inv, r, rho, b, I_num):
    """
    辅助函数：计算 h(a) = max_D { (rho + b)*D*a - rho*D }
    用于主程序在第 0 代初始化预置缺电模式时使用。
    """
    model = gp.Model("Compute_h")
    _limit_threads(model)
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
    _limit_threads(model)
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
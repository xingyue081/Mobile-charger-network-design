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

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

# 从我们刚刚创建的两个模块导入函数
from master import solve_master_problem
from oracle import find_worst_pattern, compute_h_val

# instances/ 放 10 区 pickle。福田读 CG_solver/futian/futian_parameters.npz。
# hetero_10zone 是 2026-09-22 的新 10 区；canonical_10zone 是 2026-09-19 的冻结 10 区。
INSTANCE_ROOT = Path(__file__).resolve().parent / "instances"
REAL_INSTANCES = ("hetero_10zone", "canonical_10zone", "futian")
FUTIAN_DIR = Path(__file__).resolve().parents[1] / "futian"
# 主问题用 0,1,2 表示日期类型。这个顺序必须和 solver_interface.EVENTS 一致。
DATE_TYPES = ("workday", "weekend", "holiday")


def build_toy_data():
    """原来的 3 区手写例子。--instance toy 时走这里，用来确认算法本身还能跑。"""
    I_num, J_num, S_num = 3, 3, 3
    demand_data = np.array([
        [320.0, 100.0,  50.0], [300.0, 110.0,  40.0], [350.0,  90.0,  60.0],
        [ 80.0, 340.0, 150.0], [ 60.0, 310.0, 180.0],
        [ 30.0, 150.0, 480.0], [ 40.0, 180.0, 450.0]
    ])
    date_types = [0, 0, 0, 1, 1, 2, 2]

    mu = np.mean(demand_data, axis=0)
    Sigma = np.cov(demand_data, rowvar=False) + np.eye(I_num) * 1e-4
    Sigma_inv = np.linalg.inv(Sigma)

    samples = {0: [], 1: [], 2: []}
    for t, l in enumerate(date_types):
        samples[l].append(demand_data[t])

    epsilon_l = [1.0, 1.5, 2.0]
    r, rho, b = 2.0, 2, 2
    d_ij = np.array([[0, 10, 15], [10, 0, 8], [15, 8, 0]])
    t_ij = 2 * 0.0001 * d_ij

    return {
        'name': 'toy',
        'zone_ids': (0, 1, 2),
        'hub_ids': (0, 1, 2),
        'date_names': DATE_TYPES,
        'I_num': I_num, 'J_num': J_num, 'S_num': S_num,
        'samples': samples, 'q_l': [3/7, 2/7, 2/7], 'epsilon_l': epsilon_l,
        'mu': mu, 'Sigma_inv': Sigma_inv, 'r': r,
        'f': [40, 40, 40], 'c_m': 41, 'c_f': 40, 'rho': rho, 'b': b,
        'tau': 180, 'kappa': 180, 'M_m': [20, 20, 20], 'n_bar': [50, 50, 50],
        'M_z': np.full((I_num, J_num), 20), 't_ij': t_ij,
        'h_val': {}
    }


def load_solver_bundle(instance_name):
    """读出 10 区接口里的 data / params，不重新标定。

    两个 10 区目录里已经有冻结的 solver_inputs.pkl，直接打开。
    """
    instance_dir = INSTANCE_ROOT / instance_name
    pickle_path = instance_dir / "solver_inputs.pkl"
    with pickle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    return bundle["data"], bundle["params"]


def bundle_to_model_data(instance_name, data, params):
    """把 TAZ 编号的 data/params 改成主问题使用的 0..I-1 数组。

    下标 i 对应 zone_ids[i]，下标 j 对应 hub_ids[j]，下标 l 对应 date_names[l]。
    样本只保留 support 内的 demand_hat，顺序沿用 valid_samples。
    每个样本的盒子仍是当前割平面代码的写法：L = max(0, D-epsilon_l)，U = D+epsilon_l。
    """
    zone_ids = tuple(data["I"])
    hub_ids = tuple(data["J"])
    date_names = tuple(data["date_types"])
    if date_names != DATE_TYPES:
        raise ValueError(f"日期类型顺序应为 {DATE_TYPES}，实际是 {date_names}")

    I_num, J_num, S_num = len(zone_ids), len(hub_ids), len(date_names)
    samples = {}
    q_l = []
    epsilon_l = []
    for l, date_name in enumerate(date_names):
        samples[l] = [
            np.array([data["demand_hat"][date_name, local_id, taz] for taz in zone_ids], dtype=float)
            for local_id in data["valid_samples"][date_name]
        ]
        q_l.append(float(data["date_prob"][date_name]))
        epsilon_l.append(float(params["wasserstein_radius"][date_name]))

    mu = np.array([data["mu_hat"][taz] for taz in zone_ids], dtype=float)
    sigma = np.array(
        [[data["sigma_tilde"][left, right] for right in zone_ids] for left in zone_ids],
        dtype=float,
    )
    return {
        'name': instance_name,
        'zone_ids': zone_ids,
        'hub_ids': hub_ids,
        'date_names': date_names,
        'I_num': I_num, 'J_num': J_num, 'S_num': S_num,
        'samples': samples, 'q_l': q_l, 'epsilon_l': epsilon_l,
        'mu': mu, 'Sigma_inv': np.linalg.inv(sigma), 'r': float(params["support_radius"]),
        'f': [params["hub_open_cost"][hub] for hub in hub_ids],
        'c_m': float(params["mobile_unit_cost"]),
        'c_f': float(params["fixed_unit_cost"]),
        'rho': float(params["rho"]),
        'b': float(params["shortage_penalty"]),
        'tau': float(params["fixed_capacity_per_unit"]),
        'kappa': float(params["mobile_capacity_per_unit"]),
        'M_m': [params["mobile_fleet_ub"][hub] for hub in hub_ids],
        'n_bar': [params["fixed_count_ub"][taz] for taz in zone_ids],
        'M_z': np.array(
            [[params["deployment_ub"][taz, hub] for hub in hub_ids] for taz in zone_ids],
            dtype=float,
        ),
        't_ij': np.array(
            [[data["transport_cost"][taz, hub] for hub in hub_ids] for taz in zone_ids],
            dtype=float,
        ),
        'h_val': {},
    }


SHARED_BUSINESS_FIELDS = (
    "rho", "shortage_penalty", "fixed_capacity", "mobile_capacity",
    "fixed_unit_cost", "mobile_unit_cost", "transport_cost_per_distance",
    "hub_open_cost",
)


def load_shared_business(data_dir, overrides=None):
    """读取数据包目录里的 business.json。overrides 里非 None 的项只覆盖这一次。"""
    path = Path(data_dir) / "business.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少共享业务常数：{path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in SHARED_BUSINESS_FIELDS if field not in stored]
    if missing:
        raise ValueError(f"business.json 缺少 {missing}")
    values = {field: float(stored[field]) for field in SHARED_BUSINESS_FIELDS}
    applied = {}
    for field, value in (overrides or {}).items():
        if value is None:
            continue
        if field not in SHARED_BUSINESS_FIELDS:
            raise ValueError(f"不能覆盖业务常数 {field}")
        values[field] = float(value)
        applied[field] = float(value)
    return values, applied, str(path)


def _parameter_npz(data_dir):
    """读取 futian_parameters.npz。错峰实验把平移后的样本仍写成这个文件名。"""
    path = Path(data_dir) / "futian_parameters.npz"
    if path.is_file():
        return path
    raise FileNotFoundError(f"缺少参数文件 futian_parameters.npz：{data_dir}")


def load_futian_data(overrides=None, data_dir=None):
    """把一份福田格式的目录装成割平面已经使用的字典。

    默认目录是 CG_solver/futian。data_dir 只换读哪一份 npz 和 business.json，
    不改主问题、分离或运输成本的写法。错峰实验用它读平移后的样本。
    样本、均值、协方差、半径、epsilon 和距离来自 npz。
    rho、b、tau、kappa、c^f、c^m、gamma、hub 开设成本来自同一目录的 business.json。
    往返系数 2 只属于集中补能。
    每个入选 TAZ 同时是一个候选 hub。
    """
    root = FUTIAN_DIR if data_dir is None else Path(data_dir)
    stored_path = _parameter_npz(root)
    with np.load(stored_path, allow_pickle=False) as stored:
        zone_ids = tuple(int(value) for value in stored["taz_ids"].tolist())
        distance = stored["distance"].astype(float)
        n_bar = stored["n_bar"].astype(float)
        mu = stored["mu"].astype(float)
        sigma_inv = stored["Sigma_inv"].astype(float)
        sample_rows = (
            stored["samples_workday"],
            stored["samples_weekend"],
            stored["samples_holiday"],
        )
        q_l = stored["q_l"].astype(float)
        epsilon_l = stored["epsilon_l"].astype(float)
        radius = float(stored["r"])
    zone_count = len(zone_ids)
    if distance.shape != (zone_count, zone_count) or mu.shape != (zone_count,):
        raise ValueError("数据包维度与 taz_ids 不一致")
    if q_l.shape != (3,) or epsilon_l.shape != (3,):
        raise ValueError("日期类型参数必须是 3 维")
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("距离矩阵必须有限且非负")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise ValueError("距离矩阵必须对称且对角线为 0")
    business, business_overrides, business_path = load_shared_business(root, overrides)
    hub_open_cost = business["hub_open_cost"]
    fleet_per_hub = 30.0
    samples = {
        date_index: [row.copy() for row in rows]
        for date_index, rows in enumerate(sample_rows)
    }
    return {
        "name": "futian",
        "zone_ids": zone_ids,
        "hub_ids": zone_ids,
        "date_names": DATE_TYPES,
        "I_num": zone_count,
        "J_num": zone_count,
        "S_num": len(DATE_TYPES),
        "samples": samples,
        "q_l": q_l.tolist(),
        "epsilon_l": epsilon_l.tolist(),
        "mu": mu,
        "Sigma_inv": sigma_inv,
        "r": radius,
        "f": [hub_open_cost] * zone_count,
        "c_m": business["mobile_unit_cost"],
        "c_f": business["fixed_unit_cost"],
        "rho": business["rho"],
        "b": business["shortage_penalty"],
        "tau": business["fixed_capacity"],
        "kappa": business["mobile_capacity"],
        "M_m": [fleet_per_hub] * zone_count,
        "n_bar": n_bar,
        "M_z": np.full((zone_count, zone_count), fleet_per_hub),
        "gamma": business["transport_cost_per_distance"],
        "t_ij": 2.0 * business["transport_cost_per_distance"] * distance,
        "business_path": business_path,
        "business_overrides": business_overrides,
        "h_val": {},
    }


def load_model_data(instance_name, business_overrides=None, data_dir=None):
    """--instance 的唯一入口。toy 用手写数据，futian 读 npz，10 区读 pickle。

    data_dir 只对 futian 有意义。不传时读 CG_solver/futian。
    """
    if instance_name == "toy":
        return build_toy_data()
    if instance_name == "futian":
        model_data = load_futian_data(business_overrides, data_dir)
        sample_counts = {
            model_data["date_names"][l]: len(model_data["samples"][l])
            for l in range(model_data["S_num"])
        }
        print(
            f"已载入 {model_data['name']}: 区域 {model_data['zone_ids']}，"
            f"support 内样本 {sample_counts}"
        )
        if model_data["business_overrides"]:
            print(f"本次覆盖的业务常数: {model_data['business_overrides']}")
        return model_data
    if instance_name not in REAL_INSTANCES:
        raise ValueError(f"未知实例 {instance_name}，可选 toy、{', '.join(REAL_INSTANCES)}")
    data, params = load_solver_bundle(instance_name)
    model_data = bundle_to_model_data(instance_name, data, params)
    sample_counts = {model_data["date_names"][l]: len(model_data["samples"][l]) for l in range(model_data["S_num"])}
    print(
        f"已载入 {instance_name}: 区域 {model_data['zone_ids']}，"
        f"support 内样本 {sample_counts}"
    )
    return model_data


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
    # - adaptive_gap：加割轮次间隙 0.005 或 master_time_limit（默认 600 秒）先到就停，
    #   有可行解仍写入割。没有新割后的那一轮才用 0.001，并且到时限不再翻倍。
    # - cut_tolerance：eta 比分离值高出这么多才算违反、才加割；
    # - max_iterations / max_total_seconds：限制整个割平面循环。
    started = perf_counter()
    iteration = 1
    master_solves = 0
    last_decisions = None
    stop_reason = None
    pattern_order = []
    pattern_seen = set()
    pattern_rounds = []
    # 自适应间隙：加割的轮次用较松的 MIPGap。没有新割后，下一轮才收到较紧的间隙。
    certifying = False
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
            limits['max_iterations'] *= 2
            print(f"\n达到最大迭代次数，上限加长到 {limits['max_iterations']} 后继续。")

        print(f"\n{'='*40}\n 开始第 {iteration} 轮迭代求解主模型...\n{'='*40}")
        if limits.get("adaptive_gap"):
            master_mip_gap = (
                limits["master_mip_gap_tight"] if certifying else limits["master_mip_gap_loose"]
            )
        else:
            master_mip_gap = limits["master_mip_gap"]
        
        # 3.1 求解主模型。加割轮次用 0.005 和 master_time_limit，到点不翻倍。
        # 没有新割后的那一轮用 0.001 和 master_time_limit_tight，到点也不翻倍。
        master_time_limit = limits['master_time_limit']
        if certifying and limits.get('master_time_limit_tight') is not None:
            master_time_limit = limits['master_time_limit_tight']
        if limits['max_total_seconds'] is not None:
            remaining = limits['max_total_seconds'] - (perf_counter() - started)
            master_time_limit = remaining if master_time_limit is None else min(master_time_limit, remaining)
        extend_on_time_limit = not bool(limits.get("adaptive_gap"))
        print(
            f"本轮主问题间隙目标 {master_mip_gap}，时限 {master_time_limit} 秒"
            f"{'，到时限不延长' if not extend_on_time_limit else ''}。"
            + (" 本轮收紧到 0.001。" if certifying else ""),
            flush=True,
        )
        gurobi_log = None
        if limits.get('gurobi_log_dir') is not None:
            gurobi_log = Path(limits['gurobi_log_dir']) / f"round_{iteration:03d}.log"
            gurobi_log.parent.mkdir(parents=True, exist_ok=True)
        is_solved, C_opt, eta_opt, T_opt, decisions = \
            solve_master_problem(
                active_patterns, toy_data, {
                    'mip_gap': master_mip_gap,
                    'mip_gap_abs': limits.get('master_mip_gap_abs'),
                    'optimality_tol': limits.get('optimality_tol'),
                    'time_limit': master_time_limit,
                    'extend_on_time_limit': extend_on_time_limit,
                    'output_flag': limits.get('master_log'),
                },
                # last_decisions 此时还是上一轮的解。这一轮返回后才会被盖掉。
                start_solution=last_decisions,
                log_file=gurobi_log,
            )
        
        if not is_solved:
            stop_reason = "主问题没有可行解"
            print(f"❌ 主模型求解失败或无可行解（{decisions.get('status_name')}），算法终止。")
            break
        print(
            f"主问题状态 {decisions['status_name']}，"
            f"目标 {decisions['obj']:.6g}，bound {decisions['obj_bound']:.6g}，"
            f"MIP gap {decisions['mip_gap']:.3g}（本轮要求 {master_mip_gap}），"
            f"用时 {decisions['runtime']:.1f} 秒，"
            f"节点 {decisions['node_count']}"
        )
        master_solves += 1
        last_decisions = decisions
        used_limit = decisions.get("time_limit")
        if used_limit is not None:
            master_time_limit = float(used_limit)
            if certifying:
                limits['master_time_limit_tight'] = master_time_limit
            else:
                limits['master_time_limit'] = master_time_limit
        # 加割轮次时限先到时，当前可行解仍然分离并写入割。
        # 0.001 那一轮若只是时限停止，不把未证明的方案写成新割，也不宣布收敛。
        cut_round_time_stop = (
            bool(limits.get("adaptive_gap"))
            and not certifying
            and decisions["status_name"] == "TIME_LIMIT"
        )
        if decisions['status_name'] != "OPTIMAL":
            if cut_round_time_stop:
                print("这一轮加割时限已到，用当前可行解分离并写入割。")
            else:
                print("这一轮主问题未证明最优。后面的分离用的是当前可行解。")
            
        cuts_added = 0
        accounted_violations = 0
        separation_finished = True
        active_residual = False
        # 没证明到间隙时，默认只核算、不写入。加割轮次的 10 分钟停止是例外。
        write_cuts = decisions['status_name'] == "OPTIMAL" or cut_round_time_stop
        separation_started = perf_counter()
        accounted_operating = 0.0
        
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

                sample_weight = toy_data['q_l'][l] / len(samples[l])
                accounted_operating += sample_weight * (true_worst_profit - T_opt[l])
                
                # 3.3 判断是否违反割平面约束。已经在池里的 pattern 不再写入，也不改 h。
                if eta_opt[l, n_idx] > true_worst_profit - T_opt[l] + limits['cut_tolerance']:
                    if a_worst not in active_patterns[(l, n_idx)]:
                        if not write_cuts:
                            accounted_violations += 1
                            print(
                                f"  [核算到违背，不写入] 情形 {l} 样本 {n_idx}: 模式 {a_worst}"
                            )
                            continue
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
                        
        separation_seconds = perf_counter() - separation_started
        investment = (
            sum(toy_data['f'][j] * decisions['x'][j] for j in range(J_num))
            + toy_data['c_m'] * sum(decisions['m'].values())
            + toy_data['c_f'] * sum(decisions['n'].values())
        )
        print(f"--> 本轮迭代共新增 {cuts_added} 条割平面。")
        round_record = {
            "iteration": iteration,
            "dataset": toy_data.get("name"),
            "shortage_penalty": toy_data.get("b"),
            "wall_seconds": perf_counter() - started,
            "master_status": decisions["status_name"],
            "master_obj": decisions["obj"],
            "master_bound": decisions["obj_bound"],
            "master_mip_gap_target": master_mip_gap,
            "master_mip_gap": decisions["mip_gap"],
            "master_seconds": decisions["runtime"],
            "master_nodes": decisions["node_count"],
            "master_simplex_iters": decisions.get("iter_count"),
            "num_vars": decisions.get("num_vars"),
            "num_constrs": decisions.get("num_constrs"),
            "num_nzs": decisions.get("num_nzs"),
            "warm_start": decisions.get("warm_start"),
            "first_incumbent_seconds": decisions.get("first_incumbent_seconds"),
            "first_incumbent_obj": decisions.get("first_incumbent_obj"),
            "cuts_added": cuts_added,
            "cuts_written": write_cuts,
            "accounted_violations_not_written": accounted_violations,
            "unique_patterns": len(pattern_order),
            "separation_seconds": separation_seconds,
            "separation_finished": separation_finished,
            "accounted_objective": None if not separation_finished else -investment + accounted_operating,
            "gurobi_log": None if gurobi_log is None else str(gurobi_log),
        }
        pattern_rounds.append(round_record)
        if limits.get("round_log") is not None:
            round_log = Path(limits["round_log"])
            round_log.parent.mkdir(parents=True, exist_ok=True)
            with round_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(round_record, ensure_ascii=False) + "\n")
        print(
            f"PATTERN_ROUND iteration={iteration} cuts_added={cuts_added} "
            f"unique={len(pattern_order)} master={decisions['status_name']} "
            f"obj={decisions['obj']:.6g} gap_target={master_mip_gap} "
            f"master_s={decisions['runtime']:.4f} "
            f"nodes={decisions['node_count']} sep_s={separation_seconds:.4f} "
            f"nzs={decisions.get('num_nzs')} warm={int(bool(decisions.get('warm_start')))} "
            f"first_incumbent_s={decisions.get('first_incumbent_seconds')}",
            flush=True,
        )
        
        # 3.4 收敛检查。分离被时限打断时，不能把“本轮没加完的割”当成收敛。
        if not write_cuts and separation_finished:
            stop_reason = "主问题未证明到间隙，分离只核算当前方案，不加割"
            print(f"\n{stop_reason}。本轮若写入会新增 {accounted_violations} 条割。")
            if round_record["accounted_objective"] is not None:
                print(f"按分离核算的方案利润: {round_record['accounted_objective']:.6g}")
            break
        if not separation_finished:
            print(f"\n分离未完成：{stop_reason}")
            break
        if (
            limits.get("adaptive_gap")
            and cuts_added == 0
            and separation_finished
            and write_cuts
            and not active_residual
            and not certifying
        ):
            certifying = True
            print(
                f"\n没有新的割。下一轮把主问题间隙从 {limits['master_mip_gap_loose']} "
                f"收到 {limits['master_mip_gap_tight']}。"
            )
            iteration += 1
            continue
        if limits.get("adaptive_gap") and cuts_added > 0 and certifying:
            certifying = False
            print("\n收紧间隙后出现新割，后面的轮次回到较松的间隙。")
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

def _jsonable_patterns(active) -> dict | None:
    """样本割记成 \"日期类型:样本\" -> pattern 列表，避免元组键写不进 JSON。"""
    if not active:
        return None
    return {
        f"{date_type}:{sample_idx}": [list(pattern) for pattern in patterns]
        for (date_type, sample_idx), patterns in active.items()
    }


def _jsonable_plan(plan: dict | None) -> dict | None:
    """把 x/y/m/n/z 的整数或元组键改成 JSON 可写的字符串键。

    y 写成 \"i,j\"，z 写成 \"i,j,l\"，与绘图脚本读取格式一致。
    """
    if not plan or "x" not in plan:
        return None
    return {
        "x": {str(key): int(value) for key, value in plan["x"].items()},
        "m": {str(key): int(value) for key, value in plan["m"].items()},
        "n": {str(key): int(value) for key, value in plan["n"].items()},
        "y": {
            f"{i},{j}": int(value)
            for (i, j), value in plan["y"].items()
        },
        "z": {
            f"{i},{j},{l}": int(value)
            for (i, j, l), value in plan["z"].items()
        },
    }


def _result_summary(result: dict, zone_ids=None) -> dict:
    """从割平面返回值里取出实验表需要的数，并保留完整调度决策。

    m* 是各枢纽移动桩数量之和。枢纽个数只数 x_j = 1 的枢纽。
    固定桩总数是各 zone 的 n_i 之和。
    decision 里是可直接画图的 x/y/m/n/z，不再只留汇总数。
    """
    plan = result.get("plan") or {}
    fixed_counts = plan.get("n") or {}
    mobile_counts = plan.get("m") or {}
    hub_open = plan.get("x") or {}
    open_hubs = [int(hub) for hub, opened in hub_open.items() if int(opened) == 1]
    summary = {
        "objective": result.get("objective"),
        "bound": result.get("bound"),
        "mip_gap": result.get("mip_gap"),
        "iterations": result.get("iterations"),
        "stop_reason": result.get("stop_reason"),
        "status_name": result.get("status_name"),
        "fixed_chargers": None if not fixed_counts else int(sum(fixed_counts.values())),
        "mobile_chargers": None if not mobile_counts else int(sum(mobile_counts.values())),
        "hub_count": len(open_hubs),
        "hub_indexes": open_hubs,
        "zone_ids": None if zone_ids is None else [int(value) for value in zone_ids],
        "decision": _jsonable_plan(plan),
        "active_patterns": _jsonable_patterns(result.get("active_patterns")),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用指定实例运行割平面算法")
    parser.add_argument(
        "--instance",
        default="toy",
        choices=("toy",) + REAL_INSTANCES,
        help="toy=手写 3 区；hetero_10zone=新 10 区；canonical_10zone=冻结 10 区；futian=福田 npz",
    )
    parser.add_argument("--master-mip-gap", type=float, default=None,
                        help="每一轮主问题的 Gurobi 相对 MIP gap。不写则用 Gurobi 默认 1e-4")
    parser.add_argument("--adaptive-gap", action="store_true",
                        help="加割轮次主问题间隙 0.005，或 master-time-limit（默认 600 秒）先到就停，并继续加割。没有新割后，下一轮间隙 0.001，时限 master-time-limit-tight")
    parser.add_argument("--master-mip-gap-abs", type=float, default=None,
                        help="每一轮主问题的绝对空隙。达到后 Gurobi 停。不写则用默认 1e-10")
    parser.add_argument("--master-time-limit", type=float, default=600.0,
                        help="每一轮主问题最多算多少秒，默认 600")
    parser.add_argument("--master-time-limit-tight", type=float, default=14400.0,
                        help="adaptive-gap 收到 0.001 的那一轮主问题最多算多少秒，默认 14400。到点不翻倍")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="割平面最多迭代多少轮，默认 100")
    parser.add_argument("--max-total-seconds", type=float, default=None,
                        help="整个算法的墙钟上限（秒）。不写则不限制总时间")
    parser.add_argument("--cut-tolerance", type=float, default=1e-4,
                        help="eta 超过分离值多少才加割，默认 1e-4")
    parser.add_argument("--separation-mip-gap", type=float, default=None,
                        help="分离子问题的相对 MIP gap。不写则证明到 Gurobi 默认精度")
    parser.add_argument("--master-log", action="store_true",
                        help="打开主问题的 Gurobi 节点日志。不写则主问题不打印节点")
    parser.add_argument("--rho", type=float, default=None, help="覆盖 business.json 的 rho，只作用于本次")
    parser.add_argument("--shortage-penalty", type=float, default=None, help="覆盖 business.json 的 shortage_penalty，只作用于本次")
    parser.add_argument("--tau", type=float, default=None, help="覆盖 business.json 的 fixed_capacity，只作用于本次")
    parser.add_argument("--kappa", type=float, default=None, help="覆盖 business.json 的 mobile_capacity，只作用于本次")
    parser.add_argument("--fixed-cost", type=float, default=None, help="覆盖 business.json 的 fixed_unit_cost，只作用于本次")
    parser.add_argument("--mobile-cost", type=float, default=None, help="覆盖 business.json 的 mobile_unit_cost，只作用于本次")
    parser.add_argument("--gamma", type=float, default=None, help="覆盖 business.json 的 transport_cost_per_distance，只作用于本次")
    parser.add_argument("--hub-open-cost", type=float, default=None, help="覆盖 business.json 的 hub_open_cost，只作用于本次")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="只对 futian 有效。目录里要有 futian_parameters.npz 和 business.json。不写则用 CG_solver/futian")
    parser.add_argument("--output", type=Path, default=None,
                        help="把目标、完整调度决策(x/y/m/n/z)、固定/移动桩汇总数写入 JSON。不写则只在终端打印")
    args = parser.parse_args()
    if args.master_mip_gap is not None and args.master_mip_gap < 0:
        parser.error("master-mip-gap 不能为负")
    if args.adaptive_gap and args.master_mip_gap is not None and args.master_mip_gap > 5e-3:
        parser.error("adaptive-gap 的加割间隙是 0.005，master-mip-gap 必须更紧或等于 0.005")
    if args.master_mip_gap_abs is not None and args.master_mip_gap_abs < 0:
        parser.error("master-mip-gap-abs 不能为负")
    if args.master_time_limit <= 0 or args.master_time_limit_tight <= 0 or args.max_iterations <= 0:
        parser.error("master-time-limit、master-time-limit-tight 和 max-iterations 必须为正")
    if args.max_total_seconds is not None and args.max_total_seconds <= 0:
        parser.error("max-total-seconds 必须为正")
    if args.cut_tolerance < 0:
        parser.error("cut-tolerance 不能为负")
    if args.separation_mip_gap is not None and args.separation_mip_gap < 0:
        parser.error("separation-mip-gap 不能为负")
    business_overrides = {
        "rho": args.rho,
        "shortage_penalty": args.shortage_penalty,
        "fixed_capacity": args.tau,
        "mobile_capacity": args.kappa,
        "fixed_unit_cost": args.fixed_cost,
        "mobile_unit_cost": args.mobile_cost,
        "transport_cost_per_distance": args.gamma,
        "hub_open_cost": args.hub_open_cost,
    }
    if args.hub_open_cost is not None and args.hub_open_cost < 0:
        parser.error("hub-open-cost 不能为负")
    if any(value is not None for value in business_overrides.values()) and args.instance != "futian":
        parser.error("业务常数覆盖只对 --instance futian 生效")
    if args.data_dir is not None and args.instance != "futian":
        parser.error("data-dir 只对 --instance futian 生效")
    model_data = load_model_data(args.instance, business_overrides, args.data_dir)
    round_log = None
    gurobi_log_dir = None
    if args.output is not None:
        round_log = args.output.with_name(args.output.stem + "_rounds.jsonl")
        gurobi_log_dir = args.output.with_name(args.output.stem + "_gurobi")
        if round_log.exists():
            round_log.unlink()
    result = run_cutting_plane_algorithm(model_data, {
        'master_mip_gap': args.master_mip_gap,
        'adaptive_gap': args.adaptive_gap,
        'master_mip_gap_loose': 5e-3,
        'master_mip_gap_tight': 1e-3 if args.master_mip_gap is None else args.master_mip_gap,
        'master_mip_gap_abs': args.master_mip_gap_abs,
        'master_time_limit': args.master_time_limit,
        'master_time_limit_tight': args.master_time_limit_tight if args.adaptive_gap else None,
        'max_iterations': args.max_iterations,
        'max_total_seconds': args.max_total_seconds,
        'cut_tolerance': args.cut_tolerance,
        'separation_mip_gap': args.separation_mip_gap,
        'master_log': args.master_log,
        'round_log': round_log,
        'gurobi_log_dir': gurobi_log_dir,
    })
    if args.output is not None:
        # 汇总数之外，把完整建站与调度决策一并写入，供画图直接读取。
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = _result_summary(result, model_data.get("zone_ids"))
        summary["dataset"] = model_data["name"]
        summary["shortage_penalty"] = model_data["b"]
        if "c_m" in model_data and "gamma" in model_data:
            summary["parameters"] = {
                "rho": model_data["rho"],
                "shortage_penalty": model_data["b"],
                "fixed_unit_cost": model_data["c_f"],
                "mobile_unit_cost": model_data["c_m"],
                "gamma": model_data["gamma"],
                "hub_open_cost": float(model_data["f"][0]),
            }
        summary["round_log"] = None if round_log is None else str(round_log.resolve())
        summary["gurobi_log_dir"] = None if gurobi_log_dir is None else str(gurobi_log_dir.resolve())
        summary["reformulation"] = (
            "transport and inflow lifted; unproven master does not add cuts; "
            "previous integer solution is the next MIP start; "
            "adaptive master gap 0.005 or master time limit during cut rounds, then "
            + str(1e-3 if args.master_mip_gap is None else args.master_mip_gap)
            if args.adaptive_gap else
            "transport and inflow lifted; unproven master does not add cuts; previous integer solution is the next MIP start"
        )
        summary["separation_seconds"] = result.get("separation_seconds")
        summary["elapsed_seconds"] = result.get("elapsed_seconds")
        summary["node_count"] = result.get("node_count")
        summary["num_nzs"] = result.get("num_nzs")
        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"结果已写入 {output}")
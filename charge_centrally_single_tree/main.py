import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

# 从我们刚刚创建的两个模块导入函数
from master import SingleTreeMaster
from oracle import compute_h_val

# instances/ 放 10 区 pickle。行政区读 CG_solver/<name>/<name>_parameters.npz。
# hetero_10zone 是 2026-09-22 的新 10 区；canonical_10zone 是 2026-09-19 的冻结 10 区。
INSTANCE_ROOT = Path(__file__).resolve().parent / "instances"
_CG_ROOT = Path(__file__).resolve().parents[1]
REAL_INSTANCES = ("hetero_10zone", "canonical_10zone", "futian", "luohu", "guangming", "pingshan")
FUTIAN_DIR = _CG_ROOT / "futian"
DISTRICT_DIRS = {
    "futian": FUTIAN_DIR,
    "luohu": _CG_ROOT / "luohu",
    "guangming": _CG_ROOT / "guangming",
    "pingshan": _CG_ROOT / "pingshan",
}
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


def _parameter_npz(data_dir, instance_name):
    """读取 <instance>_parameters.npz。错峰实验把平移后的样本仍写成这个文件名。"""
    filename = f"{instance_name}_parameters.npz"
    path = Path(data_dir) / filename
    if path.is_file():
        return path
    raise FileNotFoundError(f"缺少参数文件 {filename}：{data_dir}")


def load_district_data(instance_name, overrides=None, data_dir=None):
    """把一份福田格式的目录装成割平面已经使用的字典。

    默认目录是 CG_solver/<instance_name>。data_dir 只换读哪一份 npz 和 business.json，
    不改主问题、分离或运输成本的写法。错峰实验用它读平移后的样本。
    样本、均值、协方差、半径、epsilon 和距离来自 npz。
    rho、b、tau、kappa、c^f、c^m、gamma、hub 开设成本来自同一目录的 business.json。
    往返系数 2 只属于集中补能。
    每个入选 TAZ 同时是一个候选 hub。
    """
    if instance_name not in DISTRICT_DIRS:
        raise ValueError(f"未知行政区实例 {instance_name}")
    root = DISTRICT_DIRS[instance_name] if data_dir is None else Path(data_dir)
    stored_path = _parameter_npz(root, instance_name)
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
        "name": instance_name,
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


def load_futian_data(overrides=None, data_dir=None):
    """保留旧入口。新代码走 load_district_data。"""
    return load_district_data("futian", overrides, data_dir)


def load_model_data(instance_name, business_overrides=None, data_dir=None):
    """--instance 的唯一入口。toy 用手写数据，行政区实例读 npz，10 区读 pickle。

    data_dir 只对 futian、luohu、guangming、pingshan 有意义。
    不传时读 CG_solver 下同名目录。
    """
    if instance_name == "toy":
        return build_toy_data()
    if instance_name in DISTRICT_DIRS:
        model_data = load_district_data(instance_name, business_overrides, data_dir)
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
    """Single-tree branch-and-cut with lazy pattern cuts.

    Flow:
    1. Build one MIP with initial all-0 / all-1 cuts.
    2. Optimize with a MIPSOL callback that adds violated patterns as lazy cuts
       using the alternating heuristic (no nested MIQCP in the callback).
    3. After the tree stops, run exact MIQCP certification on the incumbent.
       If new cuts appear, add them as user cuts and re-optimize the same model.
    4. Optional adaptive gap: generate with a loose MIPGap, then certify with a
       tight MIPGap once exact separation finds nothing.
    """
    I_num, J_num, S_num = toy_data['I_num'], toy_data['J_num'], toy_data['S_num']
    samples = toy_data['samples']
    epsilon_l = toy_data['epsilon_l']
    mu = toy_data['mu']
    Sigma_inv = toy_data['Sigma_inv']
    r, rho, b = toy_data['r'], toy_data['rho'], toy_data['b']

    active_patterns = {}
    for l in range(S_num):
        eps = epsilon_l[l]
        for n_idx, D_sample in enumerate(samples[l]):
            L = np.maximum(0, D_sample - eps)
            U = D_sample + eps
            p0, p1 = tuple([0] * I_num), tuple([1] * I_num)
            active_patterns[(l, n_idx)] = [p0, p1]
            toy_data['h_val'][(l, n_idx, p0)] = compute_h_val(
                p0, L, U, mu, Sigma_inv, r, rho, b, I_num
            )
            toy_data['h_val'][(l, n_idx, p1)] = compute_h_val(
                p1, L, U, mu, Sigma_inv, r, rho, b, I_num
            )

    started = perf_counter()
    iteration = 1
    master_solves = 0
    last_decisions = None
    stop_reason = None
    pattern_order = []
    pattern_seen = set()
    pattern_rounds = []
    certifying = False

    cut_cap = limits.get("cut_cap")
    cut_tolerance = float(limits.get("cut_tolerance", 1e-4))
    for patterns in active_patterns.values():
        for pattern in patterns:
            if pattern not in pattern_seen:
                pattern_seen.add(pattern)
                pattern_order.append(pattern)

    master = SingleTreeMaster(
        toy_data,
        active_patterns,
        cut_tolerance=cut_tolerance,
        cut_cap=cut_cap,
    )
    print(
        f"单树主问题已建立：初始割 {sum(len(v) for v in active_patterns.values())} 条，"
        f"LazyConstraints=1，callback cut_cap={cut_cap}",
        flush=True,
    )

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

        print(f"\n{'='*40}\n 开始第 {iteration} 轮单树求解...\n{'='*40}")
        if limits.get("adaptive_gap"):
            master_mip_gap = (
                limits["master_mip_gap_tight"] if certifying else limits["master_mip_gap_loose"]
            )
        else:
            master_mip_gap = limits["master_mip_gap"]

        master_time_limit = limits['master_time_limit']
        if certifying and limits.get('master_time_limit_tight') is not None:
            master_time_limit = limits['master_time_limit_tight']
        if limits['max_total_seconds'] is not None:
            remaining = limits['max_total_seconds'] - (perf_counter() - started)
            master_time_limit = remaining if master_time_limit is None else min(master_time_limit, remaining)

        gurobi_log = None
        if limits.get('gurobi_log_dir') is not None:
            gurobi_log = Path(limits['gurobi_log_dir']) / f"round_{iteration:03d}.log"
            gurobi_log.parent.mkdir(parents=True, exist_ok=True)

        lazy_before = master.stats["lazy_cuts_added"]
        is_solved, C_opt, eta_opt, T_opt, decisions = master.optimize(
            {
                'mip_gap': master_mip_gap,
                'mip_gap_abs': limits.get('master_mip_gap_abs'),
                'optimality_tol': limits.get('optimality_tol'),
                'time_limit': master_time_limit,
                'output_flag': limits.get('master_log'),
            },
            start_solution=last_decisions,
            log_file=gurobi_log,
        )
        lazy_this_round = master.stats["lazy_cuts_added"] - lazy_before

        if not is_solved:
            stop_reason = "主问题没有可行解"
            print(f"❌ 主模型求解失败或无可行解（{decisions.get('status_name')}），算法终止。")
            break

        print(
            f"主问题状态 {decisions['status_name']}，"
            f"目标 {decisions['obj']:.6g}，bound {decisions['obj_bound']:.6g}，"
            f"MIP gap {decisions['mip_gap']:.3g}（本轮要求 {master_mip_gap}），"
            f"用时 {decisions['runtime']:.1f} 秒，"
            f"节点 {decisions['node_count']}，"
            f"本轮 lazy 割 {lazy_this_round}",
            flush=True,
        )
        master_solves += 1
        last_decisions = decisions

        # Refresh pattern_order from active pool.
        for patterns in active_patterns.values():
            for pattern in patterns:
                if pattern not in pattern_seen:
                    pattern_seen.add(pattern)
                    pattern_order.append(pattern)

        # Exact certification of the incumbent (outside the callback).
        remaining = None
        if limits['max_total_seconds'] is not None:
            remaining = limits['max_total_seconds'] - (perf_counter() - started)
            if remaining <= 0:
                stop_reason = "达到总时间上限"
                print("\n认证前已达到总时间上限，停止。")
                break

        certify_started = perf_counter()
        try:
            exact_cuts, accounted_operating, exact_details = master.exact_certify(
                C_opt, eta_opt, T_opt,
                mip_gap=limits.get('separation_mip_gap'),
                time_limit=remaining,
                cut_cap=cut_cap,
            )
            separation_finished = True
        except RuntimeError as exc:
            stop_reason = "分离未在剩余时间内达到设定 gap"
            exact_cuts, accounted_operating, exact_details = 0, None, []
            separation_finished = False
            print(f"\n{exc}")

        for patterns in active_patterns.values():
            for pattern in patterns:
                if pattern not in pattern_seen:
                    pattern_seen.add(pattern)
                    pattern_order.append(pattern)

        separation_seconds = perf_counter() - certify_started
        investment = (
            sum(toy_data['f'][j] * decisions['x'][j] for j in range(J_num))
            + toy_data['c_m'] * sum(decisions['m'].values())
            + toy_data['c_f'] * sum(decisions['n'].values())
        )
        print(
            f"--> 本轮 lazy 割 {lazy_this_round}，精确认证新割 {exact_cuts}，"
            f"callback 累计 {master.stats['callback_calls']} 次，"
            f"lazy 累计 {master.stats['lazy_cuts_added']} 条。",
            flush=True,
        )
        for detail in exact_details[:20]:
            print(
                f"  [精确认证割] 情形 {detail['sample'][0]} 样本 {detail['sample'][1]}: "
                f"模式 {detail['pattern']}，违反 {detail['violation']:.6g}",
                flush=True,
            )

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
            "lazy_cuts_this_round": lazy_this_round,
            "exact_cuts_this_round": exact_cuts,
            "callback_calls": master.stats["callback_calls"],
            "callback_seconds": master.stats["callback_seconds"],
            "unique_patterns": len(pattern_order),
            "separation_seconds": separation_seconds,
            "separation_finished": separation_finished,
            "accounted_objective": None if accounted_operating is None else -investment + accounted_operating,
            "gurobi_log": None if gurobi_log is None else str(gurobi_log),
            "certifying": certifying,
            "method": "single_tree_lazy",
        }
        pattern_rounds.append(round_record)
        if limits.get("round_log") is not None:
            round_log = Path(limits["round_log"])
            round_log.parent.mkdir(parents=True, exist_ok=True)
            with round_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(round_record, ensure_ascii=False) + "\n")
        print(
            f"PATTERN_ROUND iteration={iteration} lazy={lazy_this_round} exact={exact_cuts} "
            f"unique={len(pattern_order)} master={decisions['status_name']} "
            f"obj={decisions['obj']:.6g} gap_target={master_mip_gap} "
            f"master_s={decisions['runtime']:.4f} nodes={decisions['node_count']} "
            f"sep_s={separation_seconds:.4f}",
            flush=True,
        )

        if not separation_finished:
            print(f"\n分离未完成：{stop_reason}")
            break

        # New exact cuts: re-optimize the same model (tree restarts, but rare).
        if exact_cuts > 0:
            if certifying:
                certifying = False
                print("\n认证间隙下仍出现精确割，回到较松间隙继续单树求解。", flush=True)
            iteration += 1
            continue

        # No exact cuts on the incumbent.
        if (
            limits.get("adaptive_gap")
            and not certifying
            and decisions['status_name'] == "OPTIMAL"
        ):
            certifying = True
            print(
                f"\n精确认证无新割。下一轮把主问题间隙从 {limits['master_mip_gap_loose']} "
                f"收到 {limits['master_mip_gap_tight']}。",
                flush=True,
            )
            iteration += 1
            continue

        if decisions['status_name'] == "OPTIMAL":
            stop_reason = "没有新的割，主问题相对间隙已达到 Gurobi 接受的范围"
            print(f"\n没有新的割。主问题相对间隙为 {decisions['mip_gap']:.6g}。", flush=True)
        else:
            # TIME_LIMIT / 提前停：若总时长还有余量，继续同一棵树上搜，不要当成收敛退出。
            remaining_budget = None
            if limits['max_total_seconds'] is not None:
                remaining_budget = limits['max_total_seconds'] - (perf_counter() - started)
            if remaining_budget is None or remaining_budget > 1.0:
                print(
                    f"\n精确认证无新割，但主问题状态是 {decisions['status_name']}，"
                    f"继续单树搜索（剩余总时长 "
                    f"{'不限' if remaining_budget is None else f'{remaining_budget:.1f}s'}）。",
                    flush=True,
                )
                iteration += 1
                continue
            stop_reason = "没有新的割，但主问题只是提前停止"
            print("\n没有新的割，但主问题是提前停止的，不能当成严格收敛。", flush=True)

        # Lazy-constraint ObjVal can be slightly optimistic numerically.
        # Prefer the exact separation-accounted objective of the incumbent.
        if accounted_operating is not None:
            accounted_obj = -investment + accounted_operating
            decisions['obj_master_raw'] = decisions['obj']
            decisions['obj'] = accounted_obj
            last_decisions = decisions
            print(
                f"采用精确认证核算目标 {accounted_obj:.12g} "
                f"（主问题原始 ObjVal {decisions['obj_master_raw']:.12g}）",
                flush=True,
            )

        print("\n" + "=" * 50)
        print(f"总目标利润: {decisions['obj']}")
        print("最终容量规划 (C_opt)")
        print("=" * 50)
        for l in range(S_num):
            res = [f"区域 {i}: {C_opt[l, i]:.2f}" for i in range(I_num)]
            print(f"▶ 情形 {l}:  " + " | ".join(res))
        print("\n" + "=" * 50)
        print("最终建站决策")
        print("=" * 50)
        for j in range(J_num):
            if decisions['x'][j] == 1:
                print(f"枢纽 {j}: 建站, 移动充电桩数量 = {decisions['m'][j]}")
            else:
                print(f"枢纽 {j}: 不建站")
        print("\n" + "=" * 50)
        break

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
        'single_tree_stats': dict(master.stats) if 'master' in locals() else None,
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
        help="toy=手写 3 区；hetero_10zone=新 10 区；canonical_10zone=冻结 10 区；futian、luohu、guangming、pingshan 读对应 npz",
    )
    parser.add_argument("--master-mip-gap", type=float, default=None,
                        help="每一轮主问题的 Gurobi 相对 MIP gap。不写则用 Gurobi 默认 1e-4")
    parser.add_argument("--adaptive-gap", action="store_true",
                        help="加割轮次主问题相对间隙用 --master-mip-gap-loose（默认 0.02）；没有新割且本轮 OPTIMAL 后收到 --master-mip-gap（默认 1e-4）")
    parser.add_argument("--master-mip-gap-loose", type=float, default=2e-2,
                        help="adaptive-gap 加割阶段的相对 MIP gap，默认 0.02")
    parser.add_argument("--master-mip-gap-abs", type=float, default=None,
                        help="每一轮主问题的绝对空隙。达到后 Gurobi 停。不写则用默认 1e-10")
    parser.add_argument("--master-time-limit", type=float, default=600.0,
                        help="每一轮主问题最多算多少秒，默认 600；加割阶段时限硬停，可用可行解加割")
    parser.add_argument("--master-time-limit-tight", type=float, default=1800.0,
                        help="adaptive-gap 认证轮主问题最多算多少秒，默认 1800")
    parser.add_argument("--certify-extend-on-timelimit", action="store_true",
                        help="仅认证轮：时限到了仍可有限次加倍时限继续同一棵树")
    parser.add_argument("--max-timelimit-extends", type=int, default=2,
                        help="认证轮最多加长时限次数，默认 2")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="割平面最多迭代多少轮，默认 100；到达后停止，不再自动翻倍")
    parser.add_argument("--max-total-seconds", type=float, default=None,
                        help="整个算法的墙钟上限（秒）。不写则不限制总时间")
    parser.add_argument("--cut-tolerance", type=float, default=1e-4,
                        help="eta 超过分离值多少才加割，默认 1e-4")
    parser.add_argument("--cut-cap", type=int, default=30,
                        help="每一轮最多新加多少条割，默认 30；对照实验可设很大")
    parser.add_argument("--samples-per-iteration", type=int, default=40,
                        help="本轮已经发现违反后，最多再检查多少个样本，默认 40；未发现违反时仍扫完全部样本")
    parser.add_argument("--disallow-cuts-on-timelimit", action="store_true",
                        help="主问题时限未证到间隙时不加割（旧行为）。默认允许用可行解加割")
    parser.add_argument("--separation-mip-gap", type=float, default=None,
                        help="分离子问题的相对 MIP gap。不写则证明到 Gurobi 默认精度")
    parser.add_argument("--separation-workers", type=int, default=1,
                        help="保留参数；当前顺序扫样本时仅用于把每个样本 Gurobi 线程压到 1")
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
                        help="只对行政区实例有效。目录里要有 <instance>_parameters.npz 和 business.json。不写则用 CG_solver 下同名目录")
    parser.add_argument("--output", type=Path, default=None,
                        help="把目标、完整调度决策(x/y/m/n/z)、固定/移动桩汇总数写入 JSON。不写则只在终端打印")
    args = parser.parse_args()
    if args.master_mip_gap is not None and args.master_mip_gap < 0:
        parser.error("master-mip-gap 不能为负")
    if args.adaptive_gap and args.master_mip_gap is not None and args.master_mip_gap > args.master_mip_gap_loose:
        parser.error("master-mip-gap（认证间隙）不能比 master-mip-gap-loose 更松")
    if args.master_mip_gap_loose <= 0:
        parser.error("master-mip-gap-loose 必须为正")
    if args.master_mip_gap_abs is not None and args.master_mip_gap_abs < 0:
        parser.error("master-mip-gap-abs 不能为负")
    if args.master_time_limit <= 0 or args.master_time_limit_tight <= 0 or args.max_iterations <= 0:
        parser.error("master-time-limit、master-time-limit-tight 和 max-iterations 必须为正")
    if args.max_total_seconds is not None and args.max_total_seconds <= 0:
        parser.error("max-total-seconds 必须为正")
    if args.cut_tolerance < 0:
        parser.error("cut-tolerance 不能为负")
    if args.cut_cap <= 0 or args.samples_per_iteration <= 0:
        parser.error("cut-cap 与 samples-per-iteration 必须为正")
    if args.max_timelimit_extends < 0:
        parser.error("max-timelimit-extends 不能为负")
    if args.separation_mip_gap is not None and args.separation_mip_gap < 0:
        parser.error("separation-mip-gap 不能为负")
    if args.separation_workers <= 0:
        parser.error("separation-workers 必须为正")
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
    if any(value is not None for value in business_overrides.values()) and args.instance not in DISTRICT_DIRS:
        parser.error("业务常数覆盖只对 --instance futian、luohu、guangming、pingshan 生效")
    if args.data_dir is not None and args.instance not in DISTRICT_DIRS:
        parser.error("data-dir 只对 --instance futian、luohu、guangming、pingshan 生效")
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
        'master_mip_gap_loose': args.master_mip_gap_loose,
        'master_mip_gap_tight': 1e-4 if args.master_mip_gap is None else args.master_mip_gap,
        'master_mip_gap_abs': args.master_mip_gap_abs,
        'master_time_limit': args.master_time_limit,
        'master_time_limit_tight': args.master_time_limit_tight if args.adaptive_gap else None,
        'certify_extend_on_timelimit': args.certify_extend_on_timelimit,
        'max_timelimit_extends': args.max_timelimit_extends,
        'max_iterations': args.max_iterations,
        'max_total_seconds': args.max_total_seconds,
        'cut_tolerance': args.cut_tolerance,
        'cut_cap': args.cut_cap,
        'samples_per_iteration': args.samples_per_iteration,
        'allow_cuts_on_timelimit': not args.disallow_cuts_on_timelimit,
        'separation_mip_gap': args.separation_mip_gap,
        'separation_workers': args.separation_workers,
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
            "single-tree branch-and-cut; LazyConstraints at MIPSOL; "
            "heuristic separation in callback; exact MIQCP certify outside tree; "
            + (
                f"adaptive master gap {args.master_mip_gap_loose} then "
                + str(1e-4 if args.master_mip_gap is None else args.master_mip_gap)
                if args.adaptive_gap else
                "fixed master gap"
            )
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
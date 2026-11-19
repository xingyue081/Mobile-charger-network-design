import argparse
import copy
import json
import math
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

# 从我们刚刚创建的两个模块导入函数
from master import CentralMaster
from oracle import compute_h_val, find_worst_pattern

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


def _read_zone_centroids(path):
    """每个 TAZ 的几何中心，GCJ-02 经纬度。顺序由调用者按 taz_ids 重排。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少 zone 几何中心：{path}")
    centroids = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if header != ["taz_id", "longitude", "latitude"]:
            raise ValueError(f"zone 几何中心表头应为 taz_id,longitude,latitude：{path}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            taz_text, lon_text, lat_text = line.strip().split(",")
            taz_id = int(taz_text)
            longitude = float(lon_text)
            latitude = float(lat_text)
            if taz_id in centroids:
                raise ValueError(f"{path} 第 {line_number} 行重复了 TAZ {taz_id}")
            centroids[taz_id] = (longitude, latitude)
    return centroids


def _read_candidate_hubs(path):
    """用户传入的候选 hub。坐标是 GCJ-02，taz_id 只用来给这个 hub 起名。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少候选 hub 文件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coordinate_system") != "GCJ-02":
        raise ValueError(f"{path} 的 coordinate_system 必须是 GCJ-02")
    hubs = payload.get("hubs")
    if not isinstance(hubs, list) or not hubs:
        raise ValueError(f"{path} 里 hubs 必须是非空列表")
    parsed = []
    seen = set()
    for index, hub in enumerate(hubs):
        name = str(hub["name"])
        taz_id = int(hub["taz_id"])
        longitude = float(hub["longitude"])
        latitude = float(hub["latitude"])
        if taz_id in seen:
            raise ValueError(f"{path} 里 TAZ {taz_id} 出现了两次")
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError(f"{path} 第 {index} 个 hub 坐标不是有限数")
        seen.add(taz_id)
        parsed.append({
            "name": name,
            "taz_id": taz_id,
            "longitude": longitude,
            "latitude": latitude,
        })
    return parsed


def _hub_zone_distance_meters(zone_lon, zone_lat, hub_lon, hub_lat):
    """几何中心到 hub 坐标的平面欧式距离，单位米。"""
    radius = 6371000.0
    latitude_mid = np.radians((zone_lat[:, None] + hub_lat[None, :]) / 2.0)
    east = np.radians(hub_lon[None, :] - zone_lon[:, None]) * radius * np.cos(latitude_mid)
    north = np.radians(hub_lat[None, :] - zone_lat[:, None]) * radius
    return np.hypot(east, north)


def load_futian_data(overrides=None, data_dir=None, hubs_path=None):
    """把一份福田格式的目录装成割平面已经使用的字典。

    默认目录是 CG_solver/futian。data_dir 只换读哪一份 npz、business.json 和 hub 文件，
    不改主问题、分离或运输成本的写法。错峰实验用它读平移后的样本。
    样本、均值、协方差、半径和 epsilon 来自 npz。
    rho、b、tau、kappa、c^f、c^m、gamma、hub 开设成本来自同一目录的 business.json。
    候选 hub 来自 candidate_hubs.json，或 --hubs 指定的文件。
    hub 到 zone 的距离是该 zone 几何中心到 hub 坐标的平面欧式距离。
    往返系数 2 只属于集中补能。
    """
    root = FUTIAN_DIR if data_dir is None else Path(data_dir)
    stored_path = _parameter_npz(root)
    with np.load(stored_path, allow_pickle=False) as stored:
        zone_ids = tuple(int(value) for value in stored["taz_ids"].tolist())
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
    if mu.shape != (zone_count,):
        raise ValueError("数据包维度与 taz_ids 不一致")
    if q_l.shape != (3,) or epsilon_l.shape != (3,):
        raise ValueError("日期类型参数必须是 3 维")
    centroids = _read_zone_centroids(root / "zone_centroids.csv")
    missing_centroids = [taz_id for taz_id in zone_ids if taz_id not in centroids]
    if missing_centroids:
        raise ValueError(f"几何中心缺少 TAZ：{missing_centroids}")
    hub_file = root / "candidate_hubs.json" if hubs_path is None else Path(hubs_path)
    hubs = _read_candidate_hubs(hub_file)
    unknown_hubs = [hub["taz_id"] for hub in hubs if hub["taz_id"] not in zone_ids]
    if unknown_hubs:
        raise ValueError(f"候选 hub 不在这 35 个 zone 里：{unknown_hubs}")
    zone_lon = np.array([centroids[taz_id][0] for taz_id in zone_ids], dtype=float)
    zone_lat = np.array([centroids[taz_id][1] for taz_id in zone_ids], dtype=float)
    hub_lon = np.array([hub["longitude"] for hub in hubs], dtype=float)
    hub_lat = np.array([hub["latitude"] for hub in hubs], dtype=float)
    distance = _hub_zone_distance_meters(zone_lon, zone_lat, hub_lon, hub_lat)
    if distance.shape != (zone_count, len(hubs)) or not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("hub 到 zone 的距离必须是有限非负的 (zone, hub) 矩阵")
    business, business_overrides, business_path = load_shared_business(root, overrides)
    hub_open_cost = business["hub_open_cost"]
    fleet_per_hub = 30.0
    hub_count = len(hubs)
    samples = {
        date_index: [row.copy() for row in rows]
        for date_index, rows in enumerate(sample_rows)
    }
    return {
        "name": "futian",
        "zone_ids": zone_ids,
        "hub_ids": tuple(hub["taz_id"] for hub in hubs),
        "hub_names": tuple(hub["name"] for hub in hubs),
        "date_names": DATE_TYPES,
        "I_num": zone_count,
        "J_num": hub_count,
        "S_num": len(DATE_TYPES),
        "samples": samples,
        "q_l": q_l.tolist(),
        "epsilon_l": epsilon_l.tolist(),
        "mu": mu,
        "Sigma_inv": sigma_inv,
        "r": radius,
        "f": [hub_open_cost] * hub_count,
        "c_m": business["mobile_unit_cost"],
        "c_f": business["fixed_unit_cost"],
        "rho": business["rho"],
        "b": business["shortage_penalty"],
        "tau": business["fixed_capacity"],
        "kappa": business["mobile_capacity"],
        "M_m": [fleet_per_hub] * hub_count,
        "n_bar": n_bar,
        "M_z": np.full((zone_count, hub_count), fleet_per_hub),
        "gamma": business["transport_cost_per_distance"],
        "t_ij": 2.0 * business["transport_cost_per_distance"] * distance,
        "hub_distance_m": distance,
        "business_path": business_path,
        "business_overrides": business_overrides,
        "hubs_path": hub_file,
        "h_val": {},
    }


def load_model_data(instance_name, business_overrides=None, data_dir=None, hubs_path=None):
    """--instance 的唯一入口。toy 用手写数据，futian 读 npz，10 区读 pickle。

    data_dir 和 hubs_path 只对 futian 有意义。不传 data_dir 时读 CG_solver/futian。
    """
    if instance_name == "toy":
        return build_toy_data()
    if instance_name == "futian":
        model_data = load_futian_data(business_overrides, data_dir, hubs_path)
        sample_counts = {
            model_data["date_names"][l]: len(model_data["samples"][l])
            for l in range(model_data["S_num"])
        }
        named_hubs = list(zip(model_data["hub_names"], model_data["hub_ids"]))
        print(
            f"已载入 {model_data['name']}: 区域 {model_data['zone_ids']}，"
            f"候选 hub {named_hubs}，"
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
    # 1) 加割阶段可用较松 MIP gap；时限到了也可拿可行解加割（认证轮除外）。
    # 2) cut_cap / samples_per_iteration 限制每轮写入的割与继续扫描的样本。
    # 3) 主问题只建一次，后续只 addConstr。
    started = perf_counter()
    iteration = 1
    master_solves = 0
    last_decisions = None
    stop_reason = None
    best_upper = math.inf
    best_lower = -math.inf
    best_decisions = None
    origin_gap = None
    optimality_gap = float(limits.get("optimality_gap", 0.005))
    pattern_order = []
    pattern_seen = set()
    pattern_rounds = []
    certifying = False
    sample_cursor = 0
    sample_keys = [
        (l, n_idx)
        for l in range(S_num)
        for n_idx in range(len(samples[l]))
    ]
    n_samples = len(sample_keys)
    cut_cap = limits.get("cut_cap")
    if cut_cap is None:
        cut_cap = n_samples
    cut_cap = max(1, int(cut_cap))
    samples_per_iteration = limits.get("samples_per_iteration")
    if samples_per_iteration is None:
        samples_per_iteration = n_samples
    samples_per_iteration = max(1, int(samples_per_iteration))
    allow_cuts_on_timelimit = bool(limits.get("allow_cuts_on_timelimit", True))
    for patterns in active_patterns.values():
        for pattern in patterns:
            if pattern not in pattern_seen:
                pattern_seen.add(pattern)
                pattern_order.append(pattern)

    master = CentralMaster(toy_data, active_patterns)
    print(
        f"持久化主问题已建立：初始割 {sum(len(v) for v in active_patterns.values())} 条，"
        f"cut_cap={cut_cap}，samples_per_iteration={samples_per_iteration}，"
        f"allow_cuts_on_timelimit={allow_cuts_on_timelimit}",
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

        print(f"\n{'='*40}\n 开始第 {iteration} 轮迭代求解主模型...\n{'='*40}")
        if limits.get("adaptive_gap"):
            master_mip_gap = (
                limits["master_mip_gap_tight"] if certifying else limits["master_mip_gap_loose"]
            )
        else:
            master_mip_gap = limits["master_mip_gap"]
            if master_mip_gap is None:
                master_mip_gap = optimality_gap

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

        # 加割阶段：时限硬停，用可行解继续。认证阶段可选有限次加长。
        extend_on_timelimit = bool(certifying and limits.get("certify_extend_on_timelimit"))
        is_solved, C_opt, eta_opt, T_opt, decisions = master.optimize(
            {
                'mip_gap': master_mip_gap,
                'mip_gap_abs': limits.get('master_mip_gap_abs'),
                'optimality_tol': limits.get('optimality_tol'),
                'time_limit': master_time_limit,
                'output_flag': limits.get('master_log'),
                'extend_on_timelimit': extend_on_timelimit,
                'max_timelimit_extends': int(limits.get('max_timelimit_extends') or 2),
            },
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
        if decisions.get("obj_bound") is not None and math.isfinite(decisions["obj_bound"]):
            best_upper = min(best_upper, float(decisions["obj_bound"]))
        master_solves += 1
        last_decisions = decisions
        if decisions['status_name'] != "OPTIMAL":
            print("这一轮主问题未证明最优。后面的分离用的是当前可行解。")

        cuts_added = 0
        accounted_violations = 0
        separation_finished = True
        active_residual = False
        heuristic_cuts = 0
        miqcp_calls = 0
        samples_examined = 0
        # 加割阶段：OPTIMAL 或（允许时）时限可行解都可以写割。
        # 认证阶段：只有 OPTIMAL 才写割。
        if certifying:
            write_cuts = decisions['status_name'] == "OPTIMAL"
        else:
            write_cuts = decisions['status_name'] == "OPTIMAL" or (
                allow_cuts_on_timelimit
                and decisions['status_name'] in ("TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED")
            )
        separation_started = perf_counter()
        accounted_operating = 0.0
        lb_certified = True

        remaining = None
        if limits['max_total_seconds'] is not None:
            remaining = limits['max_total_seconds'] - (perf_counter() - started)
            if remaining <= 0:
                stop_reason = "达到总时间上限"
                separation_finished = False
                print("\n分离前已达到总时间上限，停止。")

        # 每个样本都用精确分离算真实最坏利润，才能得到原问题的可行下界。
        # cut_cap 只限制写入主问题的新割数量，不再提前结束这轮核算。
        order = sample_keys[sample_cursor:] + sample_keys[:sample_cursor]
        found_violation = False
        cut_cap_announced = False
        sep_workers = int(limits.get("separation_workers") or 1)
        sep_threads = 1 if sep_workers > 1 else None

        if separation_finished:
            for key in order:
                if limits['max_total_seconds'] is not None:
                    remaining = limits['max_total_seconds'] - (perf_counter() - started)
                    if remaining <= 0:
                        stop_reason = "达到总时间上限"
                        separation_finished = False
                        print("\n分离中达到总时间上限，停止。")
                        break

                l, n_idx = key
                D_sample = samples[l][n_idx]
                eps = epsilon_l[l]
                L = np.maximum(0, D_sample - eps)
                U = D_sample + eps
                C_star = [C_opt[l, i] for i in range(I_num)]
                samples_examined += 1
                sample_time_limit = None if remaining is None else min(float(remaining), 30.0)
                try:
                    a_worst, true_worst_profit, sep_info = find_worst_pattern(
                        C_star=C_star, L=L, U=U, mu=mu, Sigma_inv=Sigma_inv,
                        r=r, rho=rho, b=b,
                        mip_gap=limits.get('separation_mip_gap'),
                        time_limit=sample_time_limit,
                        sample_demand=np.asarray(D_sample, dtype=float),
                        eta=eta_opt[l, n_idx],
                        transport=T_opt[l],
                        cut_tolerance=limits['cut_tolerance'],
                        force_exact=True,
                        threads=sep_threads,
                    )
                except RuntimeError as exc:
                    lb_certified = False
                    print(f"\n样本 ({l},{n_idx}) 分离失败，本轮不更新下界：{exc}")
                    continue

                if sep_info.get("used_miqcp"):
                    miqcp_calls += 1
                profit_lower = sep_info.get("profit_lower")
                sample_weight = toy_data['q_l'][l] / len(samples[l])
                if profit_lower is None or not math.isfinite(profit_lower):
                    lb_certified = False
                else:
                    accounted_operating += sample_weight * (profit_lower - T_opt[l])
                cut_pattern = sep_info.get("cut_pattern")
                cut_profit = sep_info.get("cut_profit")
                if (
                    cut_pattern is not None
                    and cut_profit is not None
                    and eta_opt[l, n_idx] > cut_profit - T_opt[l] + limits['cut_tolerance']
                ):
                    a_worst = cut_pattern
                    true_worst_profit = cut_profit
                    found_violation = True
                    if a_worst not in active_patterns[(l, n_idx)]:
                        if not write_cuts or cuts_added >= cut_cap:
                            if write_cuts and cuts_added >= cut_cap and not cut_cap_announced:
                                cut_cap_announced = True
                                print(
                                    f"  本轮新割已达 cut_cap={cut_cap}，"
                                    "后面的样本继续核算原问题目标，不再加割。",
                                    flush=True,
                                )
                            if not write_cuts:
                                accounted_violations += 1
                                print(
                                    f"  [核算到违背，不写入] 情形 {l} 样本 {n_idx}: 模式 {a_worst}"
                                )
                            continue
                        h_val_new = sum((rho + b) * C_star[i] * a_worst[i] for i in range(I_num)) - true_worst_profit
                        toy_data['h_val'][(l, n_idx, a_worst)] = h_val_new
                        master.add_pattern(l, n_idx, a_worst)
                        cuts_added += 1
                        if not sep_info.get("used_miqcp"):
                            heuristic_cuts += 1
                        if a_worst not in pattern_seen:
                            pattern_seen.add(a_worst)
                            pattern_order.append(a_worst)
                        how = "MIQCP" if sep_info.get("used_miqcp") else "启发式"
                        print(f"  [发现违背/{how}] 情形 {l} 样本 {n_idx}: 新增恶劣模式 {a_worst}")
                    else:
                        excess = eta_opt[l, n_idx] - (true_worst_profit - T_opt[l])
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

        if sample_keys:
            sample_cursor = (sample_cursor + max(samples_examined, 1)) % n_samples

        separation_seconds = perf_counter() - separation_started
        investment = (
            sum(toy_data['f'][j] * decisions['x'][j] for j in range(J_num))
            + toy_data['c_m'] * sum(decisions['m'].values())
            + toy_data['c_f'] * sum(decisions['n'].values())
        )
        print(
            f"--> 本轮迭代共新增 {cuts_added} 条割平面"
            f"（启发式直接加割 {heuristic_cuts}，MIQCP 调用 {miqcp_calls}，"
            f"检查样本 {samples_examined}/{n_samples}）。"
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
            "cuts_added": cuts_added,
            "heuristic_cuts": heuristic_cuts,
            "miqcp_calls": miqcp_calls,
            "samples_examined": samples_examined,
            "cut_cap": cut_cap,
            "cuts_written": write_cuts,
            "accounted_violations_not_written": accounted_violations,
            "unique_patterns": len(pattern_order),
            "separation_seconds": separation_seconds,
            "separation_finished": separation_finished,
            "separation_workers": sep_workers,
            "accounted_objective": None if not (separation_finished and samples_examined == n_samples) else -investment + accounted_operating,
            "gurobi_log": None if gurobi_log is None else str(gurobi_log),
            "certifying": certifying,
        }
        pattern_rounds.append(round_record)
        evaluated_all = separation_finished and lb_certified and samples_examined == n_samples
        if evaluated_all:
            true_obj = -investment + accounted_operating
            if true_obj > best_lower:
                best_lower = true_obj
                best_decisions = copy.deepcopy(decisions)
        if math.isfinite(best_upper) and math.isfinite(best_lower) and abs(best_lower) > 1e-9:
            origin_gap = max(0.0, (best_upper - best_lower) / abs(best_lower))
        round_record["origin_upper"] = None if not math.isfinite(best_upper) else best_upper
        round_record["origin_lower"] = None if not math.isfinite(best_lower) else best_lower
        round_record["origin_gap"] = origin_gap
        if limits.get("round_log") is not None:
            round_log = Path(limits["round_log"])
            round_log.parent.mkdir(parents=True, exist_ok=True)
            with round_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(round_record, ensure_ascii=False) + "\n")
        print(
            f"PATTERN_ROUND iteration={iteration} cuts_added={cuts_added} "
            f"unique={len(pattern_order)} master={decisions['status_name']} "
            f"obj={decisions['obj']:.6g} master_gap={decisions['mip_gap']:.3g} "
            f"origin_gap={None if origin_gap is None else round(origin_gap, 6)} "
            f"origin_lb={None if not math.isfinite(best_lower) else round(best_lower, 4)} "
            f"origin_ub={None if not math.isfinite(best_upper) else round(best_upper, 4)} "
            f"master_s={decisions['runtime']:.4f} "
            f"nodes={decisions['node_count']} sep_s={separation_seconds:.4f} "
            f"nzs={decisions.get('num_nzs')} warm={int(bool(decisions.get('warm_start')))} "
            f"first_incumbent_s={decisions.get('first_incumbent_seconds')}",
            flush=True,
        )

        # 3.4 收敛检查。原问题间隙 = (松弛上界 - 可行下界) / |可行下界|。
        if not separation_finished:
            print(f"\n分离未完成：{stop_reason}")
            if stop_reason == "达到总时间上限":
                break
            iteration += 1
            continue
        if origin_gap is not None and origin_gap <= optimality_gap:
            stop_reason = "原问题相对间隙已达到要求"
            print(
                f"\n原问题相对间隙 {origin_gap:.4%}，"
                f"上界 {best_upper:.6g}，下界 {best_lower:.6g}，"
                f"目标 {optimality_gap:.2%}。"
            )
            decisions = best_decisions
            decisions["obj"] = best_lower
            decisions["obj_bound"] = best_upper
            decisions["mip_gap"] = origin_gap
        elif cuts_added > 0 or origin_gap is None or origin_gap > optimality_gap:
            if cuts_added == 0:
                print(
                    f"\n没有新的割。原问题相对间隙 {origin_gap}，"
                    "继续收紧主问题上界。"
                )
            iteration += 1
            continue
        if (
            stop_reason != "原问题相对间隙已达到要求"
            and limits.get("adaptive_gap")
            and cuts_added == 0
            and separation_finished
            and not active_residual
            and not certifying
            and decisions['status_name'] == "OPTIMAL"
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
        if stop_reason != "原问题相对间隙已达到要求" and cuts_added == 0 and active_residual:
            stop_reason = "已有割仍被违反"
            print("\n已有割仍被违反，不宣布收敛。没有改写已经写入的 h。")
            break
        if stop_reason == "原问题相对间隙已达到要求":
            print("\n" + "="*50)
            print(f"原问题可行目标: {decisions['obj']}")
            print(f"原问题上界: {decisions['obj_bound']}")
            print(f"原问题相对间隙: {decisions['mip_gap']:.6g}")
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
                        f"枢纽 {toy_data['hub_ids'][j]}: 建站, "
                        f"移动充电桩数量 = {decisions['m'][j]}"
                    )

                else:

                    print(f"枢纽 {toy_data['hub_ids'][j]}: 不建站")

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
                    hub_info = f"归属枢纽 {toy_data['hub_ids'][assigned_hub[0]]}"
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

                    print(f"  枢纽 {toy_data['hub_ids'][j]}:")

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

    if origin_gap is not None:
        print(
            f"\n停止：{stop_reason}。原问题下界 "
            f"{None if not math.isfinite(best_lower) else round(best_lower, 4)}，上界 "
            f"{None if not math.isfinite(best_upper) else round(best_upper, 4)}，"
            f"相对间隙 {origin_gap:.4%}。"
        )
    reported = best_decisions if best_decisions is not None else last_decisions
    plan = None
    if reported is not None and 'x' in reported:
        plan = {
            'x': reported['x'],
            'y': reported['y'],
            'm': reported['m'],
            'n': reported['n'],
            'z': reported['z'],
        }
    return {
        'objective': None if not math.isfinite(best_lower) else best_lower,
        'bound': None if not math.isfinite(best_upper) else best_upper,
        'mip_gap': origin_gap,
        'iterations': master_solves,
        'stop_reason': stop_reason,
        'status_name': None if reported is None else reported.get('status_name'),
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
    parser.add_argument("--optimality-gap", type=float, default=0.005,
                        help="原问题相对间隙 (上界-下界)/|下界|，默认 0.005。主问题不另行指定间隙时也用这个值")
    parser.add_argument("--max-total-seconds", type=float, default=3600.0,
                        help="整个算法的墙钟上限（秒），默认 3600")
    parser.add_argument("--cut-tolerance", type=float, default=1e-4,
                        help="eta 超过分离值多少才加割，默认 1e-4")
    parser.add_argument("--cut-cap", type=int, default=None,
                        help="每一轮最多新加多少条割。不写则本轮每个被检查的样本最多写一条")
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
                        help="只对 futian 有效。目录里要有 futian_parameters.npz 和 business.json。不写则用 CG_solver/futian")
    parser.add_argument("--hubs", type=Path, default=None,
                        help="只对 futian 有效。候选 hub 的 JSON。不写则用数据目录里的 candidate_hubs.json")
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
    if args.optimality_gap <= 0:
        parser.error("optimality-gap 必须为正")
    if args.max_total_seconds is not None and args.max_total_seconds <= 0:
        parser.error("max-total-seconds 必须为正")
    if args.cut_tolerance < 0:
        parser.error("cut-tolerance 不能为负")
    if args.cut_cap is not None and args.cut_cap <= 0 or args.samples_per_iteration <= 0:
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
    if any(value is not None for value in business_overrides.values()) and args.instance != "futian":
        parser.error("业务常数覆盖只对 --instance futian 生效")
    if args.data_dir is not None and args.instance != "futian":
        parser.error("data-dir 只对 --instance futian 生效")
    if args.hubs is not None and args.instance != "futian":
        parser.error("hubs 只对 --instance futian 生效")
    model_data = load_model_data(args.instance, business_overrides, args.data_dir, args.hubs)
    round_log = None
    gurobi_log_dir = None
    if args.output is not None:
        round_log = args.output.with_name(args.output.stem + "_rounds.jsonl")
        gurobi_log_dir = args.output.with_name(args.output.stem + "_gurobi")
        if round_log.exists():
            round_log.unlink()
    result = run_cutting_plane_algorithm(model_data, {
        'master_mip_gap': args.optimality_gap if args.master_mip_gap is None else args.master_mip_gap,
        'optimality_gap': args.optimality_gap,
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
        summary["hub_ids"] = [int(value) for value in model_data["hub_ids"]]
        summary["optimality_gap_target"] = args.optimality_gap
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
            "adaptive master gap 1e-3 until no new cut, then "
            + str(1e-4 if args.master_mip_gap is None else args.master_mip_gap)
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
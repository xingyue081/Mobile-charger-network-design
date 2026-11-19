"""按数值实验设计.md 的执行方案求解，并保留每次的完整 JSON。

已收敛且参数一致的解只复制。locally 旧解按保存的部署补最小费用调度流量，
不改 n、m、z，也不重跑割平面。未收敛的新解不复制到正式目录。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

CG_ROOT = Path("/Users/xingyueyi/Desktop/CG_solver")
EXP_ROOT = CG_ROOT / "numerical_exp"
ALPHA_ROOT = EXP_ROOT / "exp_futian_alpha"
FUTIAN = CG_ROOT / "futian"
PYTHON = Path(
    "/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/.venv-gurobi-13.0.3/bin/python"
)
SCREEN = ALPHA_ROOT / "outputs" / "screen"
OLD_FIXED = ALPHA_ROOT / "outputs" / "final" / "g0.3_hub55"
BASE = EXP_ROOT / "base" / "alpha1_b0"
LOG_PATH = EXP_ROOT / "run_plan.log"
PROGRESS_PATH = EXP_ROOT / "progress.json"
WRITE_LOCK = threading.RLock()
sys.path.insert(0, str(ALPHA_ROOT))
from shift import alpha_tag, write_alpha  # noqa: E402

ALPHAS = tuple(round(step / 10, 1) for step in range(11))
REUSED_FIXED_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NEW_FIXED_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def log(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass
    with WRITE_LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mark(case: str, state: str, detail: str = "") -> None:
    with WRITE_LOCK:
        progress = load_json(PROGRESS_PATH) or {}
        progress[case] = {"state": state, "detail": detail}
        write_json(PROGRESS_PATH, progress)
    log(f"[{state}] {case} {detail}".rstrip())


def converged(saved: dict, mode: str) -> bool:
    if mode == "charge-locally":
        return saved.get("converged") is True and saved.get("reason") == "ALL_SEPARATIONS_WITHIN_TOLERANCE"
    if mode == "charge-centrally":
        return saved.get("status_name") == "OPTIMAL" and "没有新的割" in str(saved.get("stop_reason"))
    if mode == "fixed-only":
        return saved.get("converged") is True
    raise ValueError(mode)


def data_dir(alpha: float) -> Path:
    if alpha == 1.0:
        return FUTIAN
    return ALPHA_ROOT / "data" / f"alpha_{alpha_tag(alpha)}"


def alpha_out(alpha: float, name: str) -> Path:
    return ALPHA_ROOT / "outputs" / "g0.25_hub50" / f"alpha_{alpha_tag(alpha)}" / name


def check_reused_local(saved: dict, g_value: float) -> None:
    if not converged(saved, "charge-locally"):
        raise RuntimeError("locally 旧解没有收敛")
    if abs(float(saved["local_g"]) - g_value) > 1e-12:
        raise RuntimeError(f"local_g 不是 {g_value}")
    overrides = saved["provenance"]["business_overrides"]
    if float(overrides.get("shortage_penalty")) != 0.0:
        raise RuntimeError("旧 locally 的 b 不是 0")
    if float(overrides.get("transport_cost_per_distance")) != 0.001:
        raise RuntimeError("旧 locally 的 gamma 不是 0.001")
    for field in ("rho", "fixed_unit_cost", "mobile_unit_cost"):
        if field in overrides:
            raise RuntimeError(f"旧 locally 覆盖了 {field}，不能直接沿用")
    saved["parameters"] = {
        "rho": 2.0,
        "shortage_penalty": 0.0,
        "fixed_unit_cost": 41.0,
        "mobile_unit_cost": 45.0,
        "gamma": 0.001,
        "g": g_value,
    }


def min_cost_flows(saved: dict) -> list[dict]:
    """固定已保存的 z，按单程距离求每条日期转移的最小费用流。"""
    decision = saved["decision"]
    deployment = {
        date_type: [int(round(value)) for value in counts]
        for date_type, counts in decision["z"].items()
    }
    fleet = int(round(float(decision["m"])))
    for date_type, counts in deployment.items():
        if sum(counts) != fleet:
            raise RuntimeError(f"{date_type} 部署合计 {sum(counts)}，车队是 {fleet}")
    provenance = saved["provenance"]
    taz_ids = [int(value) for value in provenance["taz_ids"]]
    with np.load(provenance["artifact"], allow_pickle=False) as stored:
        distance = stored["distance"].astype(float)
        stored_ids = [int(value) for value in stored["taz_ids"].tolist()]
    if stored_ids != taz_ids:
        raise RuntimeError("距离矩阵的 taz 顺序与解不一致")
    zone_count = len(taz_ids)
    flows = []
    for label, probability in provenance["transition_probability"].items():
        src, dst = label.split("->")
        supply = deployment[src]
        demand = deployment[dst]
        if float(probability) <= 0.0 or fleet == 0:
            continue
        model = gp.Model("recover_flow")
        model.Params.OutputFlag = 0
        flow = model.addVars(zone_count, zone_count, vtype=GRB.INTEGER, lb=0, name="r")
        model.addConstrs((flow.sum(origin, "*") == supply[origin] for origin in range(zone_count)))
        model.addConstrs((flow.sum("*", destination) == demand[destination] for destination in range(zone_count)))
        model.setObjective(
            gp.quicksum(
                distance[origin, destination] * flow[origin, destination]
                for origin in range(zone_count)
                for destination in range(zone_count)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{label} 的调度流量没有解到最优，状态 {model.Status}")
        for origin in range(zone_count):
            for destination in range(zone_count):
                vehicles = int(round(flow[origin, destination].X))
                if vehicles <= 0:
                    continue
                flows.append({
                    "from_date": src,
                    "to_date": dst,
                    "origin_index": origin,
                    "destination_index": destination,
                    "origin_taz": taz_ids[origin],
                    "destination_taz": taz_ids[destination],
                    "vehicles": vehicles,
                })
        model.dispose()
    return flows


def recover_local(source: Path, g_value: float, destinations: list[Path]) -> None:
    pending = [path for path in destinations if not (path.is_file() and (load_json(path) or {}).get("decision", {}).get("flow") is not None)]
    if not pending:
        log(f"沿用已补流量的 locally {source.name} g={g_value}")
        return
    saved = load_json(source)
    check_reused_local(saved, g_value)
    if saved["decision"].get("flow") is None:
        saved["decision"]["flow"] = min_cost_flows(saved)
        saved["flow_recovery"] = (
            "n、m、z 保持原收敛解。日期之间的流量是把这份部署固定后的最小单程距离流；"
            "流量不进入割，只进入目标里的 repositioning 成本。"
        )
    for path in pending:
        write_json(path, saved)
    log(f"已补流量并写入 {len(pending)} 处：g={g_value}，来源 {source}")


def prepare_alphas() -> None:
    previous = {}
    for alpha in REUSED_FIXED_ALPHAS:
        report_path = ALPHA_ROOT / "data" / f"alpha_{alpha_tag(alpha)}" / "shift_report.json"
        if report_path.is_file():
            previous[alpha] = load_json(report_path)
    for alpha in ALPHAS:
        report = write_alpha(alpha)
        old = previous.get(alpha)
        if old is None:
            continue
        if report["kept_counts"] != old["kept_counts"] or report["dropped_days"] != old["dropped_days"]:
            raise RuntimeError(f"alpha={alpha} 重新生成后的样本保留情况与旧错峰数据不同")
        if alpha == 1.0 and report["dropped_days"]:
            raise RuntimeError("alpha=1 丢掉了样本，不能把原始福田目录当作 alpha=1")
    log("错峰数据已生成")


def check_reused_fixed(saved: dict, alpha: float) -> None:
    if not converged(saved, "fixed-only"):
        raise RuntimeError("fixed-only 旧解没有收敛")
    parameters = saved["parameters"]
    if float(parameters["rho"]) != 2.0 or float(parameters["shortage_penalty"]) != 0.0:
        raise RuntimeError("fixed-only 旧解的 rho 或 b 不一致")
    if float(parameters["fixed_unit_cost"]) != 41.0:
        raise RuntimeError("fixed-only 旧解的 c^f 不是 41")
    data = str(saved["provenance"]["data_dir"])
    if alpha == 1.0:
        if not (data.rstrip("/").endswith("/futian") or data.rstrip("/").endswith("/alpha_1")):
            raise RuntimeError(f"alpha=1 的 fixed-only 数据目录不是福田原数据：{data}")
    elif f"alpha_{alpha_tag(alpha)}" not in data:
        raise RuntimeError(f"alpha={alpha} 的 fixed-only 数据目录不匹配：{data}")


def copy_verified(source: Path, destinations: list[Path], mode: str) -> None:
    saved = load_json(source)
    if saved is None or not converged(saved, mode):
        raise RuntimeError(f"不能复制未收敛结果 {source}")
    for path in destinations:
        if path.resolve() == source.resolve():
            continue
        if path.is_file() and converged(load_json(path), mode):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)


def reuse_fixed() -> None:
    for alpha in REUSED_FIXED_ALPHAS:
        source = OLD_FIXED / f"alpha_{alpha_tag(alpha)}" / "fixed_only.json"
        saved = load_json(source)
        check_reused_fixed(saved, alpha)
        destinations = [alpha_out(alpha, "fixed_only.json")]
        if alpha == 1.0:
            destinations.extend([
                BASE / "fixed_only.json",
                EXP_ROOT / "exp_b" / "b0" / "fixed_only.json",
            ])
        copy_verified(source, destinations, "fixed-only")
        mark(f"fixed-only alpha={alpha_tag(alpha)} b=0", "reused", str(source))


def run_solver(command: list[str], cwd: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    orphan = output.with_name(output.stem + "_rounds.jsonl")
    if not output.exists() and orphan.exists():
        orphan.rename(orphan.with_suffix(".jsonl.incomplete"))
    log_path = output.with_suffix(".log")
    env = os.environ.copy()
    env["CG_THREADS"] = "4"
    log("启动 " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False,
        )
    saved = load_json(output)
    if saved is None:
        raise RuntimeError(f"没有写出结果，退出码 {completed.returncode}，日志 {log_path}")
    if completed.returncode != 0:
        log(f"进程退出码 {completed.returncode}，继续检查结果文件 {output}")
    return saved


def check_parameters(saved: dict, expected: dict) -> None:
    found = saved.get("parameters")
    if not found:
        raise RuntimeError("结果里没有 parameters")
    for key, value in expected.items():
        if abs(float(found[key]) - float(value)) > 1e-9:
            raise RuntimeError(f"{key} 是 {found[key]}，案例要求 {value}")


def solve_fixed(alpha: float, penalty: float, output: Path) -> None:
    case = f"fixed-only alpha={alpha_tag(alpha)} b={penalty}"
    existing = load_json(output)
    if existing is not None and converged(existing, "fixed-only"):
        mark(case, "skipped")
        return
    saved = run_solver(
        [
            str(PYTHON), "main.py",
            "--data", str(data_dir(alpha)),
            "--rho", "2",
            "--shortage-penalty", str(penalty),
            "--fixed-cost", "41",
            "--solve",
            "--output", str(output),
        ],
        CG_ROOT / "fixed_only_solver",
        output,
    )
    if not converged(saved, "fixed-only"):
        raise RuntimeError(f"{case} 未收敛：{saved.get('reason')}")
    check_parameters(saved, {
        "rho": 2, "shortage_penalty": penalty, "fixed_unit_cost": 41,
    })
    mark(case, "solved", output.name)


def local_command(data: Path, g_value: float, penalty: float, mobile_cost: float,
                  output: Path, resume: Path | None) -> list[str]:
    command = [
        str(PYTHON), "run_20zone.py",
        "--data", str(data),
        "--g", str(g_value),
        "--rho", "2",
        "--shortage-penalty", str(penalty),
        "--fixed-cost", "41",
        "--mobile-cost", str(mobile_cost),
        "--gamma", "0.001",
        "--adaptive-gap",
        "--solve",
        "--output", str(output),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def resume_source(output: Path, warm: Path | None) -> Path | None:
    existing = load_json(output)
    if existing is not None and existing.get("active_patterns"):
        return output
    return warm


def solve_local(case: str, data: Path, g_value: float, penalty: float, mobile_cost: float,
                output: Path, warm: Path | None, copies: list[Path]) -> None:
    existing = load_json(output)
    if existing is not None and converged(existing, "charge-locally"):
        mark(case, "skipped")
        copy_verified(output, copies, "charge-locally")
        return
    resume = resume_source(output, warm)
    saved = run_solver(
        local_command(data, g_value, penalty, mobile_cost, output, resume),
        CG_ROOT / "charge_locally_solver",
        output,
    )
    if not converged(saved, "charge-locally"):
        raise RuntimeError(f"{case} 未收敛：{saved.get('reason')}")
    if not saved.get("decision", {}).get("flow") and int(round(float(saved["decision"]["m"]))) != 0:
        raise RuntimeError(f"{case} 收敛了但没有调度流量")
    check_parameters(saved, {
        "rho": 2, "shortage_penalty": penalty, "fixed_unit_cost": 41,
        "mobile_unit_cost": mobile_cost, "gamma": 0.001, "g": g_value,
    })
    copy_verified(output, copies, "charge-locally")
    mark(case, "solved", str(output))


def central_command(data: Path, penalty: float, mobile_cost: float, hub: float, output: Path) -> list[str]:
    return [
        str(PYTHON), "main.py",
        "--instance", "futian",
        "--data-dir", str(data),
        "--rho", "2",
        "--shortage-penalty", str(penalty),
        "--fixed-cost", "41",
        "--mobile-cost", str(mobile_cost),
        "--gamma", "0.001",
        "--hub-open-cost", str(hub),
        "--adaptive-gap",
        "--max-iterations", "200",
        "--output", str(output),
    ]


def solve_central(case: str, data: Path, penalty: float, mobile_cost: float, hub: float,
                  output: Path, copies: list[Path]) -> None:
    existing = load_json(output)
    if existing is not None and converged(existing, "charge-centrally"):
        mark(case, "skipped")
        copy_verified(output, copies, "charge-centrally")
        return
    saved = run_solver(
        central_command(data, penalty, mobile_cost, hub, output),
        CG_ROOT / "charge_centrally_solver_1030",
        output,
    )
    if not converged(saved, "charge-centrally"):
        raise RuntimeError(f"{case} 未收敛：{saved.get('stop_reason')}")
    if not saved.get("decision") or not saved.get("active_patterns"):
        raise RuntimeError(f"{case} 收敛了但决策或 active_patterns 没有写入")
    check_parameters(saved, {
        "rho": 2, "shortage_penalty": penalty, "fixed_unit_cost": 41,
        "mobile_unit_cost": mobile_cost, "gamma": 0.001, "hub_open_cost": hub,
    })
    copy_verified(output, copies, "charge-centrally")
    mark(case, "solved", str(output))


def guarded(case: str, function) -> None:
    try:
        function()
    except Exception as exc:
        mark(case, "failed", str(exc))
        log(traceback.format_exc())


def row_from(mode: str, saved: dict, alpha, g_value, hub, mobile_cost, penalty) -> dict:
    if mode == "charge-locally":
        mobile = int(round(float(saved["decision"]["m"])))
        fixed = int(round(sum(float(value) for value in saved["decision"]["n"])))
        hubs = ""
    elif mode == "charge-centrally":
        mobile = int(saved["mobile_chargers"])
        fixed = int(saved["fixed_chargers"])
        hubs = int(saved["hub_count"])
    else:
        mobile = 0
        fixed = int(round(sum(float(value) for value in saved["decision"]["n"])))
        hubs = ""
    return {
        "mode": mode,
        "alpha": alpha,
        "b": penalty,
        "mobile_cost": mobile_cost,
        "g": g_value,
        "hub_open_cost": hub,
        "objective": saved["objective"],
        "bound": saved["bound"],
        "absolute_gap": float(saved["bound"]) - float(saved["objective"]),
        "mobile_chargers": mobile,
        "fixed_chargers": fixed,
        "hub_count": hubs,
        "converged": True,
    }


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode", "alpha", "b", "mobile_cost", "g", "hub_open_cost",
        "objective", "bound", "absolute_gap", "mobile_chargers", "fixed_chargers",
        "hub_count", "converged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_available_summaries() -> None:
    with WRITE_LOCK:
        _write_available_summaries()


def _write_available_summaries() -> None:
    alpha_rows = []
    for alpha in ALPHAS:
        for mode, filename in (
            ("charge-centrally", "charge_centrally.json"),
            ("charge-locally", "charge_locally.json"),
            ("fixed-only", "fixed_only.json"),
        ):
            saved = load_json(alpha_out(alpha, filename))
            if saved is not None and converged(saved, mode):
                alpha_rows.append(row_from(
                    mode, saved, alpha, 0.25 if mode == "charge-locally" else "",
                    50 if mode == "charge-centrally" else "",
                    45 if mode != "fixed-only" else "", 0,
                ))
    if alpha_rows:
        write_table(ALPHA_ROOT / "outputs" / "g0.25_hub50" / "summary.csv", alpha_rows)


def recover_known_locals() -> Path:
    base_local = BASE / "charge_locally.json"
    alpha1 = alpha_out(1.0, "charge_locally.json")
    recover_local(
        SCREEN / "g0.25_hub55" / "alpha_1" / "charge_locally.json",
        0.25,
        [
            base_local,
            alpha1,
            EXP_ROOT / "exp_g" / "g_0.25" / "charge_locally.json",
            EXP_ROOT / "exp_cm" / "cm_45" / "charge_locally.json",
            EXP_ROOT / "exp_b" / "b0" / "charge_locally.json",
        ],
    )
    for g_value, folder in ((0.15, "g_0.15"), (0.2, "g_0.2"), (0.3, "g_0.3")):
        recover_local(
            SCREEN / f"g{alpha_tag(g_value)}_hub55" / "alpha_1" / "charge_locally.json",
            g_value,
            [EXP_ROOT / "exp_g" / folder / "charge_locally.json"],
        )
    recover_local(
        SCREEN / "g0.25_hub55" / "alpha_0" / "charge_locally.json",
        0.25,
        [alpha_out(0.0, "charge_locally.json")],
    )
    return base_local


def local_jobs(base_local: Path) -> None:
    warm_g = SCREEN / "g0.3_hub55" / "alpha_1" / "charge_locally.json"
    jobs = [
        ("locally g=0.35", FUTIAN, 0.35, 0.0, 45.0,
         EXP_ROOT / "exp_g" / "g_0.35" / "charge_locally.json", warm_g, []),
    ]
    for mobile_cost in (41, 43, 47, 49):
        jobs.append((
            f"locally cm={mobile_cost}", FUTIAN, 0.25, 0.0, float(mobile_cost),
            EXP_ROOT / "exp_cm" / f"cm_{mobile_cost}" / "charge_locally.json",
            base_local, [],
        ))
    jobs.append((
        "locally b=2", FUTIAN, 0.25, 2.0, 45.0,
        EXP_ROOT / "exp_b" / "b2" / "charge_locally.json", base_local, [],
    ))
    for alpha in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        jobs.append((
            f"locally alpha={alpha_tag(alpha)}", data_dir(alpha), 0.25, 0.0, 45.0,
            alpha_out(alpha, "charge_locally.json"), None, [],
        ))
    for case, data, g_value, penalty, mobile_cost, output, warm, copies in jobs:
        guarded(
            case,
            lambda data=data, g_value=g_value, penalty=penalty, mobile_cost=mobile_cost,
            output=output, warm=warm, copies=copies, case=case: solve_local(
                case, data, g_value, penalty, mobile_cost, output, warm, copies,
            ),
        )
        write_available_summaries()


def central_jobs() -> None:
    base_central = BASE / "charge_centrally.json"
    jobs = [(
        "centrally base", FUTIAN, 0.0, 45.0, 50.0, base_central,
        [
            alpha_out(1.0, "charge_centrally.json"),
            EXP_ROOT / "exp_cm" / "cm_45" / "charge_centrally.json",
            EXP_ROOT / "exp_hub" / "hub_50" / "charge_centrally.json",
            EXP_ROOT / "exp_b" / "b0" / "charge_centrally.json",
        ],
    )]
    for alpha in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        jobs.append((
            f"centrally alpha={alpha_tag(alpha)}", data_dir(alpha), 0.0, 45.0, 50.0,
            alpha_out(alpha, "charge_centrally.json"), [],
        ))
    for mobile_cost in (41, 43, 47, 49):
        jobs.append((
            f"centrally cm={mobile_cost}", FUTIAN, 0.0, float(mobile_cost), 50.0,
            EXP_ROOT / "exp_cm" / f"cm_{mobile_cost}" / "charge_centrally.json", [],
        ))
    for hub in (46, 48, 52, 54):
        jobs.append((
            f"centrally hub={hub}", FUTIAN, 0.0, 45.0, float(hub),
            EXP_ROOT / "exp_hub" / f"hub_{hub}" / "charge_centrally.json", [],
        ))
    jobs.append((
        "centrally b=2", FUTIAN, 2.0, 45.0, 50.0,
        EXP_ROOT / "exp_b" / "b2" / "charge_centrally.json", [],
    ))
    for case, data, penalty, mobile_cost, hub, output, copies in jobs:
        guarded(
            case,
            lambda data=data, penalty=penalty, mobile_cost=mobile_cost, hub=hub,
            output=output, copies=copies, case=case: solve_central(
                case, data, penalty, mobile_cost, hub, output, copies,
            ),
        )
        write_available_summaries()


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    base_local = recover_known_locals()
    prepare_alphas()
    reuse_fixed()
    for alpha in NEW_FIXED_ALPHAS:
        output = alpha_out(alpha, "fixed_only.json")
        guarded(
            f"fixed-only alpha={alpha_tag(alpha)} b=0",
            lambda alpha=alpha, output=output: solve_fixed(alpha, 0.0, output),
        )
    guarded(
        "fixed-only alpha=1 b=2",
        lambda: solve_fixed(1.0, 2.0, EXP_ROOT / "exp_b" / "b2" / "fixed_only.json"),
    )
    write_available_summaries()
    log("fixed-only 阶段结束，开始 centrally 与 locally")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(central_jobs), pool.submit(local_jobs, base_local)]
        for future in futures:
            future.result()
    write_available_summaries()
    log("全部排队案例已跑完。失败项见 progress.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise

"""福田 b=0 的错峰实验。

试参数时 locally 可以在墙钟用完时停，那个解不进最终图。
最终图只收步长 0.2 上已经收敛的 locally 和 centrally。
centrally 不依赖 g。gamma 固定 0.001。枢纽用 55。
低 alpha 两边可以都不派移动充电车；alpha=1 分出高低后，就在全部 alpha 上比较。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shift import DISPLAY_ALPHAS, OUTPUT_ROOT, alpha_tag, prepare_grid

CG_ROOT = Path("/Users/xingyueyi/Desktop/CG_solver")
PYTHON = Path(
    "/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/.venv-gurobi-13.0.3/bin/python"
)
RESULT_ROOT = Path(__file__).resolve().parent / "outputs"
EXISTING_CENTRAL_ALPHA1 = (
    CG_ROOT / "charge_centrally_solver_1030/outputs/futian_b0_adaptgap.json"
)
GAMMA = 0.001
HUB_BASE = 55.0
HUB_FLOOR = 50.0
# 试参数的 locally 墙钟。到点写入的是当前割下的上界，不是最终图用的最优解。
SCREEN_LOCAL_SECONDS = 3 * 3600
SCREEN_MARGIN = 30.0
FINAL_MARGIN = 15.0
THREADS = "4"


def _run(command: list[str], working_directory: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CG_THREADS"] = THREADS
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=working_directory, stdout=handle, stderr=subprocess.STDOUT,
            check=False, env=env,
        )
    if completed.returncode != 0:
        # 求解器有时在写出收敛结果之后仍以退出码 1 结束。结果文件能读就继续。
        saved = _load(log_path.with_suffix(".json"))
        if saved is None or saved.get("objective") is None:
            raise RuntimeError(f"求解失败，退出码 {completed.returncode}。日志：{log_path}")
        print(f"求解进程退出码 {completed.returncode}，但结果已写入，继续。日志：{log_path}", flush=True)


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _converged(saved: dict, mode: str) -> bool:
    if mode == "charge-locally":
        return saved.get("converged") is True and saved.get("reason") == "ALL_SEPARATIONS_WITHIN_TOLERANCE"
    if mode == "charge-centrally":
        return saved.get("status_name") == "OPTIMAL" and "没有新的割" in str(saved.get("stop_reason"))
    if mode == "fixed-only":
        return saved.get("converged") is True
    return False


def _numbers(saved: dict, mode: str) -> dict:
    objective = float(saved["objective"])
    bound = float(saved["bound"])
    if mode == "charge-locally":
        mobile = int(round(float(saved["decision"]["m"])))
        fixed = int(round(sum(float(value) for value in saved["decision"]["n"])))
        hubs = None
    elif mode == "charge-centrally":
        mobile = int(round(float(saved["mobile_chargers"])))
        fixed = int(round(float(saved["fixed_chargers"])))
        hubs = int(round(float(saved["hub_count"])))
    else:
        mobile = 0
        fixed = int(round(sum(float(value) for value in saved["decision"]["n"])))
        hubs = None
    return {
        "objective": objective,
        "bound": bound,
        "absolute_gap": bound - objective,
        "m_star": mobile,
        "fixed_chargers": fixed,
        "hub_count": hubs,
        "converged": _converged(saved, mode),
        "status": saved.get("status", saved.get("status_name")),
        "stop_reason": saved.get("reason", saved.get("stop_reason")),
    }


def _data_dir(alpha: float) -> Path:
    return OUTPUT_ROOT / f"alpha_{alpha_tag(alpha)}"


def _solve_central(alpha: float, hub: float, output: Path) -> None:
    if output.is_file() and _converged(_load(output), "charge-centrally"):
        print(f"沿用已收敛的 centrally alpha={alpha_tag(alpha)} hub={hub}", flush=True)
        return
    if alpha == 1.0 and hub == HUB_BASE and not output.exists():
        report = json.loads((_data_dir(alpha) / "shift_report.json").read_text(encoding="utf-8"))
        unchanged = not report["dropped_days"] and all(
            value is not None and value <= 1e-8 for value in report["sample_shift_max_abs"].values()
        )
        if unchanged and EXISTING_CENTRAL_ALPHA1.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(EXISTING_CENTRAL_ALPHA1, output)
            print("alpha=1 centrally 沿用福田 b=0 已收敛结果", flush=True)
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"求解 centrally alpha={alpha_tag(alpha)} hub={hub}", flush=True)
    _run(
        [
            str(PYTHON), "main.py",
            "--instance", "futian",
            "--data-dir", str(_data_dir(alpha)),
            "--shortage-penalty", "0",
            "--gamma", str(GAMMA),
            "--hub-open-cost", str(hub),
            "--adaptive-gap",
            "--max-total-seconds", "21600",
            "--output", str(output),
        ],
        CG_ROOT / "charge_centrally_solver_1030",
        output.with_suffix(".log"),
    )


def _converged_local_with_same_g(alpha: float, g_value: float) -> Path | None:
    # 枢纽开设成本不进入 charge locally，同一 g 的收敛解可以跨枢纽成本沿用。
    pattern = f"screen/g{alpha_tag(g_value)}_hub*/alpha_{alpha_tag(alpha)}/charge_locally.json"
    for candidate in RESULT_ROOT.glob(pattern):
        saved = _load(candidate)
        if saved is not None and _converged(saved, "charge-locally"):
            return candidate
    return None


def _solve_local(alpha: float, g_value: float, output: Path, seconds: float | None) -> None:
    if output.is_file() and _converged(_load(output), "charge-locally"):
        print(f"沿用已收敛的 locally alpha={alpha_tag(alpha)} g={g_value}", flush=True)
        return
    reused = _converged_local_with_same_g(alpha, g_value)
    if reused is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if reused.resolve() != output.resolve():
            shutil.copy2(reused, output)
        print(f"沿用已收敛的 locally alpha={alpha_tag(alpha)} g={g_value}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), "run_20zone.py",
        "--data", str(_data_dir(alpha)),
        "--g", str(g_value),
        "--shortage-penalty", "0",
        "--gamma", str(GAMMA),
        "--adaptive-gap",
        "--output", str(output),
        "--solve",
    ]
    if output.is_file() and _load(output).get("active_patterns"):
        command.extend(["--resume", str(output)])
    if seconds is not None:
        command.extend(["--seconds", str(seconds)])
    label = "试参数" if seconds is not None else "收敛"
    print(f"{label} locally alpha={alpha_tag(alpha)} g={g_value}", flush=True)
    _run(command, CG_ROOT / "charge_locally_solver", output.with_suffix(".log"))


def _solve_fixed(alpha: float, output: Path) -> None:
    if alpha == 1.0 and not output.exists():
        existing = CG_ROOT / "fixed_only_solver/outputs/futian_b0.json"
        report = json.loads((_data_dir(alpha) / "shift_report.json").read_text(encoding="utf-8"))
        unchanged = not report["dropped_days"] and all(
            value is not None and value <= 1e-8 for value in report["sample_shift_max_abs"].values()
        )
        if unchanged and existing.is_file() and _converged(_load(existing), "fixed-only"):
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, output)
            print("alpha=1 fixed-only 沿用福田 b=0 已收敛结果", flush=True)
            return
    if output.is_file() and _converged(_load(output), "fixed-only"):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"求解 fixed-only alpha={alpha_tag(alpha)}", flush=True)
    _run(
        [
            str(PYTHON), "main.py",
            "--data", str(_data_dir(alpha)),
            "--shortage-penalty", "0",
            "--output", str(output),
            "--solve",
        ],
        CG_ROOT / "fixed_only_solver",
        output.with_suffix(".log"),
    )


def _screen_dir(g_value: float, hub: float) -> Path:
    return RESULT_ROOT / "screen" / f"g{alpha_tag(g_value)}_hub{alpha_tag(hub)}"


def _central_path(alpha: float, hub: float) -> Path:
    return RESULT_ROOT / "screen" / f"hub{alpha_tag(hub)}" / f"alpha_{alpha_tag(alpha)}" / "charge_centrally.json"


def _local_screen_path(alpha: float, g_value: float, hub: float) -> Path:
    return _screen_dir(g_value, hub) / f"alpha_{alpha_tag(alpha)}" / "charge_locally.json"


def _diff_sign(local_saved: dict, central_saved: dict, margin: float) -> int:
    local_objective = float(local_saved["objective"])
    local_bound = float(local_saved["bound"])
    central_objective = float(central_saved["objective"])
    central_bound = float(central_saved["bound"])
    if _converged(local_saved, "charge-locally") and _converged(central_saved, "charge-centrally"):
        if local_objective > central_bound:
            return 1
        if central_objective > local_bound:
            return -1
        return 0
    diff = local_objective - central_objective
    if abs(diff) <= margin:
        return 0
    return 1 if diff > 0 else -1


def _screen_once(g_value: float, hub: float) -> dict:
    # centrally 不吃 g，alpha=1 在枢纽 55 时直接复用已有收敛解。
    # 两个长任务并行：alpha=0 的 centrally，以及 alpha=1 的 locally 试探。
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(_solve_central, 0.0, hub, _central_path(0.0, hub)),
            pool.submit(
                _solve_local, 1.0, g_value,
                _local_screen_path(1.0, g_value, hub), SCREEN_LOCAL_SECONDS,
            ),
        ]
        for job in jobs:
            job.result()
    _solve_central(1.0, hub, _central_path(1.0, hub))
    _solve_local(0.0, g_value, _local_screen_path(0.0, g_value, hub), SCREEN_LOCAL_SECONDS)
    signs = {}
    detail = {"g": g_value, "hub": hub, "gamma": GAMMA, "endpoints": {}}
    for alpha in (0.0, 1.0):
        local_saved = _load(_local_screen_path(alpha, g_value, hub))
        central_saved = _load(_central_path(alpha, hub))
        sign = _diff_sign(local_saved, central_saved, SCREEN_MARGIN)
        signs[alpha] = sign
        detail["endpoints"][alpha_tag(alpha)] = {
            "sign_local_minus_central": sign,
            "local_objective": local_saved.get("objective"),
            "local_converged": _converged(local_saved, "charge-locally"),
            "central_objective": central_saved.get("objective"),
            "central_converged": _converged(central_saved, "charge-centrally"),
            "difference": float(local_saved["objective"]) - float(central_saved["objective"]),
        }
    # 低 alpha 两边可以都不派移动充电车。只要 alpha=1 已经分出高低，就锁定参数，
    # 再到 0.2、0.4、0.6、0.8 上看移动充电车开始使用之后谁占优。
    if signs[1.0] != 0:
        detail["action"] = "lock"
    elif signs[0.0] == 0 or signs[1.0] == 0:
        detail["action"] = "inconclusive"
    elif signs[0.0] > 0:
        detail["action"] = "increase_g"
    else:
        detail["action"] = "decrease_g"
    return detail


def _next_setting(g_value: float, hub: float, action: str) -> tuple[float, float] | None:
    if action == "increase_g":
        if g_value < 0.4 - 1e-9:
            return min(g_value + 0.1, 0.4), hub
        if hub > HUB_FLOOR + 1e-9:
            return g_value, HUB_FLOOR
        return None
    if action == "decrease_g":
        if g_value > 0.2 - 1e-9:
            return 0.15, hub
        if g_value > 0.1 + 1e-9:
            return 0.1, hub
        if hub > HUB_FLOOR + 1e-9:
            return 0.2, HUB_FLOOR
        return None
    return None


def _final_dir(g_value: float, hub: float) -> Path:
    return RESULT_ROOT / "final" / f"g{alpha_tag(g_value)}_hub{alpha_tag(hub)}"


def _row(alpha: float, mode: str, saved: dict, g_value: float, hub: float) -> dict:
    numbers = _numbers(saved, mode)
    return {
        "alpha": alpha,
        "mode": mode,
        "g": g_value if mode == "charge-locally" else None,
        "gamma": GAMMA if mode != "fixed-only" else None,
        "hub_open_cost": hub if mode == "charge-centrally" else None,
        **numbers,
    }


def _write_table(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "alpha", "mode", "g", "gamma", "hub_open_cost", "objective", "bound", "absolute_gap",
        "m_star", "fixed_chargers", "hub_count", "converged", "status", "stop_reason",
    ]
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], directory: Path) -> None:
    import matplotlib.pyplot as plt

    local = {row["alpha"]: row for row in rows if row["mode"] == "charge-locally"}
    central = {row["alpha"]: row for row in rows if row["mode"] == "charge-centrally"}
    alphas = list(DISPLAY_ALPHAS)
    if any(alpha not in local or alpha not in central for alpha in alphas):
        raise RuntimeError("最终图缺少某个 alpha")
    if any(not local[alpha]["converged"] or not central[alpha]["converged"] for alpha in alphas):
        raise RuntimeError("最终图只使用收敛解")
    diff = [local[alpha]["objective"] - central[alpha]["objective"] for alpha in alphas]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.plot(alphas, diff, marker="o")
    axis.set_xticks(list(alphas))
    axis.set_xlabel("alpha")
    axis.set_ylabel("locally objective − centrally objective")
    figure.tight_layout()
    figure.savefig(directory / "local_minus_central.png", dpi=160)
    figure.savefig(directory / "local_minus_central.pdf")
    plt.close(figure)


def _run_final(g_value: float, hub: float) -> None:
    directory = _final_dir(g_value, hub)
    rows = []
    for alpha in DISPLAY_ALPHAS:
        central_out = directory / f"alpha_{alpha_tag(alpha)}" / "charge_centrally.json"
        screened_central = _central_path(alpha, hub)
        if alpha in (0.0, 1.0) and screened_central.is_file() and _converged(_load(screened_central), "charge-centrally"):
            central_out.parent.mkdir(parents=True, exist_ok=True)
            if not central_out.exists():
                shutil.copy2(screened_central, central_out)
        _solve_central(alpha, hub, central_out)
        local_out = directory / f"alpha_{alpha_tag(alpha)}" / "charge_locally.json"
        screened_local = _local_screen_path(alpha, g_value, hub)
        if screened_local.is_file() and not local_out.exists():
            local_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(screened_local, local_out)
        _solve_local(alpha, g_value, local_out, None)
        fixed_out = directory / f"alpha_{alpha_tag(alpha)}" / "fixed_only.json"
        _solve_fixed(alpha, fixed_out)
        rows.extend([
            _row(alpha, "charge-centrally", _load(central_out), g_value, hub),
            _row(alpha, "charge-locally", _load(local_out), g_value, hub),
            _row(alpha, "fixed-only", _load(fixed_out), g_value, hub),
        ])
        _write_table(rows, directory)
    if any(not row["converged"] for row in rows if row["mode"] != "fixed-only"):
        raise RuntimeError(f"最终网格有未收敛的解，不画图：{directory / 'summary.csv'}")
    _plot(rows, directory)
    print(f"最终图已写入 {directory / 'local_minus_central.png'}", flush=True)


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"没有找到 Gurobi 环境的 Python：{PYTHON}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = prepare_grid(DISPLAY_ALPHAS)
    (RESULT_ROOT / "shift_reports.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    # g=0.3 在 alpha=1 上 locally 更优。alpha=0 两边都不派车，不因此改参数。
    g_value, hub = 0.3, HUB_BASE
    history = []
    while True:
        detail = _screen_once(g_value, hub)
        history.append(detail)
        (RESULT_ROOT / "screen_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(json.dumps(detail, ensure_ascii=False), flush=True)
        if detail["action"] == "lock":
            _run_final(g_value, hub)
            return
        if detail["action"] == "inconclusive":
            raise RuntimeError("alpha=1 上两边的认证区间重叠，还不能锁定 g。")
        nxt = _next_setting(g_value, hub, detail["action"])
        if nxt is None:
            raise RuntimeError("g 到 0.4、枢纽到 50 仍不能让两端异号，停止，不画最终图。")
        g_value, hub = nxt


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise

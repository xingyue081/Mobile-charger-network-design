"""运行割平面，并把部署和按分离核算的 gap 补进结果。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from deployment import labeled_deployment
from settings import LOCAL_G, OUTPUT_ROOT, PYTHON, SOLVER_DIR, WALL_SECONDS, ZONE_COUNTS
from settings import DATA_ROOT


def _accounted_bound(summary: dict) -> tuple[float | None, float | None]:
    upper = summary.get("bound")
    lower = None
    round_log = summary.get("round_log")
    if round_log and Path(round_log).is_file():
        last = None
        for line in Path(round_log).read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = json.loads(line)
        if last and last.get("separation_finished") and last.get("accounted_objective") is not None:
            lower = float(last["accounted_objective"])
    if lower is None and summary.get("status_name") == "OPTIMAL" and "没有新的割" in str(summary.get("stop_reason")):
        lower = summary.get("objective")
    return None if upper is None else float(upper), None if lower is None else float(lower)


def run_cutting_plane(k: int, wall_seconds: float = WALL_SECONDS) -> dict:
    output_dir = OUTPUT_ROOT / f"k{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "cutting_plane.json"
    command = [
        str(PYTHON),
        "main.py",
        "--instance", "futian",
        "--data-dir", str(DATA_ROOT / f"k{k}"),
        "--rho", "2",
        "--shortage-penalty", "0",
        "--fixed-cost", "41",
        "--mobile-cost", "45",
        "--gamma", "0.001",
        "--hub-open-cost", "50",
        "--adaptive-gap",
        "--max-iterations", "200",
        "--max-total-seconds", str(wall_seconds),
        "--output", str(destination),
    ]
    env = os.environ.copy()
    env["CG_THREADS"] = "4"
    started = time.monotonic()
    completed = subprocess.run(command, cwd=SOLVER_DIR, env=env, check=False)
    payload = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {
        "status_name": "FAILED",
        "stop_reason": f"求解进程退出码 {completed.returncode}",
        "objective": None,
        "bound": None,
        "decision": None,
    }
    payload["method"] = "cutting_plane"
    payload["k"] = k
    payload["returncode"] = completed.returncode
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["local_g_not_used"] = LOCAL_G
    payload["converged"] = (
        payload.get("status_name") == "OPTIMAL"
        and "没有新的割" in str(payload.get("stop_reason"))
    )
    upper, lower = _accounted_bound(payload)
    payload["bound"] = upper
    payload["accounted_objective"] = lower
    payload["gap"] = None if upper in (None, 0.0) or lower is None else (upper - lower) / abs(upper)
    if payload.get("decision"):
        payload["deployment"] = labeled_deployment(payload["zone_ids"], payload["decision"])
    else:
        payload["deployment"] = None
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"k={k} cg status={payload.get('status_name')} obj={payload.get('objective')} "
        f"gap={payload.get('gap')} elapsed={payload['elapsed_seconds']:.1f}",
        flush=True,
    )
    return payload


def main() -> None:
    for k in ZONE_COUNTS:
        run_cutting_plane(k)


if __name__ == "__main__":
    main()

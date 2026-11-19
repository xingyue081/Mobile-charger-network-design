"""按统一口径汇总全量模型和割平面。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from settings import FULL_MODEL_COUNTS, OUTPUT_ROOT, ZONE_COUNTS


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _added_cuts(payload: dict | None) -> tuple[int | None, int | None]:
    """割平面记分离加入的割，以及最后一轮的不同 pattern 数。"""
    if not payload:
        return None, None
    round_log = payload.get("round_log")
    if round_log and Path(round_log).is_file():
        added = 0
        unique = None
        for line in Path(round_log).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            added += int(record.get("cuts_added") or 0)
            if record.get("unique_patterns") is not None:
                unique = int(record["unique_patterns"])
        return added, unique
    if payload.get("heights_built") is not None:
        return int(payload["heights_built"]), payload.get("pattern_count_per_sample")
    return None, None


def write_summary() -> None:
    rows = []
    full_by_k = {}
    for k in ZONE_COUNTS:
        full = _read(OUTPUT_ROOT / f"k{k}" / "full_milp.json")
        cg = _read(OUTPUT_ROOT / f"k{k}" / "cutting_plane.json")
        full_by_k[k] = full
        if k in FULL_MODEL_COUNTS:
            rows.append(_row(k, "full_pattern_milp", full, None))
        else:
            rows.append(_row(k, "full_pattern_milp", None, None, not_built=True))
        rows.append(_row(k, "cutting_plane", cg, full if k in FULL_MODEL_COUNTS else None))
    destination = OUTPUT_ROOT / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {destination}", flush=True)


def _samples(k: int) -> dict:
    metadata = _read(OUTPUT_ROOT.parent / "data" / f"k{k}" / "metadata.json") or {}
    counts = metadata.get("sample_counts") or {}
    return {
        "workday_samples": counts.get("workday", ""),
        "weekend_samples": counts.get("weekend", ""),
        "holiday_samples": counts.get("holiday", ""),
    }


def _row(k: int, method: str, payload: dict | None, full: dict | None, not_built: bool = False) -> dict:
    cuts, unique = _added_cuts(payload)
    objective = None if payload is None else payload.get("objective")
    full_objective = None if full is None else full.get("objective")
    difference = ""
    relative = ""
    same = ""
    if objective is not None and full_objective is not None:
        difference = float(objective) - float(full_objective)
        scale = abs(float(full_objective))
        relative = "" if scale == 0.0 else abs(difference) / scale
        same = bool(payload.get("converged") and full.get("converged") and relative != "" and relative <= 1e-4)
    row = {
        "k": k,
        "method": method,
        "status_name": "MODEL_NOT_BUILT" if not_built else ("" if payload is None else payload.get("status_name")),
        "stop_reason": "模型未能建立" if not_built else ("" if payload is None else payload.get("stop_reason")),
        "objective": "" if objective is None else objective,
        "bound": "" if payload is None or payload.get("bound") is None else payload.get("bound"),
        "gap": "" if payload is None or payload.get("gap") is None else payload.get("gap"),
        "converged": False if payload is None else bool(payload.get("converged")),
        "h_seconds": "" if payload is None or payload.get("h_seconds") is None else payload.get("h_seconds"),
        "solve_seconds": "" if payload is None or payload.get("solve_seconds") is None else payload.get("solve_seconds"),
        "elapsed_seconds": "" if payload is None or payload.get("elapsed_seconds") is None else payload.get("elapsed_seconds"),
        "iterations": "" if payload is None or payload.get("iterations") is None else payload.get("iterations"),
        "cuts": "" if cuts is None else cuts,
        "unique_patterns": "" if unique is None else unique,
        "objective_minus_full": difference,
        "relative_objective_gap": relative,
        "same_optimum": same,
    }
    row.update(_samples(k))
    return row


if __name__ == "__main__":
    write_summary()

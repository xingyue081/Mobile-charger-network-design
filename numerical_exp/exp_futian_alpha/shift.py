"""把福田 support 样本按错峰强度 alpha 平移，并丢掉移出椭球的整天。

alpha 只改变各 zone 在三种 date type 上的均值如何错开：

    mu_il(alpha) = mu_i + common_l + alpha * e_il

展示用的 alpha 是 0, 0.2, 0.4, 0.6, 0.8, 1。
Sigma、r、epsilon_l、q_l、距离和 business.json 都不重新估计。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

SOURCE_DIR = Path("/Users/xingyueyi/Desktop/CG_solver/futian")
OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
DATE_TYPES = ("workday", "weekend", "holiday")
DISPLAY_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ELLIPSOID_TOLERANCE = 1e-6
EXPECTED_BUSINESS = {
    "rho": 2.0,
    "shortage_penalty": 2.0,
    "fixed_capacity": 190.0,
    "mobile_capacity": 160.0,
    "fixed_unit_cost": 190.0,
    "mobile_unit_cost": 125.0,
    "transport_cost_per_distance": 0.001,
    "hub_open_cost": 900.0,
}


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):.10f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def load_source(source_dir: Path = SOURCE_DIR) -> dict:
    stored_path = source_dir / "futian_parameters.npz"
    with np.load(stored_path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    samples = {name: arrays[f"samples_{name}"].astype(float) for name in DATE_TYPES}
    probability = arrays["q_l"].astype(float)
    if probability.shape != (3,):
        raise ValueError("q_l 必须是三种 date type")
    if abs(float(probability.sum()) - 1.0) > 1e-8:
        raise ValueError("q_l 之和必须为 1")
    return {
        "arrays": arrays,
        "samples": samples,
        "probability": probability,
        "mean": arrays["mu"].astype(float),
        "precision": arrays["Sigma_inv"].astype(float),
        "radius_squared": float(arrays["r_squared"]),
    }


def check_business(source_dir: Path = SOURCE_DIR) -> None:
    stored = json.loads((source_dir / "business.json").read_text(encoding="utf-8"))
    mismatches = {
        field: (stored.get(field), expected)
        for field, expected in EXPECTED_BUSINESS.items()
        if float(stored.get(field)) != expected
    }
    if mismatches:
        raise ValueError(f"business.json 与福田原始成本不一致：{mismatches}")


def stagger_components(samples: dict[str, np.ndarray], probability: np.ndarray) -> dict:
    date_type_mean = np.stack([samples[name].mean(axis=0) for name in DATE_TYPES], axis=0)
    zone_mean = probability @ date_type_mean
    deviation = date_type_mean - zone_mean
    common = deviation.mean(axis=1)
    idiosyncratic = deviation - common[:, None]
    if np.max(np.abs(idiosyncratic.sum(axis=1))) > 1e-8:
        raise RuntimeError("错峰项在 zone 之间相加不为 0")
    if np.max(np.abs(probability @ idiosyncratic)) > 1e-8:
        raise RuntimeError("错峰项按 q_l 加权后不为 0")
    return {
        "date_type_mean": date_type_mean,
        "zone_mean": zone_mean,
        "common": common,
        "idiosyncratic": idiosyncratic,
    }


def shift_samples(samples: dict[str, np.ndarray], components: dict, alpha: float) -> dict[str, np.ndarray]:
    shifted = {}
    for index, name in enumerate(DATE_TYPES):
        adjustment = (float(alpha) - 1.0) * components["idiosyncratic"][index]
        shifted[name] = samples[name] + adjustment
    return shifted


def outside_ellipsoid(samples: dict[str, np.ndarray], source: dict) -> list[dict]:
    dropped = []
    limit = source["radius_squared"] + ELLIPSOID_TOLERANCE
    for name in DATE_TYPES:
        centered = samples[name] - source["mean"]
        quadratic = np.einsum("ni,ij,nj->n", centered, source["precision"], centered)
        for row_index, value in enumerate(quadratic):
            if float(value) > limit:
                dropped.append({
                    "date_type": name,
                    "row_index": int(row_index),
                    "mahalanobis_squared": float(value),
                    "excess_over_r_squared": float(value - source["radius_squared"]),
                })
    return dropped


def keep_inside(samples: dict[str, np.ndarray], source: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    dropped = outside_ellipsoid(samples, source)
    dropped_rows = {}
    for item in dropped:
        dropped_rows.setdefault(item["date_type"], set()).add(item["row_index"])
    kept = {}
    for name in DATE_TYPES:
        remove = dropped_rows.get(name, set())
        kept_indexes = [index for index in range(len(samples[name])) if index not in remove]
        if not kept_indexes:
            raise RuntimeError(f"{name} 平移后没有样本留在椭球内")
        kept[name] = samples[name][kept_indexes]
    return kept, dropped


def _margin_errors(samples: dict[str, np.ndarray], components: dict, alpha: float, probability: np.ndarray) -> dict:
    target_mean = np.stack([
        components["zone_mean"] + components["common"][index] + float(alpha) * components["idiosyncratic"][index]
        for index in range(3)
    ], axis=0)
    actual_mean = np.stack([samples[name].mean(axis=0) for name in DATE_TYPES], axis=0)
    return {
        "max_abs_date_type_total_error": float(np.max(np.abs(actual_mean.sum(axis=1) - target_mean.sum(axis=1)))),
        "max_abs_zone_mean_error": float(np.max(np.abs(probability @ actual_mean - components["zone_mean"]))),
    }


def write_alpha(alpha: float, source_dir: Path = SOURCE_DIR, output_root: Path = OUTPUT_ROOT) -> dict:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha 只取 [0, 1]")
    check_business(source_dir)
    source = load_source(source_dir)
    components = stagger_components(source["samples"], source["probability"])
    shifted = shift_samples(source["samples"], components, alpha)
    before_drop = _margin_errors(shifted, components, alpha, source["probability"])
    kept, dropped = keep_inside(shifted, source)
    # 福田的错峰项比单日低需求大，alpha<1 时少数 zone 会变成负的。
    # 负的部分占日需求不到千分之一，截到 0 后仍落在原椭球里。alpha=1 不发生截断。
    clipped = {}
    for name, rows in kept.items():
        negative = np.minimum(rows, 0.0)
        clipped[name] = {
            "entries": int(np.sum(rows < -1e-8)),
            "sum_kwh": float(negative.sum()),
            "min_kwh": float(rows.min()),
        }
        kept[name] = np.maximum(rows, 0.0)

    destination = output_root / f"alpha_{alpha_tag(alpha)}"
    destination.mkdir(parents=True, exist_ok=True)
    arrays = dict(source["arrays"])
    for name in DATE_TYPES:
        arrays[f"samples_{name}"] = kept[name]
    np.savez(destination / "futian_parameters.npz", **arrays)
    shutil.copy2(source_dir / "business.json", destination / "business.json")
    shutil.copy2(source_dir / "futian_metadata.json", destination / "futian_metadata.json")
    report = {
        "alpha": float(alpha),
        "source": str(source_dir / "futian_parameters.npz"),
        "common_shift_kwh": {
            name: float(components["common"][index]) for index, name in enumerate(DATE_TYPES)
        },
        "original_counts": {name: int(len(source["samples"][name])) for name in DATE_TYPES},
        "kept_counts": {name: int(len(kept[name])) for name in DATE_TYPES},
        "dropped_days": dropped,
        "clipped_negative_demand": clipped,
        "sample_shift_max_abs": {
            name: float(np.max(np.abs(kept[name] - source["samples"][name][: len(kept[name])])))
            if len(kept[name]) == len(source["samples"][name]) else None
            for name in DATE_TYPES
        },
        "margins_before_dropping_days": before_drop,
        "margins_after_dropping_days": _margin_errors(kept, components, alpha, source["probability"]),
        "unchanged": [
            "mu", "Sigma", "Sigma_inv", "r", "r_squared", "epsilon_l", "q_l",
            "distance", "n_bar", "taz_ids", "business.json",
        ],
    }
    (destination / "shift_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return report


def prepare_grid(alphas: tuple[float, ...] = DISPLAY_ALPHAS) -> list[dict]:
    return [write_alpha(alpha) for alpha in alphas]


if __name__ == "__main__":
    for item in prepare_grid():
        print(json.dumps({
            "alpha": item["alpha"],
            "kept_counts": item["kept_counts"],
            "dropped": len(item["dropped_days"]),
        }, ensure_ascii=False))

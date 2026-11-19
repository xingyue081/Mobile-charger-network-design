"""按福田 taz 顺序生成嵌套子集，并重算半径和 epsilon。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from settings import BUSINESS, DATA_ROOT, DATE_TYPES, FUTIAN_NPZ, LOCAL_G, TRAIN_Q, ZONE_COUNTS

DEMAND_CSV = Path("/Users/xingyueyi/Downloads/论文/models/demand_samples_by_taz_date_type.csv")
TRAIN_START = "2022-09-01"
TRAIN_END_EXCLUSIVE = "2023-01-01"
BOOTSTRAP_REPLICATIONS = 1000
BOOTSTRAP_SEED = 20260907


def _mahalanobis_squared(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(covariance)
    diff = points - mean
    return np.einsum("ij,jk,ik->i", diff, precision, diff)


def _calibrate(values: np.ndarray, labels: list[str]) -> dict:
    """与 futian/build_futian_parameters.py 的标定相同，避免导入 geopandas。"""
    count, dimension = values.shape
    if dimension < 2 or count <= dimension:
        raise RuntimeError(f"样本不足以估计 {dimension} 维协方差")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0:
        raise RuntimeError("样本协方差不是正定的")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    radius_draws = np.empty(BOOTSTRAP_REPLICATIONS, dtype=float)
    for _replication in range(BOOTSTRAP_REPLICATIONS):
        draw = values[generator.integers(0, count, size=count)]
        draw_mean = draw.mean(axis=0)
        draw_covariance = np.cov(draw, rowvar=False, ddof=1)
        distances = _mahalanobis_squared(draw, draw_mean, draw_covariance)
        radius_draws[_replication] = float(np.quantile(distances, 0.95, method="linear"))
    radius_squared = float(np.quantile(radius_draws, 0.90, method="linear"))
    in_support = _mahalanobis_squared(values, mean, covariance) <= radius_squared
    grouped = {label: [] for label in DATE_TYPES}
    for row_index, label in enumerate(labels):
        if in_support[row_index]:
            grouped[label].append(values[row_index].copy())
    for label in DATE_TYPES:
        if len(grouped[label]) < 2:
            raise RuntimeError(f"{label} 落在椭球内的样本少于 2 天")
    grouped = {label: np.vstack(rows) for label, rows in grouped.items()}
    epsilon = []
    for label in DATE_TYPES:
        sample = grouped[label]
        sigma_l = float(np.median(sample.std(axis=0, ddof=1)))
        sample_count = len(sample)
        epsilon.append(sigma_l * sample_count ** (1.0 / dimension) * sample_count ** (-1.0 / dimension))
    label_counts = pd.Series(labels).value_counts().to_dict()
    q_l = np.array([label_counts[label] / count for label in DATE_TYPES])
    return {
        "mu": mean,
        "Sigma": covariance,
        "r_squared": radius_squared,
        "r": float(np.sqrt(radius_squared)),
        "samples": grouped,
        "epsilon_l": np.array(epsilon),
        "q_l": q_l,
    }


def _training_panel(taz_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(DEMAND_CSV)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= TRAIN_START) & (raw["date"] < TRAIN_END_EXCLUSIVE)]
    raw = raw[raw["taz_id"].isin(taz_ids)]
    panel = raw.pivot(index="date", columns="taz_id", values="daily_volume").sort_index()
    panel = panel.reindex(columns=taz_ids)
    if panel.isna().any().any():
        raise RuntimeError("训练窗内有 zone 缺日需求")
    labels = (
        raw.drop_duplicates("date").set_index("date").loc[panel.index, "date_type"].tolist()
    )
    return panel, labels


def _check_probability(q_values: np.ndarray) -> None:
    if q_values.shape != (3,) or not np.allclose(q_values, TRAIN_Q, rtol=0.0, atol=1e-12):
        raise SystemExit(f"q_l 不是训练窗日历 {TRAIN_Q}，实际 {q_values.tolist()}")


def build_one(taz_ids: list[int], source: np.lib.npyio.NpzFile, k: int) -> dict:
    panel, labels = _training_panel(taz_ids)
    calibrated = _calibrate(panel.to_numpy(dtype=float), labels)
    _check_probability(calibrated["q_l"])
    eigenvalues = np.linalg.eigvalsh(calibrated["Sigma"])
    if float(eigenvalues.min()) <= 0.0:
        raise SystemExit(f"k={k} 的协方差不是正定的")
    distance = source["distance"].astype(float)[:k, :k]
    n_bar = source["n_bar"].astype(float)[:k]
    destination = DATA_ROOT / f"k{k}"
    destination.mkdir(parents=True, exist_ok=True)
    samples = calibrated["samples"]
    np.savez(
        destination / "futian_parameters.npz",
        taz_ids=np.array(taz_ids, dtype=int),
        mu=calibrated["mu"],
        Sigma=calibrated["Sigma"],
        Sigma_inv=np.linalg.inv(calibrated["Sigma"]),
        r=np.array(calibrated["r"]),
        r_squared=np.array(calibrated["r_squared"]),
        epsilon_l=calibrated["epsilon_l"],
        q_l=calibrated["q_l"],
        n_bar=n_bar,
        distance=distance,
        samples_workday=samples["workday"],
        samples_weekend=samples["weekend"],
        samples_holiday=samples["holiday"],
    )
    (destination / "business.json").write_text(
        json.dumps(BUSINESS, indent=2) + "\n", encoding="utf-8",
    )
    report = {
        "k": k,
        "taz_ids": taz_ids,
        "min_eigenvalue": float(eigenvalues.min()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "r": float(calibrated["r"]),
        "r_squared": float(calibrated["r_squared"]),
        "epsilon_l": {
            name: float(value)
            for name, value in zip(DATE_TYPES, calibrated["epsilon_l"])
        },
        "q_l": {
            name: float(value)
            for name, value in zip(DATE_TYPES, calibrated["q_l"])
        },
        "sample_counts": {
            name: int(samples[name].shape[0]) for name in DATE_TYPES
        },
        "local_g_not_used_by_central": LOCAL_G,
        "parameters": BUSINESS,
    }
    (destination / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(
        f"k={k} r={report['r']:.6g} eps={report['epsilon_l']} samples={report['sample_counts']}",
        flush=True,
    )
    return report


def check_full_reproduction(report: dict, source: np.lib.npyio.NpzFile) -> None:
    if report["k"] != int(source["taz_ids"].shape[0]):
        return
    mismatches = []
    if not np.allclose(report["r"], float(source["r"]), rtol=0.0, atol=1e-8):
        mismatches.append(f"r {report['r']} vs {float(source['r'])}")
    stored_eps = source["epsilon_l"].astype(float)
    fresh_eps = np.array([report["epsilon_l"][name] for name in DATE_TYPES])
    if not np.allclose(fresh_eps, stored_eps, rtol=0.0, atol=1e-8):
        mismatches.append(f"epsilon {fresh_eps.tolist()} vs {stored_eps.tolist()}")
    stored_counts = {
        "workday": int(source["samples_workday"].shape[0]),
        "weekend": int(source["samples_weekend"].shape[0]),
        "holiday": int(source["samples_holiday"].shape[0]),
    }
    if report["sample_counts"] != stored_counts:
        mismatches.append(f"样本数 {report['sample_counts']} vs {stored_counts}")
    with np.load(DATA_ROOT / "k35" / "futian_parameters.npz") as fresh:
        if not np.allclose(fresh["Sigma"], source["Sigma"], rtol=0.0, atol=1e-6):
            mismatches.append("Sigma 与原福田数据包不一致")
    if mismatches:
        raise SystemExit("k=35 没有复现福田数据包：" + "; ".join(mismatches))
    print("k=35 复现了原福田的 Sigma、r、epsilon 和样本数", flush=True)


def main() -> None:
    with np.load(FUTIAN_NPZ) as source:
        taz_ids = [int(value) for value in source["taz_ids"].tolist()]
        reports = []
        for k in ZONE_COUNTS:
            reports.append(build_one(taz_ids[:k], source, k))
        check_full_reproduction(reports[-1], source)
    print(f"已生成 {len(reports)} 个子集", flush=True)


if __name__ == "__main__":
    main()

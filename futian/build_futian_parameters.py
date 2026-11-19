"""把福田区训练样本收成三个求解器共用的 npz。

有效 zone 是代表点落在福田区、并且日需求表和固定桩表都有记录的 TAZ。
这两张表是 20 区、30 区用的同一套来源，福田对下来是 35 个。
均值、协方差、椭球半径和 epsilon_l 按 30 区脚本里的 20 区规则，
在 2022-09-01 至 2023-01-01 的全部训练日上重新估计。
固定桩上界是现有固定桩数量的两倍。业务成本在本目录的 business.json。
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.ops import transform

ROOT = Path(__file__).resolve().parent
DATE_TYPES = ("workday", "weekend", "holiday")
BOOTSTRAP_REPLICATIONS = 1000
BOOTSTRAP_SEED = 20260907
TRAIN_START = "2022-09-01"
TRAIN_END_EXCLUSIVE = "2023-01-01"
DEMAND_CSV = Path("/Users/xingyueyi/Downloads/论文/models/demand_samples_by_taz_date_type.csv")
DISTANCE_CSV = Path(
    "/Users/xingyueyi/Documents/UrbanEV_demand_for_PMCS/UrbanEV/distance_output/distance_matrix_clean.csv"
)
CHARGER_COUNTS = Path(
    "/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/taz_fixed_charger_counts_from_inf.csv"
)
TAZ_POLYGON = Path("/Users/xingyueyi/Downloads/ST-EVCDP-main/datasets/SZ_districts/SZ_districts.shp")
DISTRICT_GEOJSON = ROOT / "sz_440300_full.json"
DISTRICT_GEOJSON_URL = "https://geo.datav.aliyun.com/areas_v3/bound/440300_full.json"

# WGS84 -> GCJ-02。TAZ 多边形是 WGS84，行政区边界是 GCJ-02。
_A = 6378245.0
_EE = 0.00669342162296594323


def _transform_lat(lon: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
    return lon + dlon, lat + dlat


def _ensure_district_geojson() -> None:
    if DISTRICT_GEOJSON.is_file():
        return
    urllib.request.urlretrieve(DISTRICT_GEOJSON_URL, DISTRICT_GEOJSON)


def select_futian_taz() -> tuple[list[int], list[int], list[int], list[int]]:
    """代表点落在福田，且需求表和固定桩表都有记录的 TAZ，按编号升序。

    另外返回：有需求但没有固定桩的编号，以及有固定桩但没有需求的编号。
    """
    _ensure_district_geojson()
    with DISTRICT_GEOJSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    futian_features = [feat for feat in payload["features"] if feat["properties"]["name"] == "福田区"]
    if len(futian_features) != 1:
        raise RuntimeError("行政区边界里没有唯一的福田区")
    futian = gpd.GeoDataFrame(
        [{"geometry": shape(futian_features[0]["geometry"])}],
        crs="EPSG:4326",
    ).to_crs(3857)

    taz = gpd.read_file(TAZ_POLYGON).to_crs(4326)
    taz["taz_id"] = taz["TAZID"].astype(int)
    taz["geometry"] = taz.geometry.apply(
        lambda geom: transform(lambda x, y, z=None: _wgs84_to_gcj02(x, y), geom)
    )
    taz = taz.to_crs(3857)
    points = taz.copy()
    points["geometry"] = points.geometry.representative_point()
    inside = gpd.sjoin(points, futian, predicate="within", how="inner")
    inside_ids = set(int(value) for value in inside["taz_id"].tolist())

    demand = pd.read_csv(DEMAND_CSV, usecols=["taz_id"])
    demand_ids = set(demand["taz_id"].astype(int))
    chargers = pd.read_csv(CHARGER_COUNTS)
    charger_ids = set(chargers["taz_id"].astype(int))
    selected = sorted(inside_ids & demand_ids & charger_ids)
    demand_only = sorted(inside_ids & demand_ids - charger_ids)
    charger_only = sorted(inside_ids & charger_ids - demand_ids)
    if demand_only or charger_only:
        raise RuntimeError(
            f"福田内需求与固定桩没有对齐：只有需求 {demand_only}，只有固定桩 {charger_only}"
        )
    outside_tables = sorted(inside_ids - set(selected))
    return selected, demand_only, charger_only, outside_tables


def _mahalanobis_squared(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    precision = np.linalg.inv(covariance)
    diff = points - mean
    return np.einsum("ij,jk,ik->i", diff, precision, diff)


def _calibrate(values: np.ndarray, labels: list[str]) -> dict:
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
    for replication in range(BOOTSTRAP_REPLICATIONS):
        draw = values[generator.integers(0, count, size=count)]
        draw_mean = draw.mean(axis=0)
        draw_covariance = np.cov(draw, rowvar=False, ddof=1)
        distances = _mahalanobis_squared(draw, draw_mean, draw_covariance)
        radius_draws[replication] = float(np.quantile(distances, 0.95, method="linear"))
    radius_squared = float(np.quantile(radius_draws, 0.90, method="linear"))
    in_support = _mahalanobis_squared(values, mean, covariance) <= radius_squared

    grouped = {label: [] for label in DATE_TYPES}
    for row_index, label in enumerate(labels):
        if in_support[row_index]:
            grouped[label].append(values[row_index].copy())
    for label in DATE_TYPES:
        if len(grouped[label]) < 2:
            raise RuntimeError(f"{label} 落在椭球内的样本少于 2 天，无法计算 epsilon_l")
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
        "support_counts": {label: int(len(grouped[label])) for label in DATE_TYPES},
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "in_support_count": int(in_support.sum()),
    }


def _training_panel(taz_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(DEMAND_CSV)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= TRAIN_START) & (raw["date"] < TRAIN_END_EXCLUSIVE)]
    raw = raw[raw["taz_id"].isin(taz_ids)]
    if set(raw["date_type"]) != set(DATE_TYPES):
        raise RuntimeError(f"日期类型应为 {DATE_TYPES}，实际是 {sorted(raw['date_type'].unique())}")
    panel = raw.pivot(index="date", columns="taz_id", values="daily_volume").sort_index()
    panel = panel.reindex(columns=taz_ids)
    if panel.isna().any().any():
        raise RuntimeError("训练窗内有 zone 缺日需求")
    labels = (
        raw.drop_duplicates("date")
        .set_index("date")
        .loc[panel.index, "date_type"]
        .tolist()
    )
    if len(labels) != len(panel):
        raise RuntimeError("日期类型没有覆盖全部训练日")
    return panel, labels


def main() -> None:
    taz_ids, demand_only, charger_only, outside_tables = select_futian_taz()
    panel, labels = _training_panel(taz_ids)
    calibrated = _calibrate(panel.to_numpy(dtype=float), labels)

    distance_frame = pd.read_csv(DISTANCE_CSV, index_col="taz_id")
    distance_frame.index = distance_frame.index.astype(int)
    distance_frame.columns = [int(column) for column in distance_frame.columns]
    missing = sorted(set(taz_ids) - set(distance_frame.index))
    if missing:
        raise RuntimeError(f"距离矩阵缺少 TAZ：{missing}")
    distance = distance_frame.loc[taz_ids, taz_ids].to_numpy(dtype=float)
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise RuntimeError("距离矩阵必须有限且非负")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise RuntimeError("距离矩阵必须对称且对角线为 0")

    chargers = pd.read_csv(CHARGER_COUNTS).set_index("taz_id")
    fixed_count = chargers.loc[taz_ids, "fixed_charger_count"].to_numpy(dtype=float)
    n_bar = 2.0 * fixed_count
    if not np.equal(n_bar, np.floor(n_bar)).all() or (n_bar < 0).any():
        raise RuntimeError("固定桩上界不是非负整数")

    samples = calibrated["samples"]
    epsilon = calibrated["epsilon_l"]
    q_l = calibrated["q_l"]
    np.savez(
        ROOT / "futian_parameters.npz",
        taz_ids=np.array(taz_ids, dtype=int),
        mu=calibrated["mu"],
        Sigma=calibrated["Sigma"],
        Sigma_inv=np.linalg.inv(calibrated["Sigma"]),
        r=np.array(calibrated["r"]),
        r_squared=np.array(calibrated["r_squared"]),
        epsilon_l=epsilon,
        q_l=q_l,
        n_bar=n_bar,
        distance=distance,
        samples_workday=samples["workday"],
        samples_weekend=samples["weekend"],
        samples_holiday=samples["holiday"],
    )

    means = panel.groupby(labels).mean()
    selected = pd.DataFrame({
        "taz_id": taz_ids,
        "mean_demand": panel.mean(axis=0).to_numpy(),
        "mean_workday": means.loc["workday", taz_ids].to_numpy(),
        "mean_weekend": means.loc["weekend", taz_ids].to_numpy(),
        "mean_holiday": means.loc["holiday", taz_ids].to_numpy(),
        "fixed_charger_count": fixed_count,
    })
    selected.to_csv(ROOT / "selected_zones.csv", index=False)

    metadata = {
        "label": "futian",
        "taz_ids": taz_ids,
        "r": calibrated["r"],
        "r_squared": calibrated["r_squared"],
        "epsilon_l": {name: float(value) for name, value in zip(DATE_TYPES, epsilon)},
        "q_l": {name: float(value) for name, value in zip(DATE_TYPES, q_l)},
        "support_counts": {name: int(rows.shape[0]) for name, rows in samples.items()},
        "n_bar_rule": "2 * existing fixed_charger_count",
        "radius_rule": "Q_0.90 of bootstrap Q_0.95 Mahalanobis squared distances",
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "epsilon_rule": "median within-date-type zone standard deviation, same rule as zone20",
        "condition_number": calibrated["condition_number"],
        "in_support_count": calibrated["in_support_count"],
        "training_days": int(panel.shape[0]),
        "train_start_inclusive": TRAIN_START,
        "train_end_exclusive": TRAIN_END_EXCLUSIVE,
        "demand_field": "daily_volume",
        "demand_dimension": len(taz_ids),
        "zone_rule": (
            "representative point inside Futian after WGS84 to GCJ-02, "
            "and present in both the daily demand file and the fixed charger table"
        ),
        "excluded_demand_without_chargers": demand_only,
        "excluded_chargers_without_demand": charger_only,
        "inside_futian_without_model_tables": outside_tables,
    }
    (ROOT / "futian_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

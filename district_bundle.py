"""按福田同一套规则，把一个行政区收成求解器使用的 npz。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.ops import transform

_FUTIAN_BUILDER = Path(__file__).resolve().parent / "futian" / "build_futian_parameters.py"
_SPEC = importlib.util.spec_from_file_location("build_futian_parameters", _FUTIAN_BUILDER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"无法加载福田参数脚本：{_FUTIAN_BUILDER}")
_futian = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _futian
_SPEC.loader.exec_module(_futian)


def select_district_taz(district_name: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """代表点落在该区，且需求表和固定桩表都有记录的 TAZ，按编号升序。"""
    _futian._ensure_district_geojson()
    with _futian.DISTRICT_GEOJSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    district_features = [
        feat for feat in payload["features"] if feat["properties"]["name"] == district_name
    ]
    if len(district_features) != 1:
        raise RuntimeError(f"行政区边界里没有唯一的{district_name}")
    district = gpd.GeoDataFrame(
        [{"geometry": shape(district_features[0]["geometry"])}],
        crs="EPSG:4326",
    ).to_crs(3857)

    taz = gpd.read_file(_futian.TAZ_POLYGON).to_crs(4326)
    taz["taz_id"] = taz["TAZID"].astype(int)
    taz["geometry"] = taz.geometry.apply(
        lambda geom: transform(lambda x, y, z=None: _futian._wgs84_to_gcj02(x, y), geom)
    )
    taz = taz.to_crs(3857)
    points = taz.copy()
    points["geometry"] = points.geometry.representative_point()
    inside = gpd.sjoin(points, district, predicate="within", how="inner")
    inside_ids = set(int(value) for value in inside["taz_id"].tolist())

    demand = pd.read_csv(_futian.DEMAND_CSV, usecols=["taz_id"])
    demand_ids = set(demand["taz_id"].astype(int))
    chargers = pd.read_csv(_futian.CHARGER_COUNTS)
    charger_ids = set(chargers["taz_id"].astype(int))
    selected = sorted(inside_ids & demand_ids & charger_ids)
    demand_only = sorted(inside_ids & demand_ids - charger_ids)
    charger_only = sorted(inside_ids & charger_ids - demand_ids)
    if demand_only or charger_only:
        raise RuntimeError(
            f"{district_name}内需求与固定桩没有对齐：只有需求 {demand_only}，只有固定桩 {charger_only}"
        )
    outside_tables = sorted(inside_ids - set(selected))
    return selected, demand_only, charger_only, outside_tables


def build_district(root: Path, label: str, district_name: str, zone_rule_name: str) -> None:
    """估计椭球、epsilon_l，并写出 <label>_parameters.npz。"""
    root = Path(root)
    taz_ids, demand_only, charger_only, outside_tables = select_district_taz(district_name)
    panel, labels = _futian._training_panel(taz_ids)
    calibrated = _futian._calibrate(panel.to_numpy(dtype=float), labels)

    distance_frame = pd.read_csv(_futian.DISTANCE_CSV, index_col="taz_id")
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

    chargers = pd.read_csv(_futian.CHARGER_COUNTS).set_index("taz_id")
    fixed_count = chargers.loc[taz_ids, "fixed_charger_count"].to_numpy(dtype=float)
    n_bar = 2.0 * fixed_count
    if not np.equal(n_bar, np.floor(n_bar)).all() or (n_bar < 0).any():
        raise RuntimeError("固定桩上界不是非负整数")

    samples = calibrated["samples"]
    epsilon = calibrated["epsilon_l"]
    q_l = calibrated["q_l"]
    np.savez(
        root / f"{label}_parameters.npz",
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
    selected.to_csv(root / "selected_zones.csv", index=False)

    metadata = {
        "label": label,
        "taz_ids": taz_ids,
        "r": calibrated["r"],
        "r_squared": calibrated["r_squared"],
        "epsilon_l": {name: float(value) for name, value in zip(_futian.DATE_TYPES, epsilon)},
        "q_l": {name: float(value) for name, value in zip(_futian.DATE_TYPES, q_l)},
        "support_counts": {name: int(rows.shape[0]) for name, rows in samples.items()},
        "n_bar_rule": "2 * existing fixed_charger_count",
        "radius_rule": "Q_0.90 of bootstrap Q_0.95 Mahalanobis squared distances",
        "bootstrap_replications": _futian.BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": _futian.BOOTSTRAP_SEED,
        "epsilon_rule": "median within-date-type zone standard deviation, same rule as zone20",
        "condition_number": calibrated["condition_number"],
        "in_support_count": calibrated["in_support_count"],
        "training_days": int(panel.shape[0]),
        "train_start_inclusive": _futian.TRAIN_START,
        "train_end_exclusive": _futian.TRAIN_END_EXCLUSIVE,
        "demand_field": "daily_volume",
        "demand_dimension": len(taz_ids),
        "zone_rule": (
            f"representative point inside {zone_rule_name} after WGS84 to GCJ-02, "
            "and present in both the daily demand file and the fixed charger table"
        ),
        "excluded_demand_without_chargers": demand_only,
        "excluded_chargers_without_demand": charger_only,
        f"inside_{label}_without_model_tables": outside_tables,
    }
    (root / f"{label}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

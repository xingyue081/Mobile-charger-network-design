"""画出福田 fixed-only 与两种 mixed 的部署图。

版式：左上是固定桩减少量，其余三格是 workday、weekend、holiday 的移动桩调度。
集中补能标出枢纽，并画出从枢纽到服务区的调度。本地补能没有枢纽，只标各区部署数量。

数据是基准解 numerical_exp/base/alpha1_b0/ 的三份 JSON。
参数：rho=2，b=0，c^f=41，c^m=45，gamma=0.001，枢纽开设成本 50，本地 g=0.25，alpha=1。
入选区画色；福田内未入模区白底描边；福田外不画。数字标在区中心。

    python3 plot_exp1.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polylabel

CG_ROOT = Path("/Users/xingyueyi/Desktop/CG_solver")
BASE_DIR = CG_ROOT / "numerical_exp" / "base" / "alpha1_b0"
FIXED_PATH = BASE_DIR / "fixed_only.json"
CENTRAL_PLAN = BASE_DIR / "charge_centrally.json"
LOCAL_PATH = BASE_DIR / "charge_locally.json"
FUTIAN_META = CG_ROOT / "futian" / "futian_metadata.json"
TAZ_POLYGON_PATH = Path(
    "/Users/xingyueyi/Downloads/ST-EVCDP-main/datasets/SZ_districts/SZ_districts.shp"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DATE_TYPES = ("workday", "weekend", "holiday")
DATE_TYPE_LABELS = {"weekday": "Workday", "workday": "Workday", "weekend": "Weekend", "holiday": "Holiday"}
PANEL_LETTERS = {"workday": "b", "weekend": "c", "holiday": "d"}
REDUCTION_BINS = [0, 1, 2, 3, 5, 8, 12]


def load_fixed(path: Path) -> pd.DataFrame:
    saved = json.loads(path.read_text(encoding="utf-8"))
    decision = saved["decision"]
    frame = pd.DataFrame({
        "taz_id": [int(taz_id) for taz_id in decision["taz_ids"]],
        "fixed_only": [int(count) for count in decision["n"]],
    })
    if frame["taz_id"].duplicated().any():
        raise ValueError(f"{path} 里的 taz_id 有重复")
    return frame


def load_local(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    decision = saved["decision"]
    taz_ids = [int(taz_id) for taz_id in saved["provenance"]["taz_ids"]]
    fixed = [int(round(count)) for count in decision["n"]]
    if len(taz_ids) != len(fixed):
        raise ValueError(f"{path} 的 taz 与固定桩数量长度不一致")
    fixed_frame = pd.DataFrame({"taz_id": taz_ids, "mixed": fixed})
    rows = []
    for date_type in DATE_TYPES:
        counts = decision["z"][date_type]
        if len(counts) != len(taz_ids):
            raise ValueError(f"{path} 的 {date_type} 部署长度与 taz 不一致")
        for taz_id, count in zip(taz_ids, counts):
            rows.append({
                "date_type": date_type,
                "taz_id": taz_id,
                "mobile_chargers": int(round(count)),
            })
    deployment = pd.DataFrame(rows)
    fleet = int(round(decision["m"]))
    for date_type in DATE_TYPES:
        total = int(deployment.loc[deployment["date_type"] == date_type, "mobile_chargers"].sum())
        if total != fleet:
            raise ValueError(f"本地补能 {date_type} 部署合计 {total}，车队规模是 {fleet}")
    return fixed_frame, deployment


def load_central_plan(path: Path, taz_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读 dump_futian_central_plan.py 写出的完整决策。"""
    saved = json.loads(path.read_text(encoding="utf-8"))
    zone_ids = [int(value) for value in saved["zone_ids"]]
    if zone_ids != taz_ids:
        raise ValueError("集中补能 zone_ids 与 fixed-only taz_ids 不一致")
    decision = saved["decision"]
    zone_count = len(zone_ids)
    fixed = pd.DataFrame({
        "taz_id": zone_ids,
        "mixed": [int(decision["n"][str(index)]) for index in range(zone_count)],
    })
    hub_rows = []
    for index in range(zone_count):
        if int(decision["x"][str(index)]) != 1:
            continue
        hub_rows.append({
            "taz_id": zone_ids[index],
            "mobile_chargers_at_hub": int(decision["m"][str(index)]),
        })
    hubs = pd.DataFrame(hub_rows)
    if hubs.empty:
        raise ValueError(f"{path} 没有建站枢纽")
    dispatch_rows = []
    for key, count in decision["z"].items():
        value = int(count)
        if value <= 0:
            continue
        zone_index, hub_index, date_index = (int(part) for part in key.split(","))
        dispatch_rows.append({
            "date_type": DATE_TYPES[date_index],
            "origin_taz_id": zone_ids[hub_index],
            "destination_taz_id": zone_ids[zone_index],
            "mobile_chargers": value,
        })
    dispatch = pd.DataFrame(dispatch_rows)
    fleet = int(hubs["mobile_chargers_at_hub"].sum())
    for date_type in DATE_TYPES:
        sent = 0 if dispatch.empty else int(
            dispatch.loc[dispatch["date_type"] == date_type, "mobile_chargers"].sum()
        )
        if sent != fleet:
            raise ValueError(f"集中补能 {date_type} 调度合计 {sent}，枢纽车辆合计 {fleet}")
    return fixed, hubs, dispatch


def label_xy(geometry) -> tuple[float, float]:
    """取多边形内的视觉中心，供数字标注。"""
    geom = geometry
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda part: part.area)
    if isinstance(geom, Polygon) and not geom.is_empty:
        try:
            point = polylabel(geom, tolerance=5.0)
            return float(point.x), float(point.y)
        except Exception:
            pass
    centroid = geometry.centroid
    if geometry.contains(centroid) or geometry.covers(centroid):
        return float(centroid.x), float(centroid.y)
    point = geometry.representative_point()
    return float(point.x), float(point.y)


def load_polygons(taz_ids: list[int]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """入选 TAZ 填色；福田内未入模区作白底描边；福田外不画。"""
    gdf = gpd.read_file(TAZ_POLYGON_PATH)
    gdf["taz_id"] = gdf["TAZID"].astype(float).astype(int)
    selected = gdf[gdf["taz_id"].isin(taz_ids)].copy()
    missing = sorted(set(taz_ids) - set(selected["taz_id"]))
    if missing:
        raise ValueError(f"缺少 TAZ 多边形：{missing}")
    meta = json.loads(FUTIAN_META.read_text(encoding="utf-8"))
    hole_ids = [int(value) for value in meta["inside_futian_without_model_tables"]]
    context = gdf[gdf["taz_id"].isin(hole_ids)].copy()
    missing_holes = sorted(set(hole_ids) - set(context["taz_id"]))
    if missing_holes:
        raise ValueError(f"缺少福田未入模 TAZ 多边形：{missing_holes}")
    coords = [label_xy(geom) for geom in selected.geometry]
    selected["point_x"] = [xy[0] for xy in coords]
    selected["point_y"] = [xy[1] for xy in coords]
    selected = selected[["taz_id", "geometry", "point_x", "point_y"]]
    context = context[["taz_id", "geometry"]]
    return selected, context


def reduction_table(polygons: gpd.GeoDataFrame, fixed_only: pd.DataFrame, mixed: pd.DataFrame) -> gpd.GeoDataFrame:
    table = polygons.merge(fixed_only, on="taz_id", how="left").merge(mixed, on="taz_id", how="left")
    if table[["fixed_only", "mixed"]].isna().any().any():
        raise ValueError("固定桩数量没有覆盖全部研究区")
    table["reduction"] = table["fixed_only"] - table["mixed"]
    if (table["reduction"] < 0).any():
        raise ValueError("存在 mixed 固定桩多于 fixed-only 的区")
    return table


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.unicode_minus": False,
    })


def set_common_extent(ax: plt.Axes, layers: list[gpd.GeoDataFrame]) -> None:
    bounds = [layer.total_bounds for layer in layers if len(layer)]
    xmin = min(item[0] for item in bounds)
    ymin = min(item[1] for item in bounds)
    xmax = max(item[2] for item in bounds)
    ymax = max(item[3] for item in bounds)
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")


def draw_context(ax: plt.Axes, context: gpd.GeoDataFrame) -> None:
    if context.empty:
        return
    context.plot(ax=ax, color="white", edgecolor="#7A7A7A", linewidth=0.55, zorder=0)


def label_with_halo(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float,
    color: str,
    offset_points: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset_points,
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        zorder=8,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
    )


def draw_base(ax: plt.Axes, gdf: gpd.GeoDataFrame, context: gpd.GeoDataFrame) -> None:
    draw_context(ax, context)
    gdf.plot(ax=ax, color="#F5F5F2", edgecolor="#545454", linewidth=0.65, zorder=1)
    set_common_extent(ax, [gdf, context])


def draw_reduction_panel(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    mode_name: str,
    cmap,
    norm,
) -> None:
    draw_context(ax, context)
    gdf.plot(ax=ax, column="reduction", cmap=cmap, norm=norm, edgecolor="#404040", linewidth=0.65, zorder=2)
    set_common_extent(ax, [gdf, context])
    total = int(gdf["reduction"].sum())
    ax.set_title(
        "(a) Fixed charger reduction with mobile resources\n"
        f"Fixed-only minus {mode_name}: {total} fewer fixed chargers",
        fontsize=12,
        pad=8,
    )
    for _, row in gdf.iterrows():
        reduction = int(row["reduction"])
        if reduction <= 0:
            continue
        label_with_halo(ax, row["point_x"], row["point_y"], str(reduction), fontsize=7.0, color="#222222")


def draw_dispatch_arrows(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    for _, row in dispatch.iterrows():
        if row["origin_taz_id"] == row["destination_taz_id"]:
            continue
        distance = math.hypot(row["destination_x"] - row["origin_x"], row["destination_y"] - row["origin_y"])
        short = distance < 1500
        width = 0.75 + 0.1 * float(row["mobile_chargers"])
        arrowprops = {
            "arrowstyle": "-|>",
            "color": "#8C4A17",
            "lw": width + (0.35 if short else 0.0),
            "alpha": 0.76 if short else 0.58,
            "shrinkA": 2 if short else 5,
            "shrinkB": 2 if short else 5,
            "mutation_scale": 12 if short else 9,
        }
        if short:
            arrowprops["connectionstyle"] = "arc3,rad=0.22"
        ax.annotate(
            "",
            xy=(row["destination_x"], row["destination_y"]),
            xytext=(row["origin_x"], row["origin_y"]),
            arrowprops=arrowprops,
            zorder=5,
        )


def draw_central_dispatch_panel(
    ax: plt.Axes,
    polygons: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    hubs: pd.DataFrame,
    dispatch: pd.DataFrame,
    date_type: str,
) -> None:
    draw_base(ax, polygons, context)
    date_dispatch = dispatch[dispatch["date_type"] == date_type].copy()
    points = polygons.set_index("taz_id")[["point_x", "point_y"]]
    hub_points = hubs.merge(polygons[["taz_id", "point_x", "point_y"]], on="taz_id", how="left")
    if not date_dispatch.empty:
        date_dispatch = date_dispatch.join(points, on="origin_taz_id")
        date_dispatch = date_dispatch.rename(columns={"point_x": "origin_x", "point_y": "origin_y"})
        date_dispatch = date_dispatch.join(points, on="destination_taz_id")
        date_dispatch = date_dispatch.rename(columns={"point_x": "destination_x", "point_y": "destination_y"})
        draw_dispatch_arrows(ax, date_dispatch)
        destination = (
            date_dispatch.groupby("destination_taz_id", as_index=False)["mobile_chargers"].sum()
            .merge(polygons[["taz_id", "point_x", "point_y"]], left_on="destination_taz_id", right_on="taz_id")
        )
        hub_ids = set(hubs["taz_id"])
        outside = destination[~destination["destination_taz_id"].isin(hub_ids)]
        for _, row in outside.iterrows():
            label_with_halo(
                ax, row["point_x"], row["point_y"], str(int(row["mobile_chargers"])),
                fontsize=8.2, color="#6B2D12",
            )
    self_dispatch = {}
    if not date_dispatch.empty:
        stayed = date_dispatch[date_dispatch["origin_taz_id"] == date_dispatch["destination_taz_id"]]
        self_dispatch = stayed.groupby("origin_taz_id")["mobile_chargers"].sum().to_dict()
    for _, row in hub_points.iterrows():
        stayed_count = int(self_dispatch.get(row["taz_id"], 0))
        text = f"Hub {stayed_count}" if stayed_count > 0 else "Hub"
        label_with_halo(ax, row["point_x"], row["point_y"], text, fontsize=8.0, color="#771018")
    ax.set_title(
        f"({PANEL_LETTERS[date_type]}) {DATE_TYPE_LABELS[date_type]} mobile dispatch",
        fontsize=12,
        pad=8,
    )
    set_common_extent(ax, [polygons, context])


def draw_local_dispatch_panel(
    ax: plt.Axes,
    polygons: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    deployment: pd.DataFrame,
    date_type: str,
) -> None:
    draw_base(ax, polygons, context)
    placed = deployment[(deployment["date_type"] == date_type) & (deployment["mobile_chargers"] > 0)]
    placed = placed.merge(polygons[["taz_id", "point_x", "point_y"]], on="taz_id", how="left")
    for _, row in placed.iterrows():
        label_with_halo(
            ax, row["point_x"], row["point_y"], str(int(row["mobile_chargers"])),
            fontsize=8.2, color="#6B2D12",
        )
    ax.set_title(
        f"({PANEL_LETTERS[date_type]}) {DATE_TYPE_LABELS[date_type]} mobile dispatch",
        fontsize=12,
        pad=8,
    )
    set_common_extent(ax, [polygons, context])


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for suffix in (".png", ".pdf", ".eps"):
        path = OUTPUT_DIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.12, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def compose(
    mode_name: str,
    reduction: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    draw_dispatch,
) -> plt.Figure:
    cmap = plt.get_cmap("Blues", len(REDUCTION_BINS) - 1)
    norm = mcolors.BoundaryNorm(REDUCTION_BINS, cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.2), sharex=True, sharey=True)
    draw_reduction_panel(axes[0, 0], reduction, context, mode_name, cmap, norm)
    draw_dispatch(axes[0, 1], "workday")
    draw_dispatch(axes[1, 0], "weekend")
    draw_dispatch(axes[1, 1], "holiday")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[0, 0],
        orientation="horizontal",
        fraction=0.06,
        pad=0.045,
        ticks=REDUCTION_BINS,
    )
    colorbar.set_label(f"Reduction in fixed chargers (Fixed-only minus {mode_name})", labelpad=7)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.045, wspace=0.04, hspace=0.28)
    return fig


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CENTRAL_PLAN.is_file():
        raise FileNotFoundError(f"缺少基准解 {CENTRAL_PLAN}")
    fixed_only = load_fixed(FIXED_PATH)
    taz_ids = fixed_only["taz_id"].tolist()
    central_fixed, hubs, central_dispatch = load_central_plan(CENTRAL_PLAN, taz_ids)
    local_fixed, local_deployment = load_local(LOCAL_PATH)
    polygons, context = load_polygons(taz_ids)

    central_reduction = reduction_table(polygons, fixed_only, central_fixed)
    local_reduction = reduction_table(polygons, fixed_only, local_fixed)

    central_reduction[
        ["taz_id", "fixed_only", "mixed", "reduction", "point_x", "point_y"]
    ].to_csv(OUTPUT_DIR / "charge_centrally_fixed_reduction.csv", index=False)
    hubs.to_csv(OUTPUT_DIR / "charge_centrally_hubs.csv", index=False)
    central_dispatch.to_csv(OUTPUT_DIR / "charge_centrally_dispatch.csv", index=False)
    local_reduction[
        ["taz_id", "fixed_only", "mixed", "reduction", "point_x", "point_y"]
    ].to_csv(OUTPUT_DIR / "charge_locally_fixed_reduction.csv", index=False)
    local_deployment.to_csv(OUTPUT_DIR / "charge_locally_deployment.csv", index=False)

    central_figure = compose(
        "charge centrally",
        central_reduction,
        context,
        lambda ax, date_type: draw_central_dispatch_panel(
            ax, polygons, context, hubs, central_dispatch, date_type,
        ),
    )
    local_figure = compose(
        "charge locally",
        local_reduction,
        context,
        lambda ax, date_type: draw_local_dispatch_panel(ax, polygons, context, local_deployment, date_type),
    )
    written = save_figure(central_figure, "fig_charge_centrally_vs_fixed_only")
    written.extend(save_figure(local_figure, "fig_charge_locally_vs_fixed_only"))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

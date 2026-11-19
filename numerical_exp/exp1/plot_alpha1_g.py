"""画出 alpha=1 时 g=0.25 与 g=0.3 的 charge locally 部署图。

版式与 plot_exp1.py 相同：左上是相对 fixed-only 的固定桩减少量，
其余三格是 workday、weekend、holiday 的移动车部署。
fixed-only 与两种 g 都取福田 alpha=1 的收敛解。centrally 不随 g 变化，
且 alpha=1 的结果文件没有分区调度，这里不画。

    python3 plot_alpha1_g.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gpd

import plot_exp1 as base

ALPHA_ROOT = Path("/Users/xingyueyi/Desktop/CG_solver/numerical_exp/exp_futian_alpha/outputs")
FIXED_PATH = ALPHA_ROOT / "final" / "g0.3_hub55" / "alpha_1" / "fixed_only.json"
CASES = (
    ("g=0.25", ALPHA_ROOT / "screen" / "g0.25_hub55" / "alpha_1" / "charge_locally.json"),
    ("g=0.3", ALPHA_ROOT / "final" / "g0.3_hub55" / "alpha_1" / "charge_locally.json"),
)


def compose_case(stem: str, fleet: int, reduction: gpd.GeoDataFrame, context: gpd.GeoDataFrame, deployment) -> plt.Figure:
    bins = [0, 1, 2]
    cmap = plt.get_cmap("Blues", len(bins) - 1)
    norm = mcolors.BoundaryNorm(bins, cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.2), sharex=True, sharey=True)
    base.draw_reduction_panel(axes[0, 0], reduction, context, stem, cmap, norm)
    total = int(reduction["reduction"].sum())
    axes[0, 0].set_title(
        f"(a) Fixed charger reduction, {stem}, alpha=1\n"
        f"{total} fewer fixed chargers than fixed-only; {fleet} mobile chargers",
        fontsize=12,
        pad=8,
    )
    base.draw_local_dispatch_panel(axes[0, 1], reduction, context, deployment, "workday")
    base.draw_local_dispatch_panel(axes[1, 0], reduction, context, deployment, "weekend")
    base.draw_local_dispatch_panel(axes[1, 1], reduction, context, deployment, "holiday")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[0, 0],
        orientation="horizontal",
        fraction=0.06,
        pad=0.045,
        ticks=bins,
    )
    colorbar.set_label("Reduction in fixed chargers (fixed-only minus charge locally)", labelpad=7)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.045, wspace=0.04, hspace=0.28)
    return fig


def draw_case(stem: str, local_path: Path, fixed_only, polygons: gpd.GeoDataFrame, context: gpd.GeoDataFrame) -> None:
    local_fixed, deployment = base.load_local(local_path)
    reduction = base.reduction_table(polygons, fixed_only, local_fixed)
    fleet = int(deployment.loc[deployment["date_type"] == "workday", "mobile_chargers"].sum())
    figure = compose_case(stem, fleet, reduction, context, deployment)
    written = base.save_figure(figure, stem)
    for path in written:
        print(path)


def main() -> None:
    base.apply_style()
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_only = base.load_fixed(FIXED_PATH)
    polygons, context = base.load_polygons(fixed_only["taz_id"].tolist())
    for stem, path in CASES:
        draw_case(stem, path, fixed_only, polygons, context)


if __name__ == "__main__":
    main()

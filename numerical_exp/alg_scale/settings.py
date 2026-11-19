"""规模对照实验的固定设置。集中补能不使用 local_g。"""

from pathlib import Path

CG_ROOT = Path("/Users/xingyueyi/Desktop/CG_solver")
SOLVER_DIR = CG_ROOT / "charge_centrally_solver_1030"
FUTIAN_NPZ = CG_ROOT / "futian" / "futian_parameters.npz"
EXP_ROOT = CG_ROOT / "numerical_exp" / "alg_scale"
DATA_ROOT = EXP_ROOT / "data"
OUTPUT_ROOT = EXP_ROOT / "outputs"

ZONE_COUNTS = (3, 5, 10, 15, 20, 35)
FULL_MODEL_COUNTS = (3, 5, 10, 15)
WALL_SECONDS = 4 * 60 * 60
DATE_TYPES = ("workday", "weekend", "holiday")
TRAIN_Q = (83 / 122, 28 / 122, 11 / 122)

BUSINESS = {
    "rho": 2.0,
    "shortage_penalty": 0.0,
    "fixed_capacity": 190.0,
    "mobile_capacity": 160.0,
    "fixed_unit_cost": 41.0,
    "mobile_unit_cost": 45.0,
    "transport_cost_per_distance": 0.001,
    "hub_open_cost": 50.0,
}
LOCAL_G = 0.25
PYTHON = Path(
    "/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/.venv-gurobi-13.0.3/bin/python"
)

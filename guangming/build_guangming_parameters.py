"""把光明区训练样本收成与福田、罗湖相同字段的 npz。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from district_bundle import build_district

if __name__ == "__main__":
    build_district(Path(__file__).resolve().parent, "guangming", "光明区", "Guangming")

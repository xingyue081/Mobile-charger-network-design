"""比较约束生成和整数穷举。

在本目录运行：python validate.py
两边使用同一份合成实例和同一张支持函数表。
目标差超过 1e-5，或约束生成的决策不在穷举最优集合里，脚本以非零状态退出。

通过时只能说明：在 g=1 的合成 2 区、2 个日期类型上，
约束生成找到的目标和部署，与“枚举全部 pattern 的 beta-h”一致。
结果写在 validation_result.json。

会各跑一遍默认加速（容量提成 + 热启动 + 未证明不加割）
和 adaptive_gap，两边都必须与暴力枚举一致。
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import gurobipy as gp

from constraint_generation import solve_constraint_generation
from enumeration import enumerate_optima
from instance import synthetic_two_zone
from support import precompute_supports


def _same_decision(left: dict, right: dict) -> bool:
    """比较 n、m 和每种日期类型的 z。允许 Gurobi 读出 1.0000001 这种整数。"""
    if [int(round(value)) for value in left["n"]] != list(right["n"]):
        return False
    if int(round(left["m"])) != int(right["m"]):
        return False
    for event, deployment in right["z"].items():
        if [int(round(value)) for value in left["z"][event]] != list(deployment):
            return False
    return True


def _compare(label: str, instance: dict, supports: dict, enumerated: dict) -> dict:
    generated = solve_constraint_generation(instance, supports)
    objective_gap = None
    if generated["objective"] is not None:
        objective_gap = abs(generated["objective"] - enumerated["objective"])
    decision_matches = generated["decision"] is not None and any(
        _same_decision(generated["decision"], decision) for decision in enumerated["decisions"]
    )
    passed = bool(
        generated["converged"]
        and objective_gap is not None
        and objective_gap <= 1e-5
        and decision_matches
    )
    print(json.dumps({
        "mode": label,
        "passed": passed,
        "objective_gap": objective_gap,
        "cg_objective": generated["objective"],
        "enumeration_objective": enumerated["objective"],
        "iterations": generated["iterations"],
        "pattern_count": generated["pattern_count"],
        "reason": generated["reason"],
        "acceleration": generated.get("acceleration"),
        "decision": generated["decision"],
    }, indent=2, ensure_ascii=False, default=list), flush=True)
    return {
        "mode": label,
        "passed": passed,
        "objective_gap": objective_gap,
        "constraint_generation": generated,
    }


def main() -> None:
    base = synthetic_two_zone()
    supports = precompute_supports(base)
    enumerated = enumerate_optima(base, supports)

    tight = deepcopy(base)
    tight["master_mip_gap"] = 0.0
    tight["adaptive_gap"] = False

    adaptive = deepcopy(base)
    adaptive["master_mip_gap"] = 0.0
    adaptive["adaptive_gap"] = True

    rows = [
        _compare("tight", tight, supports, enumerated),
        _compare("adaptive", adaptive, supports, enumerated),
    ]
    passed = all(row["passed"] for row in rows)
    payload = {
        "gurobi": list(gp.gurobi.version()),
        "passed": passed,
        "enumeration_objective": enumerated["objective"],
        "enumeration_decisions": enumerated["decisions"],
        "modes": rows,
        "scope": (
            "合成 2-zone、2 个日期类型；核对容量提成、热启动、未证明不加割、"
            "以及 adaptive_gap 与暴力枚举一致。不是深圳数据。"
        ),
    }
    output = Path(__file__).with_name("validation_result.json")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
    print(json.dumps({"passed": passed, "output": str(output)}, indent=2, ensure_ascii=False), flush=True)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# DRO probes for 4-hub Futian: seek >=5% lift vs fixed-only, fleet not fully capped.
# Acceptance for formal later: optimality-gap 1%, wall <= 2h.
set -uo pipefail

ROOT="/Users/xingyueyi/Desktop/CG_solver"
OUT="$ROOT/numerical_exp/param_retune/probe_4hub_5pct"
PY="/Users/xingyueyi/Documents/ChatGPT/mobile charger_new/.venv-gurobi-13.0.3/bin/python"
FIXED="$ROOT/fixed_only_solver"
CENT="$ROOT/charge_centrally_solver11.16"
export CG_THREADS="${CG_THREADS:-4}"

# probe budgets (short screen; formal run later uses 7200 / 0.01)
FIXED_SECONDS="${FIXED_SECONDS:-1200}"
CENT_SECONDS="${CENT_SECONDS:-2400}"
GAP="${GAP:-0.01}"

mkdir -p "$OUT/logs" "$OUT/results"
SUMMARY="$OUT/summary.jsonl"
: > "$SUMMARY"

# Default grid; override by passing "cf,cm,hub" args.
if [[ $# -gt 0 ]]; then
  GRID=("$@")
else
  GRID=(
    "180,160,200"
    "180,170,200"
    "200,170,200"
    "200,180,220"
  )
fi

echo "probe grid: ${GRID[*]}"
echo "fixed_seconds=$FIXED_SECONDS cent_seconds=$CENT_SECONDS gap=$GAP"

for spec in "${GRID[@]}"; do
  IFS=',' read -r CF CM HUB <<<"$spec"
  tag="cf${CF}_cm${CM}_hub${HUB}"
  echo "===== START $tag $(date '+%F %T') =====" | tee -a "$OUT/driver.log"

  fixed_json="$OUT/results/${tag}_fixed.json"
  cent_json="$OUT/results/${tag}_centrally.json"

  fixed_rc=0
  (
    cd "$FIXED"
    "$PY" -u main.py \
      --data "$ROOT/futian" \
      --rho 2 --shortage-penalty 2 \
      --fixed-cost "$CF" \
      --seconds "$FIXED_SECONDS" \
      --solve \
      --output "$fixed_json"
  ) >"$OUT/logs/${tag}_fixed.log" 2>&1 || fixed_rc=$?

  cent_rc=0
  (
    cd "$CENT"
    "$PY" -u main.py \
      --instance futian \
      --rho 2 --shortage-penalty 2 \
      --fixed-cost "$CF" \
      --mobile-cost "$CM" \
      --hub-open-cost "$HUB" \
      --optimality-gap "$GAP" \
      --max-total-seconds "$CENT_SECONDS" \
      --output "$cent_json"
  ) >"$OUT/logs/${tag}_centrally.log" 2>&1 || cent_rc=$?

  "$PY" - <<PY
import json
from pathlib import Path
tag = "$tag"
out = Path("$OUT")
fixed_path = out / "results" / f"{tag}_fixed.json"
cent_path = out / "results" / f"{tag}_centrally.json"
rec = {
    "tag": tag,
    "cf": float("$CF"),
    "cm": float("$CM"),
    "hub": float("$HUB"),
    "fixed_rc": int("$fixed_rc"),
    "cent_rc": int("$cent_rc"),
}
if fixed_path.exists():
    fixed = json.loads(fixed_path.read_text())
    f_obj = float(fixed["objective"])
    rec.update({
        "fixed_obj": f_obj,
        "fixed_status": fixed.get("status") or fixed.get("reason"),
        "fixed_stop": fixed.get("reason") or fixed.get("stop_reason"),
    })
    if isinstance(fixed.get("decision"), dict) and "n" in fixed["decision"]:
        rec["fixed_n"] = int(round(sum(fixed["decision"]["n"])))
else:
    f_obj = None
    rec["fixed_error"] = "missing_json"
if cent_path.exists():
    cent = json.loads(cent_path.read_text())
    c_lb = float(cent["objective"])
    c_ub = float(cent.get("bound") or cent["objective"])
    rec.update({
        "cent_obj_lb": c_lb,
        "cent_obj_ub": c_ub,
        "cent_gap": cent.get("mip_gap"),
        "cent_n": cent.get("fixed_chargers"),
        "cent_m": cent.get("mobile_chargers"),
        "cent_hubs": cent.get("hub_count"),
        "stop_reason": cent.get("stop_reason"),
    })
    if f_obj:
        rec["lift_vs_fixed_using_cent_lb"] = (c_lb - f_obj) / abs(f_obj)
        rec["lift_vs_fixed_using_cent_ub"] = (c_ub - f_obj) / abs(f_obj)
else:
    rec["cent_error"] = "missing_json"
print(json.dumps(rec, ensure_ascii=False))
(out / "summary.jsonl").open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
if "lift_vs_fixed_using_cent_lb" in rec:
    print(
        f"{tag}: fixed={rec['fixed_obj']:.1f} cent_lb={rec['cent_obj_lb']:.1f} "
        f"lift_lb={100*rec['lift_vs_fixed_using_cent_lb']:.2f}% "
        f"lift_ub={100*rec['lift_vs_fixed_using_cent_ub']:.2f}% "
        f"n={rec.get('cent_n')} m={rec.get('cent_m')} hubs={rec.get('cent_hubs')} gap={rec.get('cent_gap')}"
    )
else:
    print(f"{tag}: incomplete fixed_rc={rec['fixed_rc']} cent_rc={rec['cent_rc']}")
PY

  echo "===== END $tag $(date '+%F %T') fixed_rc=$fixed_rc cent_rc=$cent_rc =====" | tee -a "$OUT/driver.log"
done

echo "ALL PROBES DONE $(date '+%F %T')" | tee -a "$OUT/driver.log"

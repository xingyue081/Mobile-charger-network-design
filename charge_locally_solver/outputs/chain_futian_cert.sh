#!/bin/bash
# 等当前 b=2 证明进程结束后：若未收敛则继续 resume 证明；
# b=2 收敛后再证明 b=0。
set -u
PY=/opt/homebrew/bin/python3
DIR=/Users/xingyueyi/Desktop/CG_solver/charge_locally_solver
DATA=/Users/xingyueyi/Desktop/CG_solver/futian
cd "$DIR"
mkdir -p outputs
CHAIN_LOG=outputs/chain_futian_cert.log

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "$CHAIN_LOG"
}

wait_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

json_field() {
  local path="$1"
  local field="$2"
  "$PY" -c "import json; d=json.load(open('$path')); print(d.get('$field'))"
}

run_cert() {
  local penalty="$1"
  local resume="$2"
  local output="$3"
  local logf="$4"
  log "START cert b=${penalty} resume=${resume} output=${output}"
  "$PY" -u run_20zone.py \
    --data "$DATA" \
    --g 0.1 \
    --shortage-penalty "$penalty" \
    --resume "$resume" \
    --master-mip-gap 0.0001 \
    --master-time-limit 7200 \
    --output "$output" \
    --solve \
    > "$logf" 2>&1
  local rc=$?
  log "END cert b=${penalty} exit=${rc} status=$(json_field "$output" status) reason=$(json_field "$output" reason) converged=$(json_field "$output" converged)"
  return "$rc"
}

B2_JSON=outputs/futian_b2_g0.1_adaptgap_cert.json
B2_LOG=outputs/futian_b2_g0.1_adaptgap_cert.log
B0_RESUME=outputs/futian_b0_g0.1_adaptgap.json
B0_JSON=outputs/futian_b0_g0.1_adaptgap_cert.json
B0_LOG=outputs/futian_b0_g0.1_adaptgap_cert.log

CURRENT_B2_PID="${1:-}"
if [[ -n "$CURRENT_B2_PID" ]]; then
  log "WAIT current b=2 pid=${CURRENT_B2_PID}"
  wait_pid "$CURRENT_B2_PID"
  log "CURRENT b=2 process exited"
fi

# b=2：未收敛就从最新 json 继续证明
while true; do
  if [[ -f "$B2_JSON" ]] && [[ "$(json_field "$B2_JSON" converged)" == "True" ]]; then
    log "b=2 CONVERGED objective=$(json_field "$B2_JSON" objective)"
    break
  fi
  if [[ ! -f "$B2_JSON" ]]; then
    log "ERROR missing $B2_JSON"
    exit 1
  fi
  # 第一次若当前进程已写入过，用 cert json 接着；否则用 adaptgap
  resume="$B2_JSON"
  if [[ "$(json_field "$B2_JSON" pattern_count)" == "None" ]]; then
    resume=outputs/futian_b2_g0.1_adaptgap.json
  fi
  run_cert 2 "$resume" "$B2_JSON" "$B2_LOG" || true
done

# b=0：先从 adaptgap checkpoint，之后从 cert json
first_b0=1
while true; do
  if [[ -f "$B0_JSON" ]] && [[ "$(json_field "$B0_JSON" converged)" == "True" ]]; then
    log "b=0 CONVERGED objective=$(json_field "$B0_JSON" objective)"
    break
  fi
  if [[ "$first_b0" -eq 1 ]]; then
    resume="$B0_RESUME"
    first_b0=0
  else
    resume="$B0_JSON"
  fi
  if [[ ! -f "$resume" ]]; then
    log "ERROR missing resume $resume"
    exit 1
  fi
  run_cert 0 "$resume" "$B0_JSON" "$B0_LOG" || true
done

log "ALL DONE b2 and b0 certified"

#!/usr/bin/env bash
# run_ablation.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-train.py}"


LOG_DIR="${LOG_DIR:-logs}"


# base / no_gin / no_encoder / no_decoder / no_gin_no_attention
# VARIANTS=(${VARIANTS:-base no_gin no_encoder no_decoder no_gin_no_attention})
VARIANTS=(${VARIANTS:-base})


QNUMS=(${QNUMS:-4 8 12 16 20})

mkdir -p "$LOG_DIR"

echo "== Ablation runner =="
echo "Python  : $PYTHON_BIN"
echo "Script  : $SCRIPT"
echo "Logs    : $LOG_DIR"
echo "Variants: ${VARIANTS[*]}"
echo "QNUMs   : ${QNUMS[*]}"
echo

run_one() {
  local variant="$1"
  local qnum="$2"
  local stamp
  stamp="$(date +'%Y%m%d_%H%M%S')"

  local log_file="${LOG_DIR}/${stamp}_q${qnum}_${variant}.log"

  echo "---- Running variant: ${variant}  (selected-query-num = ${qnum}) ----"
  echo "Log -> ${log_file}"
  echo


  (
    time "$PYTHON_BIN" "$SCRIPT" \
      --variant "$variant" \
      --selected-query-num "$qnum"
  ) 2>&1 | tee "$log_file"
  echo
}


for q in "${QNUMS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    run_one "$v" "$q"
  done
done


SUMMARY="${LOG_DIR}/summary.csv"

echo "variant,qnum,mean_qerror,abslog10_median,abslog10_q3" > "$SUMMARY"

for log in "$LOG_DIR"/*.log; do
  base="$(basename "$log")"


  qnum="$(sed -E 's/^[0-9_]+_q([0-9]+)_.*/\1/' <<< "$base")"
  v="$(sed -E 's/^[0-9_]+_q[0-9]+_([A-Za-z_]+)\.log.*/\1/' <<< "$base")"


  mean_qerr="$(sed -n 's/.*Overall mean Q-Error = \([0-9.]\+\).*/\1/p' "$log" | tail -n1)"


  median_log="$(sed -n 's/.*Overall |log10Q| median = \([0-9.]\+\).*Q3 = \([0-9.]\+\).*/\1/p' "$log" | tail -n1)"
  q3_log="$(sed -n 's/.*Overall |log10Q| median = \([0-9.]\+\).*Q3 = \([0-9.]\+\).*/\2/p' "$log" | tail -n1)"


  mean_qerr="${mean_qerr:-}"
  median_log="${median_log:-}"
  q3_log="${q3_log:-}"
  qnum="${qnum:-}"

  echo "${v},${qnum},${mean_qerr},${median_log},${q3_log}" >> "$SUMMARY"
done

echo "== Done =="
echo "Summary -> ${SUMMARY}"

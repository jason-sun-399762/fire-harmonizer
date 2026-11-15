#!/usr/bin/env bash
# Run fire-mask jobs (one "PRODUCT YEAR" per line) in parallel, resumable.
#   source ~/.r2env && source .venv/bin/activate
#   nohup bash scripts/run_parallel.sh scripts/viirs_jobs.txt > logs/runner.log 2>&1 &
# Optional: second argument = number of parallel jobs (default: half the CPU cores).
# DRY=1 only prints what would run.
set -u
cd "$(dirname "$0")/.."
JOBS=${1:-scripts/viirs_jobs.txt}
P=${2:-$(( $(nproc) / 2 ))}
[ "$P" -lt 1 ] && P=1
mkdir -p logs
if [ -z "${DRY:-}" ]; then
  : "${AWS_ACCESS_KEY_ID:?run: source ~/.r2env}" "${AWS_SECRET_ACCESS_KEY:?run: source ~/.r2env}"
fi
echo "$(date) starting $(grep -cv '^\s*\(#\|$\)' "$JOBS") jobs, $P at a time"
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | DRY=${DRY:-} xargs -P "$P" -L 1 bash -c '
  prod=$0; year=$1; log=logs/$prod.$year.log
  if grep -q "(uploaded)" "$log" 2>/dev/null; then echo "skip $prod $year (already done)"; exit 0; fi
  if [ -n "$DRY" ]; then echo "would run: $prod $year"; exit 0; fi
  for try in 1 2; do
    echo "$(date +%H:%M) start $prod $year (try $try)"
    python3 scripts/build_firemask.py "$prod" --years "$year" --threads 24 --upload > "$log" 2>&1 && grep -q "(uploaded)" "$log" && break
  done
  echo "$(date +%H:%M) end   $prod $year: $(tail -1 "$log")"
'
echo "$(date) all jobs finished"

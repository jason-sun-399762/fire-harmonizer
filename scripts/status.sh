#!/usr/bin/env bash
# Progress of the fire-mask jobs: done / running / failed, plus the last line of each running job.
cd "$(dirname "$0")/.."
done=0; run=0; fail=0
for f in logs/*A1.*.log; do
  [ -e "$f" ] || continue
  if grep -q "(uploaded)" "$f"; then done=$((done+1))
  elif grep -q "Traceback\|Killed" "$f"; then fail=$((fail+1)); echo "FAILED  $(basename "$f" .log): $(grep -m1 -E 'Error|Killed' "$f" | cut -c1-120)"
  else run=$((run+1)); echo "running $(basename "$f" .log): $(tail -1 "$f" | cut -c1-100)"; fi
done
echo "done $done | running $run | failed $fail"
echo "memory: $(free -g | awk '/Mem/{print $3" of "$2" GB used"}')   load: $(cut -d' ' -f1-3 /proc/loadavg)"

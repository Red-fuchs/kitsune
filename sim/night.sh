#!/bin/zsh
# one-shot night run (launchd local.kitsune-night, 2026-11-04 01:00): wait for the wiki embed job, then core sim via gpu-run, compare with the baseline, remove own agent.
D=${0:A:h}; LOG=$HOME/AI-Model/wiki-build-log/lead-embed.txt; OUT=$D/reports/night-$(date +%Y%m%d-%H%M).txt
LBL=local.kitsune-night; PL=$HOME/Library/LaunchAgents/$LBL.plist
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-04 08:00" +%s)
cleanup() { launchctl bootout gui/$(id -u)/$LBL 2>/dev/null; rm -f $PL; }
trap cleanup EXIT
while true; do
  if [[ -f $LOG ]] && grep -qE "^=== done|ABORT|FAILED" $LOG; then break; fi  # ABORT/FAILED = job ended too -> GPU is free
  if (( $(date +%s) >= DEADLINE )); then echo "gave up waiting for lead-embed at $(date)" > $OUT; ki-plan rm $(cat $D/reports/.night-id 2>/dev/null) 2>/dev/null; exit 0; fi
  sleep 60
done
ID=$(cat $D/reports/.night-id 2>/dev/null)
cd $D/..
gpu-run -n kitsune-sim-night --ram 12 ~/.venvs/mlxvlm/bin/python -I sim/run.py --rooms buero,kueche,hotel,auto,kopfhoerer --scen core > $OUT 2>&1
F=( $(ls -t $D/reports/*_*.json | head -5) )
{ echo; echo "## latency per path (new run, jsons: ${F[@]:t})"; python3 $D/latency.py $F; echo; echo "## baseline 20261101-214624"; python3 $D/latency.py $D/reports/20261101-214624_*.json; } >> $OUT 2>&1
[[ -n $ID ]] && ki-plan done $ID

#!/bin/zsh
# barge-in matrix, silent: false barge (TV only) + missed/cut (user only) per room/level/seed. usage: barge_matrix.sh [extra barge_only args]
D=${0:A:h}; PY=${KITSUNE_PY:-$HOME/.venvs/mlxvlm/bin/python}
for r in buero kueche hotel auto kopfhoerer; do
  for s in 0 1 2; do
    for db in -15 -12 -9; do
      echo "TV $r s$s ${db}dB: $($PY -I $D/barge_only.py --room $r --seed $s --tv --tv-at 0.5 --tv-db $db --no-user "$@" 2>&1 | grep -E '^(hit|MISSED)' | cut -c1-200)"
    done
    for at in 1.5 2.5; do
      echo "USER $r s$s at$at: $($PY -I $D/barge_only.py --room $r --seed $s --at $at "$@" 2>&1 | grep -E '^(hit|MISSED|playback cut)' | tr '\n' ' ' | cut -c1-260)"
    done
  done
done

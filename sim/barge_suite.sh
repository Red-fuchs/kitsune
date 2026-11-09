#!/bin/zsh
# barge/TV regression over all rooms: stock barge+TV scenarios + sim/scenarios_barge.yaml. usage: barge_suite.sh <tag> [seeds="0 1"]; KITSUNE_BARGE_VERIFY=0 = old detector (no pause/verify)
D=${0:A:h}; PY=${KITSUNE_PY:-$HOME/.venvs/mlxvlm/bin/python}; T=$1; SEEDS=${2:-"0 1"}
for s in ${=SEEDS}; do
  $PY -I $D/run.py --rooms buero,kueche,hotel,auto,kopfhoerer --seed $s --tag _$T$s --scen-file $D/scenarios_barge.yaml --scen tv_waehrend_antwort,stopp_barge,barge_stopp_morgen,tv_ans_m15,tv_ans_m8,tv_hard_ans,stopp_leise,tv_user_barge
done

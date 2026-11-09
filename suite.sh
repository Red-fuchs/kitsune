#!/bin/zsh
# regression suite: one stateful dialogue (dry-run, no playback). usage: suite.sh [codedir] [out.json]
C=${1:-${0:A:h}}; O=${2:-$C/wavs/out/suite.json}; W=${0:A:h}/wavs/in
exec ${KITSUNE_PY:-$HOME/.venvs/mlxvlm/bin/python} -I $C/kitsune.py --dry-run ask --json $O --wav $W/u00.wav $W/u17.wav $W/u18.wav $W/u19.wav $W/u01.wav $W/u26.wav $W/u27.wav $W/u34.wav \
  $W/u11.wav $W/u12.wav $W/u13.wav $W/u08.wav $W/u09.wav $W/u30.wav $W/u35.wav $W/u03.wav $W/u24.wav $W/u25.wav $W/u23.wav $W/u31.wav $W/u07.wav \
  $W/b20.wav $W/b21.wav $W/b33.wav $W/u22.wav $W/u04.wav $W/u28.wav $W/u29.wav $W/u32.wav $W/u16.wav

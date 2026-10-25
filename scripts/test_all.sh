#!/bin/bash
# Evaluate the released checkpoints on every config. Usage: bash scripts/test_all.sh [gpu]
GPU=${1:-0}
for cfg in configs/{ntu60,ntu120,pku51,kinetics200,kinetics400}/*.yaml; do
  name=$(echo ${cfg#configs/} | sed 's|/|_|; s|.yaml||')
  echo "== $name"
  CUDA_VISIBLE_DEVICES=$GPU python tools/test_zssar.py --config $cfg --checkpoint checkpoints/zssar/$name.pth
done

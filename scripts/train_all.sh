#!/bin/bash
# Train PoseBridge-ZSSAR on every config (Tables 1, 2, 3, 7, 8). Usage: bash scripts/train_all.sh [gpu]
GPU=${1:-0}
for cfg in configs/{ntu60,ntu120,pku51,kinetics200,kinetics400}/*.yaml; do
  CUDA_VISIBLE_DEVICES=$GPU python tools/train_zssar.py --config $cfg
done
python tools/summarize.py

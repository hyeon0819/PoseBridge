<div align="center">

<img src="docs/assets/posebridge-logo.png" alt="PoseBridge Logo" width="240">

# PoseBridge: Bridging the Skeletonization Gap for <br>Zero-Shot Skeleton-Based Action Recognition

[Sanghyeon Lee](https://sites.google.com/view/sang-hyeon-lee) &nbsp;&nbsp;&nbsp; [Jinwoo Kim](https://scholar.google.co.kr/citations?user=WdOolEsAAAAJ&hl=ko&oi=sra) &nbsp;&nbsp;&nbsp; [Jong Taek Lee](https://sites.google.com/view/k-vislab)†

Kyungpook National University, South Korea
<br><sub>† Corresponding author<sub>

### NeurIPS 2026

[![arXiv](https://img.shields.io/badge/arXiv-2605.11497-b31b1b.svg)](https://arxiv.org/abs/2605.11497)
[![Project Page](https://img.shields.io/badge/Project%20Page-PoseBridge-2ea44f.svg)](https://hyeon0819.github.io/PoseBridge/)


**TL;DR:** Skeleton coordinates keep motion, but can lose the cues that tell actions apart. PoseBridge preserves those cues inside the pose estimator and carries them into zero-shot recognition.

</div>

## Installation

```bash
conda create -n posebridge python=3.10 -y && conda activate posebridge
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Shift-GCN CUDA op (only for Shift-GCN training / feature extraction)
cd posebridge/skeleton/ops && python setup.py build_ext --inplace && cd -
```

## Data and checkpoints

Preprocessed data and trained models are on [Hugging Face](https://huggingface.co/hyeon0819/PoseBridge).
Download them into the repository root; the folder layout is kept.

| Path | Content | Size | Used by |
|---|---|---|---|
| `checkpoints/zssar/` | PoseBridge-ZSSAR and its training log, one per config | 0.6 GB | test |
| `data/features/` | Shift-GCN features, one per config | 3.3 GB | test, ZSSAR training |
| `data/pose_cues/` | pose-anchored cues (16 per video) | 14 GB | test, ZSSAR training |
| `checkpoints/hpe/` | PoseBridge-HPE | 37 MB | pose / cue extraction |
| `checkpoints/shiftgcn/` | Shift-GCN, one per config | 0.1 GB | feature extraction |
| `data/pose/` | 2D skeletons from PoseBridge-HPE | 24 GB | Shift-GCN training |

```bash
# test + ZSSAR training, all benchmarks
hf download hyeon0819/PoseBridge --local-dir . \
    --include "checkpoints/zssar/*" --include "data/features/*" --include "data/pose_cues/*"

# or a single benchmark, e.g. NTU-60 48/12 (NTU-60/120 share data/pose_cues/ntu*)
hf download hyeon0819/PoseBridge --local-dir . \
    --include "checkpoints/zssar/ntu60_xsub_seen48_unseen12.pth" \
    --include "data/features/ntu60_xsub_seen48_unseen12.npz" --include "data/pose_cues/ntu*"
```

Each benchmark split has one config, e.g. `configs/ntu60/xsub_seen48_unseen12.yaml`
(`ntu60`, `ntu120`, `pku51`, `kinetics200`, `kinetics400`; random splits end with `_split{1,2,3}`).
NTU / PKU report ZSL and GZSL; Kinetics reports ZSL.

## Quick start (NTU-60 48/12)

Pass any other file in `configs/` to run another benchmark.

**1. Evaluate a released checkpoint**

```bash
python tools/test_zssar.py --config configs/ntu60/xsub_seen48_unseen12.yaml \
    --checkpoint checkpoints/zssar/ntu60_xsub_seen48_unseen12.pth
```

**2. Train PoseBridge-ZSSAR** (~3 min on one GPU)

```bash
python tools/train_zssar.py --config configs/ntu60/xsub_seen48_unseen12.yaml
# -> work_dirs/ntu60/xsub_seen48_unseen12/{train.log, best.pth}
```

`bash scripts/train_all.sh` trains every config; `python tools/summarize.py` collects the results.

**3. Full pipeline** (HPE → poses / cues → Shift-GCN → ZSSAR)

Raw data:
[COCO 2017](https://cocodataset.org/#download) (`data/coco/images/{train2017,val2017}`, `data/coco/annotations`),
[NTU RGB+D 60/120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) RGB videos,
[PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html) (`rgb/`, `label/`, `split/`),
[Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) (`train/`, `val/`).
RTMPose-s initialization:
[cspnext-s_udp-body7_210e-256x192-8c9ccbdb_20230504.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-body7_210e-256x192-8c9ccbdb_20230504.pth)
→ `checkpoints/hpe/`. YOLO11x is downloaded by Ultralytics on first use.

```bash
# (a) PoseBridge-HPE on COCO keypoints + captions
python tools/hpe/encode_captions.py
python tools/hpe/train.py --work-dir work_dirs/hpe        # or use checkpoints/hpe/posebridge_rtmpose-s.pth

# (b) person tracks, 2D poses and pose-anchored cues
python tools/hpe/detect.py  --dataset ntu120 --video-root /path/to/nturgbd_rgb --out-dir work_dirs/ntu_boxes
python tools/hpe/extract.py --dataset ntu120 --video-root /path/to/nturgbd_rgb --box-dir work_dirs/ntu_boxes \
    --out-dir work_dirs/ntu_pose
python tools/pack_pose_cues.py --pose-emb-dir work_dirs/ntu_pose/pose_emb --out data/pose_cues/ntu

# (c) skeleton datasets -> Shift-GCN on seen classes -> features
python tools/build_skeleton_dataset.py --dataset ntu --skeleton-pkl work_dirs/ntu_pose/skeleton_results.pkl
python tools/train_shiftgcn.py --config configs/ntu60/xsub_seen48_unseen12.yaml

# (d) zero-shot recognition
python tools/train_zssar.py --config configs/ntu60/xsub_seen48_unseen12.yaml
```

- NTU-60 and NTU-120 share one run of (b) and (c) (`--dataset ntu120` covers all NTU videos).
- PKU-MMD: `--dataset pku51` with the PKU root as `--video-root` (and `--pku-root` in step c).
- Kinetics: `--dataset kinetics` with the Kinetics-400 root; step (c) builds both Kinetics-200 and -400.
- `detect.py --part k/n` shards long runs across GPUs.
- To skip (a)–(b), download `data/pose/` and pass `data/pose/{ntu,pku51,kinetics}_skeleton_results.pkl` to
  step (c). `python tools/train_shiftgcn.py --config <config> --extract-only` re-extracts features from
  `checkpoints/shiftgcn/`.

## Citation

```bibtex
@inproceedings{lee2026posebridge,
  title     = {PoseBridge: Bridging the Skeletonization Gap for
               Zero-Shot Skeleton-Based Action Recognition},
  author    = {Lee, Sanghyeon and Kim, Jinwoo and Lee, Jong Taek},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2605.11497},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2605.11497}
}
```

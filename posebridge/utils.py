import json
import logging
import os
import random

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPLITS = os.path.join(ROOT, "posebridge", "data", "splits.json")


def load_config(path, overrides=None):
    """YAML config with optional `_base_` inheritance."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        merged = load_config(os.path.join(os.path.dirname(path), base))
        merged.update(cfg)
        cfg = merged
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def config_name(path):
    """configs/ntu60/xsub_seen48_unseen12.yaml -> ntu60/xsub_seen48_unseen12 (file stem outside configs/)."""
    rel = os.path.relpath(os.path.abspath(path), os.path.join(ROOT, "configs"))
    return os.path.splitext(os.path.basename(path) if rel.startswith("..") else rel)[0]


def resolve(path):
    """Paths in configs are relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def class_split(dataset, split):
    """(num_classes, seen, unseen) for e.g. ('ntu60', 'seen48_unseen12')."""
    with open(_SPLITS) as f:
        s = json.load(f)[f"{dataset}_{split}"]
    unseen = sorted(s["unseen"])
    seen = sorted(set(range(s["num_classes"])) - set(unseen))
    return s["num_classes"], seen, unseen


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file=None, name="posebridge"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

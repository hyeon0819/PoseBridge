# Default YACS config for the PoseBridge HPE stage.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 10
_C.PRINT_FREQ = 100
_C.AUTO_RESUME = True
_C.PIN_MEMORY = True
_C.RANK = 0
_C.SEED = 21

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -------------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'posebridge_hpe'
_C.MODEL.SIZE = 'm'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.IMAGE_SIZE = [192, 256]

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.ARCH = 'P5'
_C.MODEL.BACKBONE.EXPAND_RATIO = 0.5
_C.MODEL.BACKBONE.DEEPEN_FACTOR = 0.67
_C.MODEL.BACKBONE.WIDEN_FACTOR = 0.75
_C.MODEL.BACKBONE.OUT_INDICES = (2, 3, 4)
_C.MODEL.BACKBONE.CHANNEL_ATTENTION = True

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.IN_CHANNELS = 768
_C.MODEL.HEAD.FINAL_LAYER_KERNEL_SIZE = 7
_C.MODEL.HEAD.GAU_HIDDEN_DIMS = 256
_C.MODEL.HEAD.GAU_S = 128
_C.MODEL.HEAD.GAU_EXPANSION_FACTOR = 2
_C.MODEL.HEAD.GAU_DROPOUT_RATE = 0.0
_C.MODEL.HEAD.GAU_DROP_PATH = 0.0
_C.MODEL.HEAD.GAU_ACT_FN = 'SiLU'
_C.MODEL.HEAD.GAU_USE_REL_BIAS = False
_C.MODEL.HEAD.GAU_POS_ENC = False

_C.MODEL.SIMCC = CN()
_C.MODEL.SIMCC.SPLIT_RATIO = 2.0
_C.MODEL.SIMCC.SIGMA = (4.9, 5.66)
_C.MODEL.SIMCC.NORMALIZE = False
_C.MODEL.SIMCC.USE_DARK = False

_C.MODEL.ROUTE = CN()
_C.MODEL.ROUTE.NUM_SCALES = 2
_C.MODEL.ROUTE.PROJ_DIM = 512
_C.MODEL.ROUTE.DROPOUT = 0.1
_C.MODEL.ROUTE.USE_GEM = True
_C.MODEL.ROUTE.NORMALIZE = True
_C.MODEL.ROUTE.FIXED_GATE = -1.0
_C.MODEL.ROUTE.DETACH_SHALLOW = False

# -------------------------------------------------------------------------
# LOSS
# -------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.TYPE = 'KLDiscretLoss'
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.BETA = 10.0
_C.LOSS.LABEL_SOFTMAX = True

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = 'data/coco/'
_C.DATASET.DATASET = 'coco'
_C.DATASET.TRAIN_SET = 'train2017'
_C.DATASET.TEST_SET = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'

_C.DATASET.MEAN = [123.675, 116.28, 103.53]
_C.DATASET.STD = [58.395, 57.12, 57.375]
_C.DATASET.BGR_TO_RGB = True

_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = [0.6, 1.4]
_C.DATASET.ROT_FACTOR = 80
_C.DATASET.HALF_BODY_PROB = 0.5

_C.DATASET.STAGE2_SCALE_FACTOR = [0.75, 1.25]
_C.DATASET.STAGE2_ROT_FACTOR = 60
_C.DATASET.STAGE2_SHIFT_FACTOR = 0.0

_C.DATASET.BLUR_PROB = 0.1
_C.DATASET.MEDIAN_BLUR_PROB = 0.1
_C.DATASET.COARSE_DROPOUT_PROB = 1.0
_C.DATASET.COARSE_DROPOUT_STAGE2_PROB = 0.5

_C.DATASET.HSV_H_GAIN = 0.015
_C.DATASET.HSV_S_GAIN = 0.7
_C.DATASET.HSV_V_GAIN = 0.4

_C.DATASET.CAPTION_INDEX = 0
_C.DATASET.CLIP_TEXT_PREEXTRACT = False

# -------------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.LR = 4e-3
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.MOMENTUM = 0.9

_C.TRAIN.CLIP_GRAD = True
_C.TRAIN.CLIP_MAX_NORM = 35.0
_C.TRAIN.CLIP_NORM_TYPE = 2

_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.WARMUP_ITERS = 1000
_C.TRAIN.WARMUP_FACTOR = 1e-5
_C.TRAIN.ETA_MIN_FACTOR = 0.05

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 420
_C.TRAIN.STAGE2_EPOCH = 390
_C.TRAIN.VAL_INTERVAL = 10

_C.TRAIN.BATCH_SIZE_PER_GPU = 256
_C.TRAIN.SHUFFLE = True

_C.TRAIN.USE_EMA = True
_C.TRAIN.EMA_MOMENTUM = 0.0002

_C.TRAIN.AUTO_SCALE_LR = True
_C.TRAIN.BASE_BATCH_SIZE = 1024

_C.TRAIN.CLIP_TEMPERATURE = 0.07
_C.TRAIN.CLIP_TEMPERATURE_TRAINABLE = True
_C.TRAIN.CLIP_LOSS_WEIGHT = 0.1

_C.TRAIN.TEXT_ENCODER = CN()
_C.TRAIN.TEXT_ENCODER.CLIP_MODEL = 'ViT-L-14-336'
_C.TRAIN.TEXT_ENCODER.OPENCLIP_PRETRAINED = 'openai'
_C.TRAIN.TEXT_ENCODER.PROJ_DIM = 512
_C.TRAIN.TEXT_ENCODER.PROJ_HEAD = True
_C.TRAIN.TEXT_ENCODER.PROJ_NUM_LAYER = 1
_C.TRAIN.TEXT_ENCODER.PROJ_HIDDEN_DIM = 512

# -------------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 64
_C.TEST.FLIP_TEST = True
_C.TEST.USE_GT_BBOX = True
_C.TEST.OKS_THRE = 0.9
_C.TEST.IN_VIS_THRE = 0.2
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.IMAGE_THRE = 0.0
_C.TEST.NMS_THRE = 1.0
_C.TEST.SOFT_NMS = False

# -------------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------------
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False


cfg = _C.clone()


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if hasattr(args, 'opts') and args.opts:
        cfg.merge_from_list(args.opts)

    if hasattr(args, 'output_dir') and args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if cfg.DATA_DIR:
        cfg.DATASET.ROOT = os.path.join(cfg.DATA_DIR, cfg.DATASET.ROOT)

    if cfg.MODEL.PRETRAINED and cfg.DATA_DIR:
        cfg.MODEL.PRETRAINED = os.path.join(cfg.DATA_DIR, cfg.MODEL.PRETRAINED)

    cfg.freeze()

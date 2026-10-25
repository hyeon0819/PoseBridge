# ------------------------------------------------------------------------------
# COCO Dataset for RTMPose
# Based on mmpose RTMPose official implementation
# Includes SimCC label generation and data augmentation pipeline
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import random
import logging
from collections import OrderedDict, defaultdict
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json


logger = logging.getLogger(__name__)


class COCOCaptionDataset(Dataset):
    """COCO Dataset for RTMPose with SimCC representation.
    
    Implements the full RTMPose preprocessing pipeline:
    - RandomFlip
    - RandomHalfBody
    - RandomBBoxTransform (scale, rotation)
    - TopdownAffine
    - YOLOXHSVRandomAug
    - Albumentation (Blur, MedianBlur, CoarseDropout)
    - SimCC label generation
    
    Args:
        cfg: Configuration object.
        root (str): Root directory of the dataset.
        image_set (str): Image set name (e.g., 'train2017', 'val2017').
        is_train (bool): Whether in training mode.
        transform: Image transforms (optional, not used as transforms are built-in).
    """
    
    # COCO keypoint information
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    
    UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]
    
    def __init__(
        self,
        cfg,
        root: str,
        image_set: str,
        is_train: bool,
        transform=None,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.root = root
        self.image_set = image_set
        self.is_train = is_train
        
        # Image and feature map sizes
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.input_size = (self.image_width, self.image_height)  # (w, h)
        self.aspect_ratio = self.image_width / self.image_height
        self.pixel_std = 200
        
        # SimCC settings
        self.simcc_split_ratio = cfg.MODEL.SIMCC.SPLIT_RATIO
        self.sigma = np.array(cfg.MODEL.SIMCC.SIGMA)
        self.normalize = cfg.MODEL.SIMCC.NORMALIZE
        
        # Data preprocessing settings (from mmpose config)
        self.mean = np.array(cfg.DATASET.MEAN)
        self.std = np.array(cfg.DATASET.STD)
        self.bgr_to_rgb = cfg.DATASET.BGR_TO_RGB
        
        # Augmentation settings (Stage 1 - default)
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.half_body_prob = cfg.DATASET.HALF_BODY_PROB
        self.shift_factor = 0.0  # Default shift factor
        
        # Stage 2 augmentation settings (from mmpose PipelineSwitchHook)
        # Activated in the last 30 epochs (epoch 390~420)
        self.stage2_scale_factor = cfg.DATASET.STAGE2_SCALE_FACTOR
        self.stage2_rot_factor = cfg.DATASET.STAGE2_ROT_FACTOR
        self.stage2_shift_factor = cfg.DATASET.STAGE2_SHIFT_FACTOR
        self.stage2_coarse_dropout_prob = cfg.DATASET.COARSE_DROPOUT_STAGE2_PROB
        self.is_stage2 = False  # Flag to switch augmentation
        
        # HSV augmentation
        self.hsv_h_gain = cfg.DATASET.HSV_H_GAIN
        self.hsv_s_gain = cfg.DATASET.HSV_S_GAIN
        self.hsv_v_gain = cfg.DATASET.HSV_V_GAIN
        
        # Albumentation settings
        self.blur_prob = cfg.DATASET.BLUR_PROB
        self.median_blur_prob = cfg.DATASET.MEDIAN_BLUR_PROB
        self.coarse_dropout_prob = cfg.DATASET.COARSE_DROPOUT_PROB
        
        # Test settings
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.nms_thre = cfg.TEST.NMS_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        
        self.num_joints = cfg.MODEL.NUM_JOINTS
        
        # Load COCO annotations
        self.coco = COCO(self._get_ann_file())
        
        # Load class information
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:]
        ])
        
        # Load captions
        self.caption_index = getattr(cfg.DATASET, 'CAPTION_INDEX', 0)  # Default to first caption (0-4)
        self.captions_dict = self._load_captions()
        
        # Load image indices
        self.image_set_index = self._load_image_set_index()
        
        # Load database
        self.db = self._get_db()
        
        logger.info(f'=> Load {len(self.db)} samples from {image_set}')
        if self.captions_dict:
            logger.info(f'=> Load {len(self.captions_dict)} images with captions')
    
    def _get_ann_file(self) -> str:
        """Get annotation file path."""
        prefix = 'person_keypoints' if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            f'{prefix}_{self.image_set}.json'
        )
    
    def _load_captions(self) -> dict:
        """Load COCO captions.
        
        Returns:
            dict: Mapping from image_id to list of captions (5 captions per image).
        """
        caption_file = os.path.join(
            self.root,
            'annotations',
            f'captions_{self.image_set}.json'
        )
        
        if not os.path.exists(caption_file):
            logger.warning(f'Caption file not found: {caption_file}')
            return {}
        
        try:
            import json
            with open(caption_file, 'r') as f:
                caption_data = json.load(f)
            
            # Build image_id -> captions mapping
            captions_dict = defaultdict(list)
            for ann in caption_data['annotations']:
                image_id = ann['image_id']
                caption = ann['caption']
                captions_dict[image_id].append(caption)
            
            # Convert to regular dict
            captions_dict = dict(captions_dict)
            
            logger.info(f'=> Loaded captions from {caption_file}')
            
            # Check caption counts
            caption_counts = [len(caps) for caps in captions_dict.values()]
            if caption_counts:
                logger.info(f'=> Caption count per image: min={min(caption_counts)}, max={max(caption_counts)}, avg={np.mean(caption_counts):.1f}')
            
            return captions_dict
            
        except Exception as e:
            logger.error(f'Error loading captions from {caption_file}: {e}')
            return {}
    
    def _load_image_set_index(self) -> List[int]:
        """Load image ids."""
        return self.coco.getImgIds()
    
    def _get_db(self) -> List[dict]:
        """Load database."""
        if self.is_train or self.use_gt_bbox:
            return self._load_coco_keypoint_annotations()
        else:
            return self._load_coco_person_detection_results()
    
    def _load_coco_keypoint_annotations(self) -> List[dict]:
        """Load ground truth annotations."""
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(index))
        return gt_db
    
    def _load_coco_keypoint_annotation_kernel(self, index: int) -> List[dict]:
        """Load annotation for a single image."""
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        
        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)
        
        # Sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        
        rec = []
        for obj in valid_objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue
            
            # Skip objects without keypoint annotation
            if max(obj['keypoints']) == 0:
                continue
            
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
            
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0
            
            center, scale = self._box2cs(obj['clean_bbox'][:4])
            
            rec.append({
                'image': self._image_path_from_index(index),
                'image_id': index,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })
        
        return rec
    
    def _load_coco_person_detection_results(self) -> List[dict]:
        """Load detection results for testing."""
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)
        
        if not all_boxes:
            logger.error(f'=> Load {self.bbox_file} fail!')
            return []
        
        logger.info(f'=> Total boxes: {len(all_boxes)}')
        
        kpt_db = []
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue
            
            img_name = self._image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']
            
            if score < self.image_thre:
                continue
            
            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float32)
            
            kpt_db.append({
                'image': img_name,
                'image_id': det_res['image_id'],
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })
        
        logger.info(f'=> Total boxes after filter: {len(kpt_db)}')
        return kpt_db
    
    def _box2cs(self, box: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert box to center and scale."""
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)
    
    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert xywh to center and scale."""
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        
        if center[0] != -1:
            scale = scale * 1.25
        
        return center, scale
    
    def _image_path_from_index(self, index: int) -> str:
        """Get image path from index."""
        file_name = f'{index:012d}.jpg'
        
        prefix = 'test2017' if 'test' in self.image_set else self.image_set
        
        image_path = os.path.join(self.root, 'images', prefix, file_name)
        
        return image_path
    
    def set_stage2(self, enable: bool = True):
        """Switch to stage2 augmentation pipeline.
        
        Stage2 (last 30 epochs: 390~420) uses different augmentation:
        - scale_factor: [0.75, 1.25] (instead of [0.6, 1.4])
        - rotate_factor: 60 (instead of 80)
        - shift_factor: 0.0
        - CoarseDropout probability: 0.5 (instead of 1.0)
        
        Args:
            enable (bool): Whether to enable stage2 augmentation.
        """
        self.is_stage2 = enable
        if enable:
            logger.info('=> Switched to Stage2 augmentation pipeline')
    
    def __len__(self) -> int:
        return len(self.db)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Get item.
        
        Returns:
            input (Tensor): Preprocessed image. Shape: [C, H, W].
            target_x (Tensor): SimCC label for x-axis. Shape: [K, W*split_ratio].
            target_y (Tensor): SimCC label for y-axis. Shape: [K, H*split_ratio].
            target_weight (Tensor): Keypoint weights. Shape: [K].
            meta (dict): Metadata including center, scale, image path, etc.
        """
        db_rec = copy.deepcopy(self.db[idx])
        
        # Load image
        image_file = db_rec['image']
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR)
        
        if data_numpy is None:
            logger.error(f'=> fail to read {image_file}')
            raise ValueError(f'Fail to read {image_file}')
        
        if self.bgr_to_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        
        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        
        if self.is_train:
            # Random flip
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :].copy()
                joints, joints_vis = self._flip_joints(
                    joints, joints_vis, data_numpy.shape[1])
                c[0] = data_numpy.shape[1] - c[0] - 1
            
            # Random half body
            if self.half_body_prob > 0 and random.random() < self.half_body_prob:
                c_half_body, s_half_body = self._half_body_transform(
                    joints, joints_vis)
                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body
            
            # Random scale and rotation (RandomBBoxTransform)
            # Use stage2 augmentation if enabled (last 30 epochs: 390~420)
            if self.is_stage2:
                sf = self.stage2_scale_factor
                rf = self.stage2_rot_factor
                shift_factor = self.stage2_shift_factor
            else:
                sf = self.scale_factor
                rf = self.rot_factor
                shift_factor = self.shift_factor
            
            # Scale factor: uniform in [sf[0], sf[1]]
            s_factor = np.random.uniform(sf[0], sf[1])
            s = s * s_factor
            
            # Rotation factor: uniform in [-rf, rf]
            r = np.random.uniform(-rf, rf)
            
            # Shift factor (stage2 has shift_factor=0)
            if shift_factor > 0:
                dx = np.random.uniform(-shift_factor, shift_factor) * s[0] * self.pixel_std
                dy = np.random.uniform(-shift_factor, shift_factor) * s[1] * self.pixel_std
                c[0] += dx
                c[1] += dy
        
        # Affine transform
        trans = get_affine_transform(c, s, r, self.input_size)
        input_img = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )
        
        # Transform keypoints
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        if self.is_train:
            # HSV augmentation (YOLOXHSVRandomAug)
            input_img = self._hsv_augment(input_img)
            
            # Albumentation augmentation
            input_img = self._albumentation_augment(input_img)
        
        # Normalize image
        input_img = input_img.astype(np.float32)
        input_img = (input_img - self.mean) / self.std
        
        # To tensor: [H, W, C] -> [C, H, W]
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        
        # Generate SimCC labels
        keypoints = joints[:, :2]
        keypoints_visible = joints_vis[:, 0]
        
        target_x, target_y, target_weight = self._generate_simcc_label(
            keypoints, keypoints_visible)
        
        target_x = torch.from_numpy(target_x).float()
        target_y = torch.from_numpy(target_y).float()
        target_weight = torch.from_numpy(target_weight).float()
        
        # Get captions for this image
        image_id = db_rec.get('image_id', None)
        caption = None
        if image_id is not None and image_id in self.captions_dict:
            captions = self.captions_dict[image_id]
            if captions:
                if self.is_train:
                    # Training: randomly select one caption from 5
                    caption = random.choice(captions)
                else:
                    # Testing: return all captions or select by index
                    if self.caption_index == -1:
                        # Return all captions
                        caption = captions
                    else:
                        # Return caption at specified index (0-4)
                        idx = min(self.caption_index, len(captions) - 1)
                        caption = captions[idx]
        
        meta = {
            'image': image_file,
            'image_id': image_id,
            'filename': '',
            'imgnum': 0,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'flip_indices': self.FLIP_INDICES,
            'caption': caption,
        }
        
        return input_tensor, target_x, target_y, target_weight, meta
    
    def _flip_joints(
        self,
        joints: np.ndarray,
        joints_vis: np.ndarray,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flip joints horizontally."""
        joints = joints.copy()
        joints_vis = joints_vis.copy()
        
        # Flip x coordinate
        joints[:, 0] = width - joints[:, 0] - 1
        
        # Swap left-right joints
        for pair in self.FLIP_PAIRS:
            joints[pair[0], :], joints[pair[1], :] = \
                joints[pair[1], :].copy(), joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :].copy(), joints_vis[pair[0], :].copy()
        
        return joints, joints_vis
    
    def _half_body_transform(
        self,
        joints: np.ndarray,
        joints_vis: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Random half body transform."""
        upper_joints = []
        lower_joints = []
        
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id, 0] > 0:
                if joint_id in self.UPPER_BODY_IDS:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])
        
        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            return None, None
        
        selected_joints = np.array(selected_joints, dtype=np.float32)
        
        center = selected_joints.mean(axis=0)[:2]
        
        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)
        
        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]
        
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32
        )
        scale = scale * 1.5
        
        return center, scale
    
    def _hsv_augment(self, img: np.ndarray) -> np.ndarray:
        """HSV augmentation (YOLOXHSVRandomAug)."""
        hsv_gains = np.array([self.hsv_h_gain, self.hsv_s_gain, self.hsv_v_gain])
        r = np.random.uniform(-1, 1, 3) * hsv_gains + 1
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        img_hsv[..., 0] = img_hsv[..., 0] * r[0]
        img_hsv[..., 1] = img_hsv[..., 1] * r[1]
        img_hsv[..., 2] = img_hsv[..., 2] * r[2]
        
        img_hsv[img_hsv > 255] = 255
        img_hsv[img_hsv < 0] = 0
        
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return img
    
    def _albumentation_augment(self, img: np.ndarray) -> np.ndarray:
        """Albumentation augmentation (Blur, MedianBlur, CoarseDropout)."""
        # Blur
        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5, 7])
            img = cv2.blur(img, (ksize, ksize))
        
        # MedianBlur
        if random.random() < self.median_blur_prob:
            ksize = random.choice([3, 5])
            img = cv2.medianBlur(img, ksize)
        
        # CoarseDropout - use different probability for stage2
        dropout_prob = self.stage2_coarse_dropout_prob if self.is_stage2 else self.coarse_dropout_prob
        if random.random() < dropout_prob:
            h, w = img.shape[:2]
            
            # Random hole size (20% ~ 40% of image size)
            hole_h = int(h * np.random.uniform(0.2, 0.4))
            hole_w = int(w * np.random.uniform(0.2, 0.4))
            
            # Random position
            y = np.random.randint(0, h - hole_h)
            x = np.random.randint(0, w - hole_w)
            
            img[y:y+hole_h, x:x+hole_w] = 0
        
        return img
    
    def _generate_simcc_label(
        self,
        keypoints: np.ndarray,
        keypoints_visible: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate SimCC labels with Gaussian smoothing.
        
        Args:
            keypoints (np.ndarray): Keypoint coordinates. Shape: [K, 2].
            keypoints_visible (np.ndarray): Keypoint visibility. Shape: [K].
        
        Returns:
            x_labels (np.ndarray): SimCC labels for x-axis. Shape: [K, W*split_ratio].
            y_labels (np.ndarray): SimCC labels for y-axis. Shape: [K, H*split_ratio].
            keypoint_weights (np.ndarray): Keypoint weights. Shape: [K].
        """
        K = len(keypoints)
        w, h = self.input_size
        W = int(np.around(w * self.simcc_split_ratio))
        H = int(np.around(h * self.simcc_split_ratio))
        
        # Map coordinates to SimCC space
        keypoints_split = np.around(keypoints * self.simcc_split_ratio).astype(np.int64)
        keypoint_weights = keypoints_visible.copy()
        
        target_x = np.zeros((K, W), dtype=np.float32)
        target_y = np.zeros((K, H), dtype=np.float32)
        
        # 3-sigma rule
        radius = self.sigma * 3
        
        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)
        
        for k in range(K):
            # Skip unlabeled keypoints
            if keypoints_visible[k] < 0.5:
                continue
            
            mu = keypoints_split[k]
            mu_x, mu_y = mu[0], mu[1]
            
            # Check that the gaussian has in-bounds part
            # radius is [radius_x, radius_y]
            left = mu_x - radius[0]
            top = mu_y - radius[1]
            right = mu_x + radius[0] + 1
            bottom = mu_y + radius[1] + 1
            
            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[k] = 0
                continue
            
            target_x[k] = np.exp(-((x - mu_x) ** 2) / (2 * self.sigma[0] ** 2))
            target_y[k] = np.exp(-((y - mu_y) ** 2) / (2 * self.sigma[1] ** 2))
        
        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]
        
        return target_x, target_y, keypoint_weights
    
    def evaluate(
        self,
        cfg,
        preds: np.ndarray,
        output_dir: str,
        all_boxes: np.ndarray,
        img_path: List[str],
        *args, **kwargs
    ) -> Tuple[dict, float]:
        """Evaluate predictions using COCO API.
        
        Args:
            cfg: Configuration object.
            preds (np.ndarray): Predictions. Shape: [N, K, 3].
            output_dir (str): Output directory.
            all_boxes (np.ndarray): Bounding boxes. Shape: [N, 6].
            img_path (List[str]): Image paths.
        
        Returns:
            name_value (dict): Metric names and values.
            perf_indicator (float): Main performance metric (AP).
        """
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        
        res_file = os.path.join(
            res_folder,
            f'keypoints_{self.image_set}_results_{cfg.RANK}.json'
        )
        
        # Person x keypoints
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        
        # Image x person x keypoints
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)
        
        # Rescoring and OKS NMS
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score += t_s
                        valid_num += 1
                if valid_num != 0:
                    kpt_score /= valid_num
                # Rescoring
                n_p['score'] = kpt_score * box_score
            
            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])
        
        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
        
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0
    
    def _write_coco_keypoint_results(self, keypoints: List, res_file: str):
        """Write results to JSON file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind['person'],
            'cls_ind': 1,
            'cls': 'person',
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }]
        
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        
        logger.info(f'=> writing results json to {res_file}')
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
    
    def _coco_keypoint_results_one_category_kernel(self, data_pack: dict) -> List[dict]:
        """Generate results for one category."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        
        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue
            
            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            key_points = np.zeros((_key_points.shape[0], self.num_joints * 3), dtype=np.float32)
            
            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]
            
            result = [{
                'image_id': img_kpts[k]['image'],
                'category_id': cat_id,
                'keypoints': list(key_points[k]),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            } for k in range(len(img_kpts))]
            
            cat_results.extend(result)
        
        return cat_results
    
    def _do_python_keypoint_eval(self, res_file: str, res_folder: str) -> List[Tuple[str, float]]:
        """Evaluate using COCO API."""
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        
        return info_str


# Helper functions

def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: np.ndarray = np.array([0, 0], dtype=np.float32),
    inv: bool = False,
) -> np.ndarray:
    """Get affine transform matrix."""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    
    return trans


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply affine transform to a point."""
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the third point from two points."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
    """Get direction vector."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    
    return np.array(src_result)


def oks_nms(kpts_db: List[dict], thresh: float) -> List[int]:
    """OKS-based NMS."""
    if len(kpts_db) == 0:
        return []
    
    scores = np.array([kpt['score'] for kpt in kpts_db])
    areas = np.array([kpt['area'] for kpt in kpts_db])
    kpts = np.array([kpt['keypoints'].flatten() for kpt in kpts_db])
    
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        oks = compute_oks(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]])
        
        inds = np.where(oks <= thresh)[0]
        order = order[inds + 1]
    
    return keep


def soft_oks_nms(kpts_db: List[dict], thresh: float) -> List[int]:
    """Soft OKS-based NMS."""
    # For simplicity, use same as oks_nms
    return oks_nms(kpts_db, thresh)


def compute_oks(kpt1: np.ndarray, kpts: np.ndarray, area1: float, areas: np.ndarray) -> np.ndarray:
    """Compute OKS between one keypoint and multiple keypoints."""
    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
    ]) / 10.0
    vars = (sigmas * 2) ** 2
    
    # kpt1: [K*3], kpts: [N, K*3]
    N = kpts.shape[0]
    K = len(sigmas)
    
    kpt1 = kpt1.reshape(K, 3)
    kpts = kpts.reshape(N, K, 3)
    
    xg, yg, vg = kpt1[:, 0], kpt1[:, 1], kpt1[:, 2]
    xd, yd = kpts[:, :, 0], kpts[:, :, 1]
    
    dx = xd - xg[None, :]
    dy = yd - yg[None, :]
    
    e = (dx ** 2 + dy ** 2) / vars / (areas[:, None] + np.spacing(1)) / 2
    e = np.sum(np.exp(-e) * (vg > 0)[None, :], axis=1) / np.sum(vg > 0)
    
    return e

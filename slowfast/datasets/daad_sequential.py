#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from functools import partial
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.utils.data
from slowfast.utils.env import pathmgr
from torchvision import transforms
import slowfast.utils.logging as logging

from . import (
    decoder as decoder,
    transform as transform,
    utils as utils,
    video_container as container,
)
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, MaskingGenerator, MaskingGenerator3D

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Daadsequential(torch.utils.data.Dataset):
    """
    DAAD Sequential video loader. Construct the DAAD Sequential video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformly sampled from every
    video with uniform cropping.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the DAAD Sequential video loader with a given csv file.
        
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Split '{mode}' not supported for DAAD Sequential"
        
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
        )
        self.dummy_output = None
        
        # Set number of clips based on mode
        self._num_clips = 1 if self.mode in ["train", "val"] else (
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        )

        logger.info(f"Constructing DAAD {mode}...")
        self._construct_loader()
        
        # Set augmentation flags
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0
        self.synchronize_across_views = self.cfg.DATA.SYNC_VIEWS

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """Construct the video loader."""
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, f"{self.mode}.csv"
        )
        assert pathmgr.exists(path_to_file), f"{path_to_file} dir not found"

        # Initialize video path lists
        self._path_to_videos = {
            'front': [],
            'left': [],
            'right': [],
            'rear': [],
            'driver': [],
            'gaze': []
        }
        
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0

        with pathmgr.open(path_to_file, "r") as f:
            lines = [x.strip() for x in f.readlines()]
            for clip_idx, row in enumerate(lines):
                fetch_info = row.split(" ")
                assert len(fetch_info) == 7, f"Invalid format for file: {path_to_file}"
                
                paths, label = fetch_info[:-1], fetch_info[-1]
                
                view_names = ['front', 'left', 'right', 'rear', 'driver', 'gaze']
                
                for idx in range(self._num_clips):
                    # Add all video paths with prefix
                    for view_idx, view_name in enumerate(view_names):
                        self._path_to_videos[view_name].append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, paths[view_idx])
                        )
                    
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        # Verify all view videos exist
        for view_name in view_names:
            assert len(self._path_to_videos[view_name]) > 0, \
                f"Failed to load DAAD Sequential {view_name} videos"
        
        total_videos = len(self._path_to_videos['front'])
        logger.info(f"Constructing DAAD Sequential dataloader (size: {total_videos}, "
                   f"skip_rows {self.skip_rows}) from {path_to_file}.")

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _load_video_container(self, index, view_name):
        """
        Helper function to load a single video container.
        Returns the container or None if failed.
        """
        try:
            video_path = self._path_to_videos[view_name][index]
            video_container = container.get_video_container(
                video_path,
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )
            if video_container is None:
                logger.warning(f"Failed to meta load {view_name} video idx {index} from {video_path}")
                return None
            return video_container
        except Exception as e:
            logger.info(f"Failed to load {view_name} video from {self._path_to_videos[view_name][index]} with error {e}")
            return None

    def _decode_video(self, index, container, sampling_rate, num_frames, 
                     temporal_sample_index, target_fps, min_scale,
                     get_time_idx_only=False, time_idx_override=None):
        """Helper function to decode a single video view."""
        if container is None:
            return None, None
            
        frames, time_idx, _ = decoder.decode(
            container,
            sampling_rate,
            num_frames,
            temporal_sample_index,
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            video_meta=(self._video_meta[index] if len(self._video_meta) < 5e6 else {}),
            target_fps=target_fps,
            backend=self.cfg.DATA.DECODING_BACKEND,
            use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            max_spatial_scale=(min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0),
            time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
            temporally_rnd_clips=True,
            min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
            max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            get_time_idx_only=get_time_idx_only,
            time_idx_override=time_idx_override,
        )
        return frames, time_idx

    def _process_frame(self, frame, time_idx, idx, i, spatial_sample_index, min_scale, 
                      max_scale, crop_size, view_name, relative_scales, relative_aspect):
        """Process a single frame with all augmentations."""
        if frame is None:
            return None, None
            
        f_out = frame.clone()
        time_idx_out = time_idx[i, :]
        f_out = f_out.float() / 255.0

        # Apply color jitter if in training mode and enabled
        if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
            f_out = transform.color_jitter_video_ssl(
                f_out,
                bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                hue=self.cfg.DATA.SSL_COLOR_HUE,
                p_convert_gray=self.p_convert_gray,
                moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
            )

        # Apply auto augment if enabled
        if self.aug and self.cfg.AUG.AA_TYPE:
            aug_transform = create_random_augment(
                input_size=(f_out.size(1), f_out.size(2)),
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
            )
            # T H W C -> T C H W
            f_out = f_out.permute(0, 3, 1, 2)
            list_img = self._frame_to_list_img(f_out)
            list_img = aug_transform(list_img)
            f_out = self._list_img_to_frames(list_img)
            f_out = f_out.permute(0, 2, 3, 1)

        # Normalize
        f_out = utils.tensor_normalize(f_out, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        
        # T H W C -> C T H W
        f_out = f_out.permute(3, 0, 1, 2)

        # Spatial sampling
        f_out = utils.spatial_sampling(
            f_out,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale[i],
            max_scale=max_scale[i],
            crop_size=crop_size[i],
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=(self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode in ["train"] else False),
        )

        # Random erasing if enabled
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            f_out = erase_transform(f_out.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # Pack pathway output
        f_out = utils.pack_pathway_output(self.cfg, f_out)

        # Add mask if enabled
        if self.cfg.AUG.GEN_MASK_LOADER:
            mask = self._gen_mask()
            f_out = f_out + [torch.Tensor(), mask]

        return f_out, time_idx_out

    def __getitem__(self, index):
        """
        Get item from the dataset with proper error handling and retries.
        """
        short_cycle_idx = None

        # Handle tuple index case
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
                
        if self.dummy_output is not None:
            return self.dummy_output
        
        # Set up sampling parameters based on mode
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1  # Random sampling
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE

            # Handle short cycle case
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )

            # Scale adjustment for multigrid
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )

        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )

            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )

            # Ensure consistency for test mode
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(f"Does not support {self.mode} mode")
            
        # Set up decoding parameters
        num_decode = self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode in ["train"] else 1
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        
        # Extend parameters if needed
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (num_decode - len(min_scale))
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (num_decode - len(max_scale))
            
            if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE:
                crop_size_extension = [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
            else:
                crop_size_extension = [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
                
            crop_size += crop_size_extension
            assert self.mode in ["train", "val"]

        # Try to decode videos with proper retry logic
        view_names = ['front', 'left', 'right', 'rear', 'driver', 'gaze']
        
        for i_try in range(self._num_retries):
            # Step 1: Load all video containers in parallel
            video_containers = {}
            all_containers_valid = True
            
            for view_name in view_names:
                video_containers[view_name] = self._load_video_container(index, view_name)
                if video_containers[view_name] is None:
                    all_containers_valid = False
                    if self.mode not in ["test"] and i_try > self._num_retries // 8:
                        # Try another random video
                        index = random.randint(0, len(self._path_to_videos['front']) - 1)
                    break
            
            if not all_containers_valid:
                continue
                
            # Setup decoding parameters
            num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            sampling_rate = [sampling_rate]
            
            # Adjust parameters if needed
            if len(num_frames) < num_decode:
                num_frames.extend([num_frames[-1] for _ in range(num_decode - len(num_frames))])
                sampling_rate.extend([sampling_rate[-1] for _ in range(num_decode - len(sampling_rate))])
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            # Verify parameter lengths in training mode
            if self.mode in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            # Set target FPS with jitter for training
            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

            # Step 2: Decode all videos
            frames_decoded = OrderedDict()
            time_idx_decoded = OrderedDict()
            
            all_decoded_valid = True
            
            if self.synchronize_across_views:
                candidate_view = 'front'
                _, sync_time_idx = self._decode_video(
                        index,
                        video_containers[candidate_view], 
                        sampling_rate,
                        num_frames,
                        temporal_sample_index,
                        target_fps,
                        min_scale,
                        get_time_idx_only=True,
                    )
            else:
                sync_time_idx = None

            for view_name in view_names:
                frames, time_idx = self._decode_video(
                    index,
                    video_containers[view_name], 
                    sampling_rate,
                    num_frames,
                    temporal_sample_index,
                    target_fps,
                    min_scale,
                    get_time_idx_only=False,
                    time_idx_override=sync_time_idx,
                )
                
                if frames is None or None in frames:
                    all_decoded_valid = False
                    if self.mode not in ["test"] and (i_try % (self._num_retries // 8)) == 0:
                        index = random.randint(0, len(self._path_to_videos['front']) - 1)
                    break
                
                frames_decoded[view_name] = frames
                time_idx_decoded[view_name] = time_idx
            
            if not all_decoded_valid:
                continue
            
            # If we got here, all videos were successfully decoded
            # Set up augmentation parameters
            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            
            # Initialize output arrays
            f_outs = {view_name: [None] * num_out for view_name in view_names}
            time_idx_outs = {view_name: [None] * num_out for view_name in view_names}
            
            # Get relative scales and aspect ratios
            scl, asp = (
                self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
            )
            relative_scales = None if (self.mode not in ["train"] or len(scl) == 0) else scl
            relative_aspect = None if (self.mode not in ["train"] or len(asp) == 0) else asp
            
            # Process all frames with augmentations
            # idx = -1
            label = self._labels[index]
            
            for i in range(num_decode):
                # Process all views with the same augmentation
                for view_name in view_names:
                    for idx in range(num_out):
                        f_outs[view_name][idx], time_idx_outs[view_name][idx] = self._process_frame(
                            frames_decoded[view_name][i],
                            time_idx_decoded[view_name],
                            idx, i, spatial_sample_index,
                            min_scale, max_scale, crop_size,
                            view_name, relative_scales, relative_aspect
                        )
            
            # Prepare final output
            frames_out = OrderedDict()
            time_idx_out = OrderedDict()
            
            for view_name in view_names:
                frames_out[view_name] = f_outs[view_name][0] if num_out == 1 else f_outs[view_name]
                time_idx_out[view_name] = [x.tolist() for x in time_idx_outs[view_name]]
            
            # Adjust label and index for multi-augmentation case
            if (num_aug * num_decode > 1 and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode

            
            output_frames = []
            for view_name in view_names:
                for tensor in frames_out[view_name]:
                    output_frames.append(tensor)
            output_frames = tuple(output_frames)
            
            output_time_indices = []
            for view_name in view_names:
                inview = []
                for time_tensor in time_idx_out[view_name]:
                    inview.append(time_tensor)

                if self.synchronize_across_views:
                    # Assert all tensors in `inview` are equal
                    first = inview[0]
                    assert all(t == first for t in inview), \
                        f"Not all time tensors in view '{view_name}' are equal"

                output_time_indices.append(torch.tensor(inview, dtype=torch.float64))

            # Handle dummy output case
            if self.cfg.DATA.DUMMY_LOAD and self.dummy_output is None:
                self.dummy_output = (
                    output_frames,
                    label,
                    index,
                    output_time_indices,
                    {}
                )
            
            return (
                output_frames,
                label,
                index,
                output_time_indices,
                {}
            )
            
        # Failed to fetch video after all retries
        logger.warning(f"Failed to fetch video after {self._num_retries} retries.")
        #return None

    def _gen_mask(self):
        """Generate mask for masking augmentation."""
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=int)
            n_mask = round(self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO)
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        """Convert tensor frames to list of PIL images."""
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        """Convert list of PIL images to tensor frames."""
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """Returns the number of videos in the dataset."""
        return len(self._path_to_videos['front'])

    @property
    def num_videos(self):
        """Returns the number of videos in the dataset."""
        return len(self._path_to_videos['front'])

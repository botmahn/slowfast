#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random

import numpy as np
import pandas

import slowfast.utils.logging as logging
import torch
import torch.utils.data
from slowfast.utils.env import pathmgr
from torchvision import transforms

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
class Daad(torch.utils.data.Dataset):
    """
    DAAD video loader. Construct the DAAD video loader for multiple
    synchronized camera views, then sample clips from the videos. Videos from different
    views are resized and horizontally stacked.
    
    For training and validation, a single clip is randomly sampled from every video
    with random cropping, scaling, and flipping. For testing, multiple clips are 
    uniformaly sampled from every video with uniform cropping.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Multi-view DAAD video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_vid1_straight path_to_vid1_left path_to_vid1_right path_to_vid1_rear path_to_vid1_ego path_to_vid1_gaze label_1
        path_to_vid2_straight path_to_vid2_left path_to_vid2_right path_to_vid2_rear path_to_vid2_ego path_to_vid2_gaze label_2
        ...
        path_to_vidN_straight path_to_vidN_left path_to_vidN_right path_to_vidN_rear path_to_vidN_ego path_to_vidN_gaze label_N
        ```
        
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for DAAD".format(mode)
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
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        # Number of camera views per video
        #self.num_views = cfg.DATA.NUM_VIEWS if hasattr(cfg.DATA, "NUM_VIEWS") else 4 # The remaining 2 are for ego and gaze which will not undergo the same transforms
        self.num_views = cfg.DATA.NUM_VIEWS if hasattr(cfg.DATA, "NUM_VIEWS") else 6
        
        # Target size for each view before stacking
        self.view_height = cfg.DATA.VIEW_HEIGHT if hasattr(cfg.DATA, "VIEW_HEIGHT") else 224
        self.view_width = cfg.DATA.VIEW_WIDTH if hasattr(cfg.DATA, "VIEW_WIDTH") else 224

        logger.info("Constructing DAAD {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the multi-view video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()
            for clip_idx, path_label in enumerate(rows):
                fetch_info = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                
                # Parse multiple view paths and label
                # Last element is the label, rest are paths to different views
                if len(fetch_info) < 2:
                    logger.warning(f"Invalid format in line {clip_idx}: {path_label}")
                    continue
                
                # Extract label (last element) and video paths (all other elements)
                label = fetch_info[-1]
                #view_paths = fetch_info[:4]
                #ego_view = fetch_info[4]
                #gaze_view = fetch_info[5]
                view_paths = fetch_info[:-1]

                # Ensure we have the expected number of views
                if len(view_paths) != self.num_views: #2 views for ego and gaze
                    logger.warning(
                        f"Expected {self.num_views} views but got {len(view_paths)} for clip {clip_idx}"
                    )
                    continue
                
                # Store paths with a special format to indicate multi-view
                for idx in range(self._num_clips):
                    multi_view_paths = [
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path) for path in view_paths
                    ]
                    self._path_to_videos.append(multi_view_paths)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                    
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load DAAD split from {}".format(path_to_file)
        
        logger.info(
            "Constructing DAAD dataloader (size: {} skip_rows {}) from {}".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        This version handles multiple camera views and horizontally stacks them.
        
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames sampled from the stacked multi-view video. 
                The dimension is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:
            return self.dummy_output
            
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
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
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
            
        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode in ["train"] else 1
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val"]
            
        # Try to decode and sample clips from all views
        for i_try in range(self._num_retries):
            multi_view_paths = self._path_to_videos[index]
            all_views_frames = []
            all_views_time_idx = []
            
            # Process each view
            failed_views = False
            for view_idx, video_path in enumerate(multi_view_paths):
                video_container = None
                try:
                    video_container = container.get_video_container(
                        video_path,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video view {} from {} with error {}".format(
                            view_idx, video_path, e
                        )
                    )
                    failed_views = True
                    break
                    
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video view {} idx {} from {}; trial {}".format(
                            view_idx, index, video_path, i_try
                        )
                    )
                    failed_views = True
                    break
                
                # Decode current view
                num_frames = [self.cfg.DATA.NUM_FRAMES]
                sampling_rate = utils.get_random_sampling_rate(
                    self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                    self.cfg.DATA.SAMPLING_RATE,
                )
                sampling_rate = [sampling_rate]
                if len(num_frames) < num_decode:
                    num_frames.extend(
                        [num_frames[-1] for i in range(num_decode - len(num_frames))]
                    )
                    sampling_rate.extend(
                        [sampling_rate[-1] for i in range(num_decode - len(sampling_rate))]
                    )
                elif len(num_frames) > num_decode:
                    num_frames = num_frames[:num_decode]
                    sampling_rate = sampling_rate[:num_decode]
                
                target_fps = self.cfg.DATA.TARGET_FPS
                if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                    target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

                # Use the same temporal sampling across all views for synchronization
                frames, time_idx, tdiff = decoder.decode(
                    video_container,
                    sampling_rate,
                    num_frames,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=(
                        self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                    ),
                    target_fps=target_fps,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    max_spatial_scale=(
                        min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0
                    ),
                    time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
                    temporally_rnd_clips=True,
                    min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                    max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
                )
                
                if frames is None:
                    logger.warning(
                        "Failed to decode video view {} idx {} from {}; trial {}".format(
                            view_idx, index, video_path, i_try
                        )
                    )
                    failed_views = True
                    break
                    
                all_views_frames.append(frames)
                all_views_time_idx.append(time_idx)
            
            # If any view failed to decode, try another random video
            if failed_views:
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
                
            # Now process and stack all successfully decoded views
            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]
            
            # Process each temporal segment
            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    
                    # Process each view for the current clip
                    processed_views = []
                    for view_idx in range(len(all_views_frames)):
                        # Clone the current view's frames
                        view_frames = all_views_frames[view_idx][i].clone().float() / 255.0
                        
                        # Apply same transformations as in original code
                        if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                            view_frames = transform.color_jitter_video_ssl(
                                view_frames,
                                bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                                hue=self.cfg.DATA.SSL_COLOR_HUE,
                                p_convert_gray=self.p_convert_gray,
                                moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                                gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                                gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                            )

                        if self.aug and self.cfg.AUG.AA_TYPE:
                            aug_transform = create_random_augment(
                                input_size=(view_frames.size(1), view_frames.size(2)),
                                auto_augment=self.cfg.AUG.AA_TYPE,
                                interpolation=self.cfg.AUG.INTERPOLATION,
                            )
                            # T H W C -> T C H W.
                            view_frames = view_frames.permute(0, 3, 1, 2)
                            list_img = self._frame_to_list_img(view_frames)
                            list_img = aug_transform(list_img)
                            view_frames = self._list_img_to_frames(list_img)
                            view_frames = view_frames.permute(0, 2, 3, 1)

                        # Perform color normalization.
                        view_frames = utils.tensor_normalize(
                            view_frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                        )

                        # T H W C -> C T H W.
                        view_frames = view_frames.permute(3, 0, 1, 2)

                        # Apply spatial sampling (crop, resize, flip)
                        scl, asp = (
                            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                        )
                        relative_scales = (
                            None if (self.mode not in ["train"] or len(scl) == 0) else scl
                        )
                        relative_aspect = (
                            None if (self.mode not in ["train"] or len(asp) == 0) else asp
                        )
                        view_frames = utils.spatial_sampling(
                            view_frames,
                            spatial_idx=spatial_sample_index,
                            min_scale=min_scale[i],
                            max_scale=max_scale[i],
                            crop_size=crop_size[i],
                            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                            aspect_ratio=relative_aspect,
                            scale=relative_scales,
                            motion_shift=(
                                self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                                if self.mode in ["train"]
                                else False
                            ),
                        )
                        
                        # Resize view to target dimensions for stacking
                        # C T H W -> T C H W for resize function
                        view_frames = view_frames.permute(1, 0, 2, 3)
                        
                        # Use interpolate to resize each view to standard dimensions
                        if view_frames.shape[2] != self.view_height or view_frames.shape[3] != self.view_width:
                            logger.info("Interpolating View Frames...")
                            view_frames = torch.nn.functional.interpolate(
                                view_frames, 
                                size=(self.view_height, self.view_width),
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        # T C H W -> C T H W to match expected format
                        view_frames = view_frames.permute(1, 0, 2, 3)
                        
                        processed_views.append(view_frames)
                    
                    # Horizontally stack views
                    # Each view is C T H W, we want to stack along width dimension
                    # First, make sure all views have the same number of frames
                    min_frames = min([v.shape[1] for v in processed_views])
                    processed_views = [v[:, :min_frames] for v in processed_views]
                    
                    # Stack views (assumes all views have same dimensions after processing)
                    # Convert to C T H (W*num_views) tensor
                    # We will have to modify the model to split views while processing.
                    stacked_frames = torch.cat(processed_views, dim=3)  # Concatenate along width dimension
                    
                    # Store the stacked frames
                    f_out[idx] = stacked_frames
                    
                    # Apply random erasing if enabled
                    if self.rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(
                            f_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                    # Pack for slow-fast pathway if needed
                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                    
                    # Generate mask if needed
                    if self.cfg.AUG.GEN_MASK_LOADER:
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]
                        
                    # Use time index from first view (should be synchronized across views)
                    time_idx_out[idx] = all_views_time_idx[0][i, :]
            
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            
            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode
                
            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, {})
            
            return frames, label, index, time_idx, {}
        else:
            logger.warning(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )

    def _gen_mask(self):
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
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

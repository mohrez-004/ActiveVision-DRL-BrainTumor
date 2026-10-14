"""
Core Architecture & Inference Pipeline for Active-Vision Brain Tumor Localization.

This module provides the production-grade implementation of:
1. The 12-Channel Multi-Scale Foveated Gymnasium Environment specification.
2. The Stage-2 Adaptive ROI-Guided Morphological Chan-Vese Segmentation pipeline.

Note: Full PPO training loops and proprietary reward-shaping coefficients are
restricted pending peer-review publication.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from skimage.segmentation import morphological_chan_vese


class BrainTumorActiveVisionEnv(gym.Env):
    """Custom Gymnasium Environment for 12-Channel Multi-Scale Foveated Localization.

    Decouples lesion search from dense segmentation by allowing an autonomous agent
    to navigate 240x240 multi-modal MRI slices (FLAIR, T1ce, T2) using a hierarchical
    action space before triggering localized segmentation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        volume_shape: Tuple[int, int, int] = (3, 240, 240),
        patch_size: int = 32,
        mid_context_size: int = 96,
        max_steps: int = 60,
        memory_decay: float = 0.98,
    ) -> None:
        super().__init__()
        self.c, self.h, self.w = volume_shape
        self.patch_size = patch_size
        self.mid_context_size = mid_context_size
        self.max_steps = max_steps
        self.memory_decay = memory_decay

        # 17 Discrete Actions:
        # 0..7: Macro steps (10 px stride in 8 cardinal/intercardinal directions)
        # 8..15: Micro steps (2 px stride for fine boundary stabilization)
        # 16: Terminal fixation trigger (initiates Stage-2 ROI segmentation)
        self.action_space = spaces.Discrete(17)

        # 12-Channel Observation Space (12 x 32 x 32), normalized to [0, 1]
        # Channels 0-2: Local high-acuity patch (32x32)
        # Channels 3-5: Intermediate context (96x96 -> downsampled to 32x32)
        # Channels 6-8: Global context (240x240 -> downsampled to 32x32)
        # Channel 9: Decaying visitation memory map
        # Channels 10-11: Normalized spatial coordinate grids (X, Y)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        self._deltas: Dict[int, Tuple[int, int]] = self._build_action_deltas()
        self.image: Optional[np.ndarray] = None
        self.gt_mask: Optional[np.ndarray] = None
        self.memory_map: Optional[np.ndarray] = None
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.current_step: int = 0

    @staticmethod
    def _build_action_deltas() -> Dict[int, Tuple[int, int]]:
        """Constructs coordinate displacement vectors for Macro (10px) and Micro (2px) actions."""
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        deltas = {}
        for idx, (dy, dx) in enumerate(directions):
            deltas[idx] = (dy * 10, dx * 10)       # Actions 0-7: Macro (10 px)
            deltas[idx + 8] = (dy * 2, dx * 2)     # Actions 8-15: Micro (2 px)
        deltas[16] = (0, 0)                        # Action 16: Terminal Fixation
        return deltas

    def _extract_crop(self, img: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
        """Extracts a zero-padded square crop centered at (cy, cx) across all channels."""
        half = size // 2
        padded = np.pad(img, ((0, 0), (half, half), (half, half)), mode="constant")
        py, px = cy + half, cx + half
        return padded[:, py - half : py + half, px - half : px + half]

    def _get_foveated_observation(self) -> np.ndarray:
        """Constructs the 12-channel multi-scale state tensor at current fixation coordinates."""
        assert self.image is not None and self.memory_map is not None
        cy, cx = self.agent_pos
        p = self.patch_size

        # 1. Local High-Acuity Patch (3 x 32 x 32)
        local_patch = self._extract_crop(self.image, cy, cx, p)

        # 2. Intermediate Anatomical Context (96x96 -> 32x32 via Area Interpolation)
        mid_crop = self._extract_crop(self.image, cy, cx, self.mid_context_size)
        mid_patch = np.stack([
            cv2.resize(mid_crop[ch], (p, p), interpolation=cv2.INTER_AREA)
            for ch in range(self.c)
        ], axis=0)

        # 3. Global Context (240x240 -> 32x32 via Area Interpolation)
        global_patch = np.stack([
            cv2.resize(self.image[ch], (p, p), interpolation=cv2.INTER_AREA)
            for ch in range(self.c)
        ], axis=0)

        # 4. Local Visitation Memory Map (1 x 32 x 32)
        mem_crop = self._extract_crop(self.memory_map[np.newaxis, ...], cy, cx, p)

        # 5. Normalized Spatial Coordinate Channels (2 x 32 x 32)
        y_norm = np.full((1, p, p), cy / float(self.h), dtype=np.float32)
        x_norm = np.full((1, p, p), cx / float(self.w), dtype=np.float32)

        obs = np.concatenate(
            [local_patch, mid_patch, global_patch, mem_crop, y_norm, x_norm],
            axis=0,
        )
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one active-vision transition step."""
        assert self.memory_map is not None
        self.current_step += 1
        dy, dx = self._deltas[int(action)]

        # Update coordinates within brain bounds
        ny = int(np.clip(self.agent_pos[0] + dy, 0, self.h - 1))
        nx = int(np.clip(self.agent_pos[1] + dx, 0, self.w - 1))
        self.agent_pos = (ny, nx)

        # Update decaying spatial memory trace
        self.memory_map *= self.memory_decay
        self.memory_map[ny, nx] = 1.0

        terminated = bool(action == 16)
        truncated = bool(self.current_step >= self.max_steps and not terminated)

        # NOTE: Proprietary reward-shaping formulation (distance potential + boundary
        # step-size regularization + terminal fixation bonus) is redacted here
        # pending manuscript publication.
        reward = 0.0

        info = {
            "fixation_point": self.agent_pos,
            "steps_taken": self.current_step,
            "terminated_by_trigger": terminated,
        }
        return self._get_foveated_observation(), reward, terminated, truncated, info


def adaptive_roi_chan_vese(
    flair_slice: np.ndarray,
    fixation_point: Tuple[int, int],
    initial_window: int = 64,
    expanded_window: int = 96,
    iterations: int = 35,
    border_margin: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stage-2 Adaptive ROI-Guided Morphological Chan-Vese Segmentation.

    Restricts active contour evolution to a localized window around the agent's
    terminal fixation point. Dynamically expands the ROI from 64x64 to 96x96 if
    the initial flood-fill seed contacts the crop boundary.

    Args:
        flair_slice: 2D normalized FLAIR MRI array of shape (H, W) in [0, 1].
        fixation_point: (y, x) coordinates predicted by the Stage-1 PPO agent.
        initial_window: Initial ROI crop dimension (default: 64).
        expanded_window: Expanded ROI dimension on boundary contact (default: 96).
        iterations: Number of morphological Chan-Vese evolution iterations.
        border_margin: Pixel margin used to detect boundary contact.

    Returns:
        full_pred_mask: Binary segmentation mask of shape (H, W).
        meta: Dictionary containing ROI expansion status and bounding box coordinates.
    """
    h, w = flair_slice.shape
    cy, cx = fixation_point
    expanded = False

    def _run_localized_segmentation(win_size: int) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
        half = win_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)

        roi = flair_slice[y1:y2, x1:x2]
        roi_u8 = (np.clip(roi, 0.0, 1.0) * 255.0).astype(np.uint8)
        smoothed = cv2.GaussianBlur(roi_u8, (5, 5), 0)

        # Seed coordinate relative to the cropped ROI
        sy, sx = cy - y1, cx - x1
        flood_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(
            smoothed.copy(),
            flood_mask,
            seedPoint=(sx, sy),
            newVal=255,
            loDiff=(20,),
            upDiff=(20,),
            flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
        )
        init_level_set = flood_mask[1:-1, 1:-1] > 0

        # Check if lesion seed touches the ROI boundary
        touches_border = bool(
            np.any(init_level_set[:border_margin, :])
            or np.any(init_level_set[-border_margin:, :])
            or np.any(init_level_set[:, :border_margin])
            or np.any(init_level_set[:, -border_margin:])
        )

        refined_roi = morphological_chan_vese(
            roi,
            num_iter=iterations,
            init_level_set=init_level_set,
            smoothing=1,
        )
        return refined_roi.astype(np.uint8), (y1, y2, x1, x2), touches_border

    # Step 1: Execute on initial 64x64 ROI
    roi_mask, (y1, y2, x1, x2), touches_border = _run_localized_segmentation(initial_window)

    # Step 2: Dynamically expand to 96x96 if boundary contact is detected
    if touches_border:
        expanded = True
        roi_mask, (y1, y2, x1, x2), _ = _run_localized_segmentation(expanded_window)

    full_pred_mask = np.zeros((h, w), dtype=np.uint8)
    full_pred_mask[y1:y2, x1:x2] = roi_mask

    meta = {
        "expanded_to_96": expanded,
        "final_window_size": expanded_window if expanded else initial_window,
        "roi_bbox": (y1, y2, x1, x2),
    }
    return full_pred_mask, meta

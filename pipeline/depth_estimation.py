"""Standalone depth estimation and camera extraction using Depth Anything 3.

Run with --input and --output for the CLI.
"""

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Union

import cv2
import numpy as np
import torch

# Add the project root so sibling packages are importable.
_code_dir = os.path.join(os.path.dirname(__file__), "..")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from pipeline.depth_utils import (  # noqa: E402 - direct-file CLI adds the repository root
    align_ground_plane,
    save_depth_rgba_float,
    smooth_gaussian,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default inference configuration
# ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = {
    "model_path": os.path.join(_PROJECT_ROOT, "checkpoints", "depth"),
    "fix_resize": True,
    "fix_resize_height": 480,
    "fix_resize_width": 832,
    "num_frames": 1000,
    "process_res": 504,
    "debug_exports": False,
}


# ──────────────────────────────────────────────
# Video / image loaders
# ──────────────────────────────────────────────

def load_video(video_path: str, max_frames: int = 81):
    frames_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    return frames_list


def load_images(image_paths: List[str], max_frames: int = None):
    frames_list = []
    if max_frames is None:
        max_frames = len(image_paths)
    else:
        max_frames = min(max_frames, len(image_paths))
    for i in range(max_frames):
        img_bgr = cv2.imread(image_paths[i])
        if img_bgr is None:
            raise IOError(f"Failed to load image: {image_paths[i]}")
        frames_list.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return frames_list


class DepthEstimator:
    """Depth estimation + pose extraction using Depth-Anything-3."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        logger.info("DepthEstimator init, model_path=%s", self.config["model_path"])

        from depth_anything_3.api import DepthAnything3

        # DA3 can pass BF16 tensors to torch.dot during metric-depth alignment.
        # cuBLAS on some GPUs (including L20Z) does not support that dot-product
        # path, so perform this small reduction in FP32.
        import depth_anything_3.model.da3 as da3_model

        def least_squares_scale_fp32(a, b, eps=1e-12):
            a_flat = a.reshape(-1).float()
            b_flat = b.reshape(-1).float()
            numerator = (a_flat * b_flat).sum()
            denominator = (b_flat * b_flat).sum().clamp_min(eps)
            return numerator / denominator

        da3_model.least_squares_scale_scalar = least_squares_scale_fp32

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnything3.from_pretrained(self.config["model_path"])
        self.model = model.to(device=DEVICE)
        self.device = DEVICE
        logger.info("DepthEstimator model loaded on %s", DEVICE)

    # ── helpers ──

    @staticmethod
    def save_frames_to_images(frames, w, h, output_dir):
        for i, frame in enumerate(frames):
            if torch.is_tensor(frame):
                frame_np = frame.detach().cpu().numpy()
                frame_np = np.transpose(frame_np, (1, 2, 0))
                frame_np = (frame_np * 255).astype(np.uint8)
                frame = frame_np
            elif isinstance(frame, np.ndarray):
                frame = frame.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported frame type: {type(frame)}")

            frame_path = f"{output_dir}/{i:04d}.jpg"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (w, h):
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if not cv2.imwrite(frame_path, frame_bgr):
                raise RuntimeError(f"Failed to save frame {i} to {frame_path}")

    @staticmethod
    def load_seg_masks(mask_path: str, max_frames: int = 41):
        from PIL import Image as PILImage
        sources = []
        cap = cv2.VideoCapture(mask_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {mask_path}")
        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sources.append(PILImage.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
        return np.stack(sources)[:, :, :, 0] / 255.0

    @staticmethod
    def depthmap_to_local_points(depth_map: np.ndarray, intrinsics: np.ndarray):
        H, W = depth_map.shape
        if intrinsics.shape == (3, 3):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        else:
            fx, fy, cx, cy = intrinsics[:4]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth_map.reshape(-1)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1)

    # ── main pipeline ──

    def run(self, input_files: Dict[str, Union[str, List[str]]], output_dir: str) -> bool:
        st = time.time()

        image_files = input_files.get("images")
        video_files = input_files.get("videos")
        frames_list = None

        fix_resize_params = None
        if self.config["fix_resize"]:
            fix_resize_params = (self.config["fix_resize_width"], self.config["fix_resize_height"])

        if fix_resize_params:
            target_height, target_width = fix_resize_params[1], fix_resize_params[0]

        max_frames = self.config["num_frames"]

        if image_files:
            if not isinstance(image_files, list) or len(image_files) == 0:
                raise ValueError("images must be a non-empty list")
            image_files = sorted(image_files)
            frames_list = load_images(image_files, max_frames=max_frames)
            target_height = frames_list[0].shape[0]
            target_width = frames_list[0].shape[1]
            logger.info("Loaded %d frames from %d image files", len(frames_list), len(image_files))
        elif video_files:
            video_path = video_files[0]
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            frames_list = load_video(video_path, max_frames=max_frames)
            logger.info("Loaded %d frames from video", len(frames_list))
        else:
            raise ValueError("input_files must contain 'images' or 'videos' key")

        # Ground mask
        ground_seg_idx = -100
        ground_mask = None
        try:
            ground_mask_file = glob.glob(os.path.join(output_dir, "ground_mask_*.png"))[0]
            ground_mask = cv2.imread(ground_mask_file, cv2.IMREAD_GRAYSCALE) / 255.0
            ground_mask[ground_mask < 0.5] = 0.0
            ground_mask[ground_mask >= 0.5] = 1.0
            ground_seg_idx = int(os.path.basename(ground_mask_file).split("_")[-1].split(".")[0])
        except Exception:
            pass

        if frames_list is None:
            raise ValueError("Could not parse input files")

        logger.info("Loaded %d images total", len(frames_list))

        # Create output directories
        frame_dir = os.path.join(output_dir, "frames")
        depth_dir = os.path.join(output_dir, "depth")
        depth_half_dir = os.path.join(output_dir, "depth_half")
        os.makedirs(depth_dir, exist_ok=True)
        if self.config["debug_exports"]:
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(depth_half_dir, exist_ok=True)
            self.save_frames_to_images(frames_list, target_width, target_height, frame_dir)

        # Seg masks
        mask_file = os.path.join(output_dir, "mask.mp4")
        seg_masks = None
        if os.path.exists(mask_file):
            seg_masks = self.load_seg_masks(mask_file, max_frames=max_frames)
            seg_masks[seg_masks < 0.5] = 0.0
            seg_masks[seg_masks >= 0.5] = 1.0

        masked_frames = []
        for idx, frame in enumerate(frames_list):
            if seg_masks is not None:
                frame[seg_masks[idx] == 0] = np.array([0, 0, 0])
            masked_frames.append(frame)

        # Inference
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        s_infer = time.time()
        logger.info("Running depth estimation...")
        res = self.model.inference(
            masked_frames,
            use_ray_pose=False,
            process_res=self.config["process_res"],
        )
        infer_h, infer_w = res.processed_images.shape[1:3]
        logger.info("Inference done in %.1fs", time.time() - s_infer)
        peak = torch.cuda.max_memory_allocated()
        logger.info("Peak GPU memory: %.2f GB", peak / 1024**3)

        ratio_w = target_width * 1.0 / infer_w
        ratio_h = target_height * 1.0 / infer_h
        camera_poses = res.extrinsics
        depth_maps = res.depth
        frames = res.processed_images
        intrinsics = res.intrinsics

        # Convert to 4x4 c2w poses
        if torch.is_tensor(camera_poses):
            camera_poses = camera_poses.cpu().numpy()
        N = camera_poses.shape[0]
        camera_poses_4x4 = np.zeros((N, 4, 4), dtype=camera_poses.dtype)
        camera_poses_4x4[:, :3, :4] = camera_poses
        camera_poses_4x4[:, 3, 3] = 1.0
        camera_poses = np.linalg.inv(camera_poses_4x4)  # w2c -> c2w

        # Normalize to first frame
        T0_inv = np.linalg.inv(camera_poses[0])
        camera_poses = T0_inv @ camera_poses

        depth_scale_factor = 1.0
        camera_poses[:, :3, 3] *= depth_scale_factor

        # Ground plane alignment
        ground_rt_matrix = np.eye(4)

        if ground_mask is not None and 0 <= ground_seg_idx < len(frames):
            idx = ground_seg_idx
            local_pts = self.depthmap_to_local_points(depth_maps[idx], intrinsics[idx]) * depth_scale_factor
            local_pts_homo = np.concatenate([local_pts, np.ones((len(local_pts), 1))], axis=1)
            global_pts = (camera_poses[idx] @ local_pts_homo.T).T[:, :3]
            global_pts = global_pts.reshape(*depth_maps[idx].shape, 3)
            ground_rt_matrix, *_ = align_ground_plane(
                global_pts, ground_mask, ransac_iterations=2000, ransac_threshold=0.05, min_inliers_ratio=0.2)

        # Export only the full-resolution depth consumed by the scene converter.
        for idx, values in enumerate(depth_maps):
            depth_map = cv2.resize(values * depth_scale_factor, (target_width, target_height),
                                   interpolation=cv2.INTER_NEAREST)
            save_depth_rgba_float(f"{depth_dir}/{idx:04d}.png", depth_map)
            if self.config["debug_exports"]:
                depth_half = cv2.resize(depth_map, (target_width // 2, target_height // 2),
                                        interpolation=cv2.INTER_NEAREST)
                save_depth_rgba_float(f"{depth_half_dir}/{idx:04d}.png", depth_half)

        # Save intrinsics
        intrinsic_path = os.path.join(output_dir, "intrinsic.txt")
        with open(intrinsic_path, "w") as fp:
            for mat in intrinsics:
                mat[0, :] *= ratio_w
                mat[1, :] *= ratio_h
                np.savetxt(fp, mat)

        # Save extrinsics (smoothed)
        extrinsic_path = os.path.join(output_dir, "extrinsic.txt")
        try:
            smooth_poses = smooth_gaussian(camera_poses, sigma=2.0)
        except Exception:
            smooth_poses = camera_poses
        with open(extrinsic_path, "w") as fp:
            for mat in smooth_poses:
                mat_rot = ground_rt_matrix @ mat
                mat_rot_w2c = np.linalg.inv(mat_rot)
                np.savetxt(fp, mat_rot_w2c[:3, :])

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Finished in %.1fs", time.time() - st)
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_video_files(input_path: str, video_extensions: List[str]) -> List[str]:
    """Find video files from a path (file or directory)."""
    video_files = []
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in video_extensions:
            video_files.append(input_path)
        else:
            print(f"Warning: {input_path} is not a supported video format {video_extensions}")
    elif os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                video_files.append(os.path.join(input_path, filename))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    return video_files


def process_video(model, video_path, output_dir, video_idx, total_videos, flat_output=False):
    """Process a single video."""
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]

    print(f"\n{'=' * 60}")
    print(f"Processing [{video_idx}/{total_videos}]: {video_name}")
    print(f"{'=' * 60}")

    video_output_dir = output_dir if flat_output else os.path.join(output_dir, video_stem)
    input_files = {"videos": [video_path], "images": []}

    try:
        success = model.run(input_files, video_output_dir)
        if success:
            print(f"Done: {video_name} -> {video_output_dir}")
        else:
            print(f"Failed: {video_name}")
        return success
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Depth estimation CLI")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input video file or directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--video_ext", type=str, default=".mp4,.avi,.mov,.mkv",
                        help="Supported video extensions, comma-separated")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Filter videos by filename keyword")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos with existing output")
    parser.add_argument("--config-json", type=str, default=None,
                        help="JSON string with depth model config overrides")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override the depth model path")
    parser.add_argument("--debug_exports", action="store_true", help="Also export RGB JPEGs and half-resolution depth")
    args = parser.parse_args()

    # Parse config
    config = {}
    if args.config_json:
        config = json.loads(args.config_json)
    if args.model_path:
        config["model_path"] = args.model_path

    if args.debug_exports:
        config["debug_exports"] = True

    # Parse extensions
    video_extensions = [ext.strip().lower() for ext in args.video_ext.split(",")]

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    video_files = get_video_files(args.input, video_extensions)
    if not video_files:
        print("Error: no video files found")
        sys.exit(1)

    if args.filter:
        video_files = [v for v in video_files if args.filter in os.path.basename(v)]

    if args.skip_existing:
        is_single = os.path.isfile(args.input)
        filtered = []
        for vp in video_files:
            check_dir = args.output if is_single else os.path.join(args.output, os.path.splitext(os.path.basename(vp))[0])
            if os.path.exists(os.path.join(check_dir, "intrinsic.txt")):
                print(f"Skipping (already processed): {os.path.basename(vp)}")
            else:
                filtered.append(vp)
        video_files = filtered

    if not video_files:
        print("All videos already processed")
        sys.exit(0)

    print(f"Found {len(video_files)} video(s) to process")
    os.makedirs(args.output, exist_ok=True)

    # Init model
    print("\nInitializing DepthEstimator...")
    model = DepthEstimator(config=config)

    is_single_file = os.path.isfile(args.input)
    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        success = process_video(model, video_path, args.output, idx, len(video_files),
                                flat_output=is_single_file)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {success_count} success, {fail_count} failed, {len(video_files)} total")
    print(f"{'=' * 60}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

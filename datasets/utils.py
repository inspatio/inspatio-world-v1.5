"""Bounded video decoding for inference."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def iter_video_chunks(path, valid_frames, padded_frames, video_size=(480, 832), repeat_first=False):
    """Yield uint8 TCHW: one initial frame, then four at a time.

    Padding repeats the final valid frame; image sources only decode frame zero.
    """
    if valid_frames < 1 or padded_frames < valid_frames or (padded_frames - 1) % 4:
        raise ValueError("Invalid valid/padded frame counts")
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        start = 0
        last = None
        while start < padded_frames:
            stop = min(start + (1 if start == 0 else 4), padded_frames)
            frames = []
            for index in range(start, stop):
                if last is None or (not repeat_first and index < valid_frames):
                    ok, image = reader.read()
                    if not ok:
                        raise ValueError(f"Video is shorter than {valid_frames} frames: {path}")
                    if image.shape[:2] != tuple(video_size):
                        raise ValueError(f"Video resolution must be {tuple(video_size)}: {path}")
                    last = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(last)
            yield torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            start = stop
    finally:
        reader.release()


def normalize_rgb(frames, device, dtype):
    # Match float32 normalization before converting to model dtype.
    return (frames.float() / 255.0 * 2 - 1).to(device=device, dtype=dtype)


def iter_mask_latents(chunks, device, dtype):
    for index, frames in enumerate(chunks):
        mask = (frames[:, :1] > 127.5).float() * 2 - 1
        mask = mask.to(device=device, dtype=dtype)
        mask = F.interpolate(mask, size=(frames.shape[-2] // 8, frames.shape[-1] // 8),
                             mode="bilinear", align_corners=False)
        if index == 0:
            mask = mask.expand(4, -1, -1, -1)
        yield mask[:, 0][None, None]

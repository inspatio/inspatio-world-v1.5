#!/usr/bin/env python3
"""Linear uint16 depth for prepared scenes; one shared min/max per scene.

PNG stores one uint16 channel. MP4 stores its high and low bytes in lossless
RGB channels R and G; B is zero. Both use the same metadata.txt min/max.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


MAX_UINT16 = (1 << 16) - 1
DEPTH_ENCODING = "uint16_minmax_v1"
def image_depth_paths(depth_dir: Path, views: int):
    if views == 1:
        return [depth_dir / "depth.png"]
    return [depth_dir / f"depth_{index:02d}.png" for index in range(views)]


def read_range(metadata_path: Path):
    values = np.loadtxt(metadata_path, ndmin=1)
    if values.size != 2 or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError(f"Invalid depth min/max: {metadata_path}")
    return float(values[0]), float(values[1])


def write_range(metadata_path: Path, minimum: float, maximum: float):
    metadata_path.write_text(f"{minimum:.9g} {maximum:.9g}\n")


def encode_depth(depth, minimum: float, maximum: float):
    values = np.asarray(depth, dtype=np.float32)
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint16)
    else:
        normalized = np.clip((values.astype(np.float64) - minimum) / (maximum - minimum), 0, 1)
        return np.rint(normalized * MAX_UINT16).astype(np.uint16)


def decode_depth(quantized, minimum: float, maximum: float):
    if quantized.ndim != 2 or quantized.dtype != np.uint16:
        raise ValueError("Linear uint16 depth must be one uint16 channel")
    return (minimum + quantized.astype(np.float64) / MAX_UINT16 *
            (maximum - minimum)).astype(np.float32)


def decode_image_depth(path: Path, metadata_path: Path):
    minimum, maximum = read_range(metadata_path)
    with Image.open(path) as image:
        if image.mode not in ("I;16", "I"):
            raise ValueError(f"Expected uint16 depth PNG: {path}")
        return decode_depth(np.asarray(image, dtype=np.uint16), minimum, maximum)


def decode_video_depth_frame(frame_bgr, minimum: float, maximum: float):
    if frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Video depth must be three uint8 channels")
    quantized = ((frame_bgr[..., 2].astype(np.uint16) << 8) |
                 frame_bgr[..., 1].astype(np.uint16))
    return decode_depth(quantized, minimum, maximum)


def encode_video_frame(quantized):
    if quantized.ndim != 2 or quantized.dtype != np.uint16:
        raise ValueError("Expected one uint16 depth channel")
    rgb = np.zeros((*quantized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = quantized >> 8
    rgb[..., 1] = quantized
    return rgb


def range_from_float_depths(paths, height, width):
    minimum, maximum = float("inf"), float("-inf")
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGBA" or image.size != (width, height):
                raise ValueError(f"Expected {width}x{height} RGBA float32 depth: {path}")
            depth = np.frombuffer(np.asarray(image).tobytes(), dtype=np.float32)
        valid = depth[np.isfinite(depth)]
        if valid.size:
            minimum = min(minimum, float(valid.min()))
            maximum = max(maximum, float(valid.max()))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum > maximum:
        raise ValueError("No finite values in estimated video depth")
    return minimum, maximum


def image_source(scene: Path):
    local = scene / "depth" / "source_depths.npy"
    if local.is_file():
        return local
    raise FileNotFoundError(f"Original image depth missing: {local}")


def convert_image(scene: Path, meta: dict, force=False):
    directory = scene / "depth"
    paths = image_depth_paths(directory, meta["views"])
    metadata = directory / "metadata.txt"
    if not force and all(path.is_file() for path in paths) and metadata.is_file():
        with Image.open(paths[0]) as image:
            if image.mode in ("I;16", "I"):
                return
    depth = np.load(image_source(scene), mmap_mode="r")
    if depth.shape != (meta["views"], *meta["resolution"]):
        raise ValueError(f"Unexpected image depth dimensions: {scene}")
    valid = np.isfinite(depth) & (depth >= 0)
    minimum = 0.0
    maximum = float(np.max(depth[valid])) if valid.any() else 1.0
    maximum = max(maximum, 1e-6)
    for index, path in enumerate(paths):
        values = np.where(valid[index], depth[index], 0)
        temporary = path.with_suffix(".tmp.png")
        Image.fromarray(encode_depth(values, minimum, maximum)).save(temporary)
        temporary.replace(path)
    write_range(metadata, minimum, maximum)
    print(f"Converted {scene}: {len(paths)} uint16 depth PNG(s)", flush=True)


def convert_video(scene: Path, meta: dict, source_depth_dir: Path = None,
                  metadata_path: Path = None, force=False):
    directory = scene / "depth"
    destination = directory / "depth.mp4"
    if destination.is_file() and (directory / "metadata.txt").is_file() and not force:
        return
    if source_depth_dir is None:
        raise ValueError("Converting missing video depth requires --source_depth_dir")
    directory.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_depth_dir.glob("*.png"))
    if len(sources) != meta["views"]:
        raise ValueError(f"Unexpected video depth count: {scene}")
    height, width = meta["resolution"]
    minimum, maximum = (read_range(metadata_path) if metadata_path else
                        range_from_float_depths(sources, height, width))
    temporary = directory / "depth.tmp.mp4"
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(meta["fps"]),
               "-i", "-", "-an", "-c:v", "libx264rgb", "-crf", "0", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for source in sources:
            with Image.open(source) as image:
                if image.mode == "RGBA":
                    depth = np.frombuffer(np.asarray(image).tobytes(), dtype=np.float32).reshape(height, width)
                elif image.mode in ("I;16", "I"):
                    depth = decode_depth(np.asarray(image, dtype=np.uint16), minimum, maximum)
                else:
                    raise ValueError(f"Unexpected original video depth format: {source}")
            if depth.shape != (height, width):
                raise ValueError(f"Unexpected video depth dimensions: {source}")
            rgb = encode_video_frame(encode_depth(depth, minimum, maximum))
            process.stdin.write(rgb.tobytes())
    except BaseException:
        process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    process.stdin.close()
    if process.wait() != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Depth video encoding failed: {scene}")
    temporary.replace(destination)
    write_range(directory / "metadata.txt", minimum, maximum)
    print(f"Converted {scene}: {len(sources)} uint16 depth frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--source_depth_dir", type=Path,
                        help="Original float32 RGBA video depth PNG directory")
    parser.add_argument("--metadata_path", type=Path,
                        help="Optional original depth min/max metadata")
    parser.add_argument("--force", action="store_true", help="Rebuild from the provided source depth")
    args = parser.parse_args()
    scene = args.scene_dir.resolve()
    meta = json.loads((scene / "scene.json").read_text())
    if meta["kind"] == "image":
        convert_image(scene, meta, args.force)
    elif meta["kind"] == "video":
        convert_video(scene, meta, args.source_depth_dir, args.metadata_path, args.force)
    else:
        parser.error(f"Unsupported scene kind: {meta['kind']}")


if __name__ == "__main__":
    main()

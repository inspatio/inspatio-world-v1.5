#!/usr/bin/env python3
"""Render and infer prepared scenes or a video, prompt, and target trajectory."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pipeline.gpu import choose_gpu
from pipeline.inference_config import load_config, inference_settings
from pipeline.scene_schema import source_type_for, latent_frame_groups
from pipeline.scene_cache import (file_digest, video_metadata, check_video, condition_paths,
                                 condition_ready, render_fingerprint, prediction_fingerprint,
                                 prediction_ready, write_json)
from datasets.scene_depth_io import (DEPTH_ENCODING, convert_image, convert_video,
                                     image_depth_paths, read_range)


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "examples" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "output"


def video_record(video, scene_id, scene_path):
    frames, fps, _ = video_metadata(video)
    return {"id": scene_id, "scene_id": scene_id, "path": str(scene_path),
            "kind": "video", "views": frames, "frames": frames,
            "valid_frames": frames, "fps": fps, "resolution": [480, 832],
            "target_intrinsics": "estimated_source_frame_000000_fixed",
            "depth_encoding": DEPTH_ENCODING}


def load_record(scene):
    scene = scene.resolve()
    meta = json.loads((scene / "scene.json").read_text())
    if meta.get("kind") == "video" and (scene / "input" / "video.mp4").is_file():
        meta["video_signature"] = file_digest(scene / "input" / "video.mp4")
    scene_id = scene.name
    output_id = meta.get("output_id", scene_id)
    if not isinstance(output_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_id):
        raise ValueError(f"output_id must be an ASCII directory name: {output_id!r}")
    return {**meta, "id": output_id, "scene_id": scene_id, "path": str(scene)}


def direct_video_record(video, prompt, target_traj, output_id, output_root):
    video = video.resolve(strict=True)
    target_traj = target_traj.resolve(strict=True)
    scene_id = output_id or video.stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scene_id):
        raise ValueError("Video filename is not an ASCII output_id; pass --output_id")
    record = video_record(video, scene_id, output_root / scene_id / ".prepared")
    record.update({"video_input": str(video), "prompt_text": prompt,
                   "target_traj_path": str(target_traj),
                   "video_signature": file_digest(video)})
    return record


def materialize_direct_video(record):
    scene = Path(record["path"])
    identity_path = scene / "source_path.txt"
    if scene.exists() and not identity_path.is_file() and any(scene.iterdir()):
        raise ValueError(f"Prepared directory already exists without ownership marker: {scene}")
    directory = scene / "input"
    directory.mkdir(parents=True, exist_ok=True)
    source_path = Path(record["video_input"])
    if identity_path.is_file() and identity_path.read_text().strip() != str(source_path):
        raise ValueError(f"{record['id']} already refers to another video; choose --output_id")
    signature_path = scene / "source_video_signature.txt"
    source_changed = (not signature_path.is_file() or
                      signature_path.read_text().strip() != record["video_signature"])
    identity_path.write_text(str(source_path) + "\n")
    video_path = directory / "video.mp4"
    _, _, resolution = video_metadata(source_path)
    if resolution == (480, 832):
        if video_path.is_symlink() and video_path.resolve() == source_path:
            pass
        elif source_changed and signature_path.is_file():
            temporary = directory / "video.tmp.mp4"
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(source_path)
            temporary.replace(video_path)
        elif video_path.exists() or video_path.is_symlink():
            raise ValueError(f"Prepared video already exists: {video_path}")
        else:
            video_path.symlink_to(source_path)
    elif source_changed or not video_path.is_file():
        temporary = directory / "video.tmp.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source_path),
             "-vf", "scale=832:480:force_original_aspect_ratio=increase,crop=832:480",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temporary)], check=True)
        temporary.replace(video_path)
    check_video(video_path, record["frames"], record["fps"], record["resolution"])
    signature_path.write_text(record["video_signature"] + "\n")
    record["video_signature"] = file_digest(video_path)
    trajectory = Path(record["target_traj_path"])
    target_path = directory / "target_tcw.txt"
    if not target_path.is_symlink() or target_path.resolve() != trajectory:
        if target_path.is_symlink():
            target_path.unlink()
        elif target_path.exists():
            raise ValueError(f"Prepared trajectory already exists: {target_path}")
        target_path.symlink_to(trajectory)
    (directory / "prompt.txt").write_text(record["prompt_text"].strip() + "\n")


def load_examples():
    records = [load_record(DEFAULT_MANIFEST.parent / name)
               for name in json.loads(DEFAULT_MANIFEST.read_text())]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate output_id in {DEFAULT_MANIFEST}")
    return records


def matrices(path, size):
    values = np.loadtxt(path, ndmin=2).astype(np.float32)
    if values.shape[1] != size * size or not np.isfinite(values).all():
        raise ValueError(f"Invalid {size}x{size} matrix file: {path}")
    return values.reshape(-1, size, size)


def validate_scene(record):
    scene = Path(record["path"])
    height, width = record["resolution"]
    frames, views, valid = (int(record[key]) for key in ("frames", "views", "valid_frames"))
    if (record["kind"] not in ("image", "video") or not 0 < valid <= frames
            or (height, width) != (480, 832) or float(record["fps"]) <= 0):
        raise ValueError(f"Invalid scene metadata: {scene}")
    source_type = source_type_for(record)
    if source_type == "video" and views != frames:
        raise ValueError(f"Video views must equal frames: {scene}")
    if source_type == "multi-image" and len(list(latent_frame_groups(frames))) % 3:
        raise ValueError(f"Multi-image frames must form complete three-latent chunks: {scene}")
    if record.get("depth_encoding") != DEPTH_ENCODING:
        raise ValueError(f"Unsupported depth encoding: {scene}")
    if record["target_intrinsics"] not in ("estimated_source_view_00_fixed", "estimated_source_frame_000000_fixed"):
        raise ValueError(f"Unsupported target intrinsics: {scene}")
    lengths = (len(matrices(scene / "depth" / "source_intrinsics.txt", 3)),
               len(matrices(scene / "depth" / "source_tcw.txt", 4)),
               len(matrices(scene / "input" / "target_tcw.txt", 4)))
    if lengths != (views, views, frames):
        raise ValueError(f"RGB/depth/camera count mismatch: {scene}")
    if not (scene / "input" / "prompt.txt").is_file():
        raise FileNotFoundError(scene / "input" / "prompt.txt")
    read_range(scene / "depth" / "metadata.txt")
    if record["kind"] == "image":
        inputs = sorted((scene / "input").glob("view_*.png"))
        depth = image_depth_paths(scene / "depth", views)
        if len(inputs) != views or not all(path.is_file() for path in depth):
            raise ValueError(f"Image/depth mismatch: {scene}")
        for path in depth:
            with Image.open(path) as image:
                if image.size != (width, height) or image.mode not in ("I;16", "I"):
                    raise ValueError(f"Depth PNG must be uint16 {width}x{height}: {path}")
        for path in inputs:
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f"Input resolution mismatch: {path}")
    else:
        input_video = scene / "input" / "video.mp4"
        depth_video = scene / "depth" / "depth.mp4"
        if not input_video.is_file() or not depth_video.is_file():
            raise ValueError(f"Video/depth mismatch: {scene}")
        check_video(input_video, frames, record["fps"], record["resolution"])
        check_video(depth_video, frames, record["fps"], record["resolution"])


def validate_video_inputs(record):
    scene = Path(record["path"])
    input_video = scene / "input" / "video.mp4"
    prompt_path = scene / "input" / "prompt.txt"
    trajectory = scene / "input" / "target_tcw.txt"
    check_video(input_video, record["frames"], record["fps"], record["resolution"])
    if not prompt_path.is_file() or not prompt_path.read_text().strip():
        raise ValueError(f"Missing video prompt: {prompt_path}")
    if len(matrices(trajectory, 4)) != record["frames"]:
        raise ValueError(f"Target trajectory must have one 4x4 Tcw per video frame: {trajectory}")


def source_camera_ready(record):
    depth_dir = Path(record["path"]) / "depth"
    paths = (depth_dir / "source_intrinsics.txt", depth_dir / "source_tcw.txt")
    if not all(path.is_file() for path in paths):
        return False
    return (len(matrices(paths[0], 3)) == record["views"] and
            len(matrices(paths[1], 4)) == record["views"])


def save_estimated_cameras(raw, scene, frames):
    intrinsics = np.loadtxt(raw / "intrinsic.txt", ndmin=2).astype(np.float32)
    extrinsics = np.loadtxt(raw / "extrinsic.txt", ndmin=2).astype(np.float32)
    if (intrinsics.shape != (frames * 3, 3) or
            extrinsics.shape != (frames * 3, 4) or
            not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all()):
        raise ValueError(f"DA3 camera count/format mismatch: {raw}")
    intrinsics = intrinsics.reshape(frames, 3, 3)
    source_tcw = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    source_tcw[:, :3, :4] = extrinsics.reshape(frames, 3, 4)
    depth_dir = scene / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("source_intrinsics.txt", intrinsics),
                       ("source_tcw.txt", source_tcw)):
        temporary = depth_dir / f"{name}.tmp"
        np.savetxt(temporary, data.reshape(frames, -1), fmt="%.9g")
        temporary.replace(depth_dir / name)


def video_depth_ready(record):
    depth_dir = Path(record["path"]) / "depth"
    return (depth_files_ready(record) and source_camera_ready(record) and
            ("video_input" not in record or (depth_dir / "video_signature.txt").is_file()))


def prepare_depth(record, output_root, depth_mode, depth_model_path, gpu=None):
    scene = Path(record["path"])
    depth_dir = scene / "depth"
    if record["kind"] == "image":
        convert_image(scene, record)
        return
    signature_path = depth_dir / "video_signature.txt"
    if depth_mode != "estimate" and video_depth_ready(record):
        # Associate imported depth with its RGB input for subsequent runs.
        if not signature_path.is_file():
            signature_path.write_text(record["video_signature"] + "\n")
        return
    if depth_mode == "existing":
        raise FileNotFoundError(f"Video depth or estimated source cameras are missing: {depth_dir}")
    if record["frames"] > 1000:
        raise ValueError("DA3 fallback supports at most 1000 frames; provide depth/depth.mp4")
    input_video = scene / "input" / "video.mp4"
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    validate_video_inputs(record)
    destination = output_root / record["id"]
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".depth_estimation_", dir=destination) as temporary:
        raw = Path(temporary)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run([sys.executable, str(ROOT / "pipeline" / "depth_estimation.py"),
                        "--input", str(input_video), "--output", str(raw),
                        "--model_path", str(depth_model_path)], check=True, env=env)
        staged = raw / "prepared"
        convert_video(staged, record, source_depth_dir=raw / "depth", force=True)
        save_estimated_cameras(raw, staged, record["frames"])
        depth_dir.mkdir(parents=True, exist_ok=True)
        incomplete = depth_dir / ".estimate_in_progress"
        incomplete.touch()
        for path in (staged / "depth").iterdir():
            path.replace(depth_dir / path.name)
        if record.get("video_signature"):
            signature_path.write_text(record["video_signature"] + "\n")
        incomplete.unlink()


def depth_files_ready(record):
    depth_dir = Path(record["path"]) / "depth"
    files = (image_depth_paths(depth_dir, record["views"]) if record["kind"] == "image"
             else [depth_dir / "depth.mp4"])
    signature_path = depth_dir / "video_signature.txt"
    return (not (depth_dir / ".estimate_in_progress").exists() and
            (depth_dir / "metadata.txt").is_file() and
            all(path.is_file() for path in files) and
            (not signature_path.is_file() or
             signature_path.read_text().strip() == record.get("video_signature")))


def inference_record(record, output_root, settings=None):
    paths = condition_paths(record, output_root)
    source_type = source_type_for(record)
    result = {"id": record["id"], "source_video": str(paths["source"]),
            "render_video": str(paths["render"]), "mask_video": str(paths["mask"]),
            "output_path": str(output_root / record["id"] / "pred.mp4"),
            "source_type": source_type,
            "valid_frames": record["valid_frames"], "output_fps": record["fps"],
            "text": (Path(record["path"]) / "input" / "prompt.txt").read_text().strip(),
            "scene_path": record["path"], "target_intrinsics": record["target_intrinsics"],
            "resolution": record["resolution"], "render_fingerprint": render_fingerprint(record)}
    if settings is not None:
        result["prediction_fingerprint"] = prediction_fingerprint(result, render_fingerprint(record), settings)
    return result


def ensure_t5_weights(folder):
    folder = Path(folder)
    safetensors = folder / "models_t5_umt5-xxl-enc-bf16.safetensors"
    if safetensors.is_file():
        return
    source = folder / "models_t5_umt5-xxl-enc-bf16.pth"
    if not source.is_file():
        raise FileNotFoundError(f"Missing Wan T5 weights: {source}")
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "convert_pth_to_safetensors.py"),
                    "--input", str(source), "--output", str(safetensors)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scene_dir", type=Path, help="Prepared scene directory with scene.json")
    source.add_argument("--video", type=Path, help="Input video; requires --prompt and --target_traj")
    parser.add_argument("--prompt", help="Prompt text for --video")
    parser.add_argument("--target_traj", type=Path, help="Per-frame 4x4 target Tcw text for --video")
    parser.add_argument("--output_id", help="Result directory name for --video (defaults to video stem)")
    parser.add_argument("--depth_mode", choices=("auto", "existing", "estimate"), default="auto")
    parser.add_argument("--depth_model_path", type=Path, default=ROOT / "checkpoints" / "depth")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--config_path", type=Path, default=ROOT / "configs" / "inference_1.3b.yaml")
    parser.add_argument("--gpu", default="auto", help="Visible GPU index, UUID, or auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile_dit", action="store_true")
    args = parser.parse_args()
    output_root = DEFAULT_OUTPUT.resolve()
    if args.video:
        if not args.prompt or not args.prompt.strip() or not args.target_traj:
            parser.error("--video requires --prompt and --target_traj")
        records = [direct_video_record(args.video, args.prompt, args.target_traj,
                                       args.output_id, output_root)]
    else:
        if args.prompt is not None or args.target_traj is not None or args.output_id is not None:
            parser.error("--prompt, --target_traj, and --output_id require --video")
        records = [load_record(args.scene_dir)] if args.scene_dir else load_examples()
    catalog = output_root / "manifest.json"
    catalog_records = {item["id"]: item for item in json.loads(catalog.read_text())} if catalog.is_file() else {}
    for record in records:
        old = catalog_records.get(record["id"])
        if old and not (Path(old["scene_path"]).is_dir() and
                        Path(old["scene_path"]).samefile(record["path"])):
            parser.error(f"output_id {record['id']!r} already belongs to another scene; choose a new output_id")
    config = load_config(args.config_path)
    gpu = None

    def selected_gpu():
        nonlocal gpu
        if gpu is None:
            gpu = choose_gpu(args.gpu)
        return gpu

    checkpoint = (args.checkpoint_path or ROOT / "checkpoints/InSpatio-World-1.3B/InSpatio-World-1.5-1.3B.safetensors").resolve()
    ensure_t5_weights(config.wan_model_folder)
    settings = inference_settings(config, checkpoint, args.seed, compile_dit=args.compile_dit)

    for record in records:
        if args.video:
            materialize_direct_video(record)
        if record["kind"] == "video":
            validate_video_inputs(record)
            if args.depth_mode != "existing" and (args.depth_mode == "estimate" or not video_depth_ready(record)):
                selected_gpu()
        prepare_depth(record, output_root, args.depth_mode, args.depth_model_path, gpu)
        validate_scene(record)
        print(f"Validated {record['scene_id']}", flush=True)
    missing = [record for record in records if not condition_ready(record, output_root)]
    if missing:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = selected_gpu()
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output_root) as stream:
            json.dump(missing, stream)
            stream.flush()
            subprocess.run([sys.executable, "-m", "pipeline.render_scene", "--manifest", stream.name,
                            "--output_root", str(output_root)], cwd=ROOT, env=env, check=True)
        for record in missing:
            if not condition_ready(record, output_root):
                raise RuntimeError(f"Condition render incomplete: {record['id']}")
    ready_records = [inference_record(record, output_root, settings) for record in records]
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_records = {item["id"]: item for item in json.loads(catalog.read_text())} if catalog.is_file() else {}
    catalog_records.update({item["id"]: item for item in ready_records})
    write_json(catalog, list(catalog_records.values()))
    print(f"Catalog: {len(catalog_records)} scenes -> {catalog}", flush=True)
    pending = []
    for item in ready_records:
        if prediction_ready(item):
            print(f"Already inferred: {item['id']}", flush=True)
        else:
            pending.append(item)
    if not pending:
        print("All scenes already have valid results.", flush=True)
        return
    inference_python = Path(os.environ.get("INSPATIO_INFERENCE_PYTHON", sys.executable)).expanduser().resolve()
    if not inference_python.is_file():
        raise FileNotFoundError(f"Inference Python is missing: {inference_python}")
    selected_gpu()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=".run_example_",
                                     dir=output_root, delete=False) as stream:
        json.dump(pending, stream, ensure_ascii=False, indent=2)
        run_manifest = Path(stream.name)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [str(inference_python), "inference.py", "--config_path", str(args.config_path.resolve()),
               "--checkpoint_path", str(checkpoint), "--scene_manifest", str(run_manifest),
               "--seed", str(args.seed)]
    if args.compile_dit:
        command.append("--compile_dit")
    print(f"Infer {len(pending)} scenes on GPU {gpu}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode:
        raise RuntimeError(f"Inference failed; pending manifest kept at {run_manifest}")
    run_manifest.unlink()
    print("All scenes completed.", flush=True)


if __name__ == "__main__":
    main()

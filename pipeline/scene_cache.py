"""Content fingerprints and completion records for scene outputs."""

import hashlib
import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from datasets.scene_depth_io import image_depth_paths
from pipeline.scene_schema import source_type_for

RENDER_VERSION = 2
INFERENCE_VERSION = 2


def file_digest(path):
    # Hash bytes rather than trusting file timestamps (which may be restored or
    # have coarse resolution on shared filesystems).
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames,r_frame_rate,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    return streams[0]


def video_metadata(path):
    info = video_info(path)
    fps = float(Fraction(info["r_frame_rate"]))
    frames = info.get("nb_frames")
    if not frames or frames == "N/A":
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
            capture_output=True, text=True, check=True)
        frames = json.loads(result.stdout)["streams"][0].get("nb_read_frames")
    if not frames or frames == "N/A" or int(frames) < 1 or fps <= 0:
        raise ValueError(f"Cannot determine frames or fps: {path}")
    return int(frames), fps, (int(info["height"]), int(info["width"]))


def check_video(path, frames, fps, resolution):
    actual_frames, actual_fps, actual_size = video_metadata(path)
    if actual_frames != frames or actual_size != tuple(resolution) or abs(actual_fps - fps) > .01:
        raise ValueError(f"Video frames/fps/resolution mismatch: {path}")


def render_fingerprint(record):
    scene = Path(record["path"])
    files = [scene / "input" / "target_tcw.txt", scene / "depth" / "source_tcw.txt",
             scene / "depth" / "source_intrinsics.txt", scene / "depth" / "metadata.txt"]
    if record["kind"] == "video":
        files.extend([scene / "input" / "video.mp4", scene / "depth" / "depth.mp4"])
    else:
        files.extend(sorted((scene / "input").glob("view_*.png")))
        files.extend(image_depth_paths(scene / "depth", record["views"]))
    return fingerprint({
        "version": RENDER_VERSION,
        "scene": {key: record[key] for key in ("kind", "views", "frames", "fps", "resolution",
                                                "depth_encoding", "target_intrinsics")},
        "files": {str(path.relative_to(scene)): file_digest(path) for path in files},
    })


def condition_paths(record, output_root):
    return {name: Path(output_root) / record["id"] / f"{name}.mp4"
            for name in ("source", "render", "mask")}


def condition_ready(record, output_root):
    paths = condition_paths(record, output_root)
    try:
        done = json.loads((paths["render"].parent / "render_complete.json").read_text())
        if done.get("fingerprint") != render_fingerprint(record):
            return False
        for name, path in paths.items():
            if done.get("outputs", {}).get(name) != file_digest(path):
                return False
            check_video(path, record["frames"], record["fps"], record["resolution"])
        return True
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return False


def complete_render(record, output_root, extra=None):
    paths = condition_paths(record, output_root)
    done = {"fingerprint": render_fingerprint(record), "frames": record["frames"],
            "fps": record["fps"], "source_type": source_type_for(record),
            "outputs": {name: file_digest(path) for name, path in paths.items()}}
    done.update(extra or {})
    write_json(paths["render"].parent / "render_complete.json", done)


def prediction_fingerprint(record, render_id, settings):
    return fingerprint({"version": INFERENCE_VERSION, "render": render_id, "settings": settings,
                        "conditions": {name: file_digest(record[f"{name}_video"]) for name in ("source", "render", "mask")},
                        "prompt": record["text"], "frames": record["valid_frames"],
                        "fps": record["output_fps"]})


def prediction_ready(record):
    path = Path(record["output_path"])
    try:
        done = json.loads(path.with_name("prediction_complete.json").read_text())
        if not record.get("prediction_fingerprint") or done.get("fingerprint") != record["prediction_fingerprint"]:
            return False
        if done.get("output") != file_digest(path):
            return False
        check_video(path, record["valid_frames"], record["output_fps"], record.get("resolution", (480, 832)))
        return True
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return False


def complete_prediction(record):
    path = Path(record["output_path"])
    check_video(path, record["valid_frames"], record["output_fps"], record.get("resolution", (480, 832)))
    write_json(path.with_name("prediction_complete.json"),
               {"fingerprint": record["prediction_fingerprint"], "output": file_digest(path)})

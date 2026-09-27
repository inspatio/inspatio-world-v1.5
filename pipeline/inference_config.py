"""Load the supported inference configuration and identify model inputs."""

from pathlib import Path

from omegaconf import OmegaConf

from pipeline.scene_cache import file_digest, fingerprint
from pipeline.scene_schema import validate_config

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = OmegaConf.merge(OmegaConf.load(ROOT / "configs/default_config.yaml"), OmegaConf.load(path))
    validate_config(config)
    config.wan_model_folder = str((ROOT / config.wan_model_folder).resolve())
    config.generator.model_path = str((ROOT / config.generator.model_path).resolve())
    return config


def inference_settings(config, checkpoint, seed=0, *, compile_dit=False):
    files = {Path(checkpoint).resolve()}
    folder = Path(config.wan_model_folder)
    t5 = folder / "models_t5_umt5-xxl-enc-bf16.safetensors"
    files.add(t5 if t5.is_file() else folder / "models_t5_umt5-xxl-enc-bf16.pth")
    files.update(path for path in (folder / "google" / "umt5-xxl").rglob("*") if path.is_file())
    files.add(folder / "Wan2.1_VAE.pth")
    files.add(Path(config.generator.model_path) / "config.json")
    return fingerprint({"config": OmegaConf.to_container(config, resolve=True),
                        "weights": {str(path): file_digest(path) for path in sorted(files)},
                        "seed": seed, "compile_dit": compile_dit})

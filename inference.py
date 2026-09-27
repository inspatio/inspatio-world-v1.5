"""Single-GPU scene inference with bounded video encoding and decoding."""

import argparse
import gc
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file

from datasets.scene_dataset import SceneDataset
from datasets.utils import iter_mask_latents, normalize_rgb
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import denoise_block
from pipeline.inference_config import load_config, inference_settings
from pipeline.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from pipeline.scene_cache import (complete_prediction, prediction_ready, prediction_fingerprint,
                                 file_digest, fingerprint)
from pipeline.scene_schema import (FRAMES_PER_BLOCK, LATENT_CHANNELS, LATENT_HEIGHT,
                                   LATENT_WIDTH)
from pipeline.video_writer import video_writer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vae_scale(vae, device, dtype):
    return [vae.mean.to(device=device, dtype=dtype), 1 / vae.std.to(device=device, dtype=dtype)]


def encode_stream(chunks, pipeline, device):
    model_chunks = (normalize_rgb(frames, device, torch.bfloat16).permute(1, 0, 2, 3)[None]
                    for frames in chunks)
    scale = vae_scale(pipeline.vae, device, torch.bfloat16)
    latents = list(pipeline.vae.model.encode_stream(model_chunks, scale))
    return torch.cat(latents, dim=2).permute(0, 2, 1, 3, 4)


def decode_stream(latents, pipeline, device):
    scale = vae_scale(pipeline.vae, device, torch.bfloat16)
    for frames in pipeline.vae.model.decode_stream(latents.permute(0, 2, 1, 3, 4), scale):
        yield (frames.float().clamp(-1, 1) * .5 + .5).permute(0, 2, 1, 3, 4)


def write_prediction(record, chunks):
    height, width = record.get("resolution", (480, 832))
    written = 0
    try:
        with video_writer(record["output_path"], record["output_fps"], width, height) as writer:
            for frames in chunks:
                count = min(frames.shape[1], record["valid_frames"] - written)
                if count:
                    pixels = (frames[0, :count].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    writer.write(pixels.tobytes())
                    written += count
                if written == record["valid_frames"]:
                    break
            if written != record["valid_frames"]:
                raise RuntimeError(f"Generated {written} frames, expected {record['valid_frames']}")
    finally:
        chunks.close()
    complete_prediction(record)


def warmup(pipeline, device, compile_dit):
    if not compile_dit:
        return
    pipeline.generator.model = torch.compile(pipeline.generator.model, mode="max-autotune",
                                            fullgraph=False, dynamic=False, backend="inductor")
    noise = torch.randn(1, FRAMES_PER_BLOCK, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH,
                        device=device, dtype=torch.bfloat16)
    render = torch.randn(1, FRAMES_PER_BLOCK, 20, LATENT_HEIGHT, LATENT_WIDTH,
                         device=device, dtype=torch.bfloat16)
    condition = {"prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)}
    for context_size in (FRAMES_PER_BLOCK, 2 * FRAMES_PER_BLOCK):
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        context = torch.randn(1, context_size, 36, LATENT_HEIGHT, LATENT_WIDTH,
                              device=device, dtype=torch.bfloat16)
        denoise_block(pipeline.generator, pipeline.scheduler, noise, condition, pipeline.kv_cache1,
                      context_frames=context, render_block=render,
                      denoising_steps=pipeline.denoising_step_list)
    torch.cuda.synchronize(device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--scene_manifest", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile_dit", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Scene inference uses one GPU per process")
    config = load_config(args.config_path)
    if not torch.cuda.is_available():
        parser.error("Scene inference requires a CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    settings = inference_settings(config, args.checkpoint_path, args.seed, compile_dit=args.compile_dit)
    dataset = SceneDataset(args.scene_manifest, config.dataset.video_size)
    pending = []
    for record in dataset:
        # The runner identifies the scene inputs. Direct invocation identifies
        # the exact condition videos as well, so an old output cannot be reused.
        render_id = record.get("render_fingerprint") or fingerprint(
            {name: file_digest(record[f"{name}_video"]) for name in ("source", "render", "mask")})
        record["prediction_fingerprint"] = prediction_fingerprint(record, render_id, settings)
        if not prediction_ready(record):
            pending.append(record)
    if not pending:
        print("All predictions are current.")
        return
    pipeline = CausalInferencePipeline(config)
    state_dict = load_file(args.checkpoint_path)
    pipeline.generator.load_state_dict(state_dict, strict=True)
    del state_dict
    pipeline = pipeline.to(dtype=torch.bfloat16).eval()
    if get_cuda_free_memory_gb(device) < 40:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    warmup(pipeline, device, args.compile_dit)
    gc.collect()
    torch.cuda.empty_cache()
    for record in pending:
        # Each scene has a stable seed, independent of batching and cache hits.
        set_seed(args.seed)
        start = time.monotonic()
        render = encode_stream(dataset.frames(record, "render"), pipeline, device)
        source = encode_stream(dataset.frames(record, "source"), pipeline, device)
        mask = torch.cat(list(iter_mask_latents(dataset.frames(record, "mask"), device, torch.bfloat16)), dim=1)
        if render.shape != source.shape or mask.shape[1] != source.shape[1]:
            raise ValueError(f"Condition latent lengths differ: {record['id']}")
        noise = torch.randn_like(source)
        result = pipeline.inference(noise=noise, text_prompts=[record["text"]],
                                    ref_latent=source, render_latent=render, mask_latent=mask)
        del noise, source, render, mask
        write_prediction(record, decode_stream(result, pipeline, device))
        del result
        print(f"Completed {record['id']} in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

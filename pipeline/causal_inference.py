from typing import List
import torch
import torch.nn.functional as F
import time
from einops import rearrange
from pipeline.scene_schema import (validate_config, FRAMES_PER_BLOCK, TOKENS_PER_FRAME,
                                   LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)
from pipeline.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

def convert_mask_video(mask_video):
    batch = mask_video.shape[0]
    height, width = mask_video.shape[-2:]
    latent_height, latent_width = height // 8, width // 8
    mask_video = rearrange(mask_video[:, :1], 'b c t h w -> (b t) c h w')
    mask_video = F.interpolate(
        mask_video, size=(latent_height, latent_width), mode='bilinear', align_corners=False
    )
    mask_video = rearrange(mask_video, '(b t) c h w -> b t c h w', b=batch)
    mask_video = torch.cat([mask_video[:, :1].repeat_interleave(4, dim=1), mask_video[:, 1:]], dim=1)
    usable_frames = mask_video.shape[1] - mask_video.shape[1] % 4
    mask_video = mask_video[:, :usable_frames]
    return mask_video.reshape(batch, usable_frames // 4, 4, latent_height, latent_width)


@torch.no_grad()
def denoise_block(
    generator,
    scheduler,
    noisy_input,
    conditional_dict,
    kv_cache,
    *,
    context_frames,
    render_block,
    denoising_steps,
):
    """Cache the context, then return the block's final denoised latents."""
    B, F = noisy_input.shape[:2]
    device = noisy_input.device

    generator(
        noisy_image_or_video=context_frames,
        conditional_dict=conditional_dict,
        timestep=torch.zeros([B, F], device=device, dtype=torch.int64),
        kv_cache=kv_cache,
        render_latent_input=render_block,
        kv_size=(0, -1),
        freqs_offset=0,
    )
    denoising_kv_size = context_frames.shape[1] * TOKENS_PER_FRAME

    for index, current_timestep in enumerate(denoising_steps):
        is_last_step = (index == len(denoising_steps) - 1)
        timestep = torch.ones([B, F], device=device, dtype=torch.int64) * current_timestep

        _, denoised_pred = generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            kv_size=(0, denoising_kv_size),
            render_latent_input=render_block,
            freqs_offset=2 * FRAMES_PER_BLOCK,
        )

        if not is_last_step:
            next_t = denoising_steps[index + 1]
            noisy_input = scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_t * torch.ones([B * F], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

    return denoised_pred


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        validate_config(args)
        # Step 1: Initialize all models
        time_start = time.time() 
        self.generator = WanDiffusionWrapper(**args.generator)
        print(f"Time taken to initialize generator: {time.time() - time_start} seconds")

        time_start = time.time()
        wan_model_folder = getattr(args, "wan_model_folder", None)
        self.text_encoder = WanTextEncoder(model_folder=wan_model_folder) if text_encoder is None else text_encoder
        print(f"Time taken to initialize text encoder: {time.time() - time_start} seconds")

        time_start = time.time()
        self.vae = WanVAEWrapper(model_folder=wan_model_folder) if vae is None else vae
        print(f"Time taken to initialize vae: {time.time() - time_start} seconds")

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.scheduler
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            print("warping denoising step list")
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.kv_cache1 = None
        self.num_frame_per_block = getattr(args, "num_frame_per_block", FRAMES_PER_BLOCK)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        
    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        ref_latent: torch.Tensor,
        render_latent: torch.Tensor,
        mask_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return denoised latents in [batch, frames, channels, height, width].
        The inference entry point streams these through the selected decoder.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if (num_channels, height, width) != (LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH):
            raise ValueError("Unsupported latent shape for the released checkpoint")
        assert num_frames % self.num_frame_per_block == 0, f"num_frames {num_frames} is not a multiple of num_frame_per_block {self.num_frame_per_block}"
        num_blocks = num_frames // self.num_frame_per_block

        num_output_frames = num_frames   # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
 
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )


        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["k"].detach_().zero_()
                self.kv_cache1[block_index]["v"].detach_().zero_()
 
        # Step 3: Temporal denoising loop
        print(f"Generating {num_blocks} blocks...")
        all_num_frames = [self.num_frame_per_block] * num_blocks

        start_index = 0
        last_pred = None
        for num_block_frame in all_num_frames:
            noisy_input = noise[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            ref_block = ref_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = render_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            mask_block = mask_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = torch.cat([mask_block, render_block], dim=2)

            # Prepare context
            context_frames = None
            zero_latents = torch.zeros_like(ref_block)
            ref_block = torch.cat([ref_block, zero_latents[:, :, :4], zero_latents], dim=2)
            if start_index == 0:
                context_frames = ref_block
            else:
                # Prepare context similar to training pipeline
                zero_latents = torch.zeros_like(last_pred)
                last_pred_padded = torch.cat([last_pred, zero_latents[:, :, :4], zero_latents], dim=2)
                context_frames = torch.cat([ref_block, last_pred_padded], dim=1)

            denoised_pred = denoise_block(
                self.generator, self.scheduler, noisy_input, conditional_dict,
                self.kv_cache1,
                context_frames=context_frames,
                render_block=render_block,
                denoising_steps=self.denoising_step_list,
            )


            # Step 3.2: record the model's output
            output[:, start_index:start_index + num_block_frame] = denoised_pred
            last_pred = denoised_pred.detach()

            # Step 3.4: update the start and end frame indices
            start_index += num_block_frame
 

        return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize or reuse KV cache for the Wan model.
        Uses detach() + zero_() to safely reuse cache without gradient issues.
        Cache is allocated only once; subsequent calls only zero the existing tensors.
        """
        if self.kv_cache1 is not None and len(self.kv_cache1) == self.num_transformer_blocks \
                and self.kv_cache1[0]["k"].shape[0] == batch_size \
                and self.kv_cache1[0]["k"].dtype == dtype \
                and self.kv_cache1[0]["k"].device == device:
            for block_cache in self.kv_cache1:
                block_cache["k"].detach_().zero_()
                block_cache["v"].detach_().zero_()
            return

        kv_cache_size = TOKENS_PER_FRAME * self.num_frame_per_block * 2
        num_heads = self.generator.model.config.num_heads
        dim = self.generator.model.config.dim

        print(f"Initializing kv cache with size: {kv_cache_size}")
        self.kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            self.kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
            })

from typing import List, Tuple
import torch

from pipeline.scheduler import FlowMatchScheduler
from pipeline.scene_schema import TOKENS_PER_FRAME
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
import os

from safetensors.torch import load_file as safe_load_file

class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_folder: str) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('meta')
        ).eval().requires_grad_(False)
        self.text_encoder.to_empty(device='cpu')

        safetensors_path = os.path.join(model_folder, "models_t5_umt5-xxl-enc-bf16.safetensors")
        self.text_encoder.load_state_dict(safe_load_file(safetensors_path))

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_folder, "google", "umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        result = { "prompt_embeds": context }
        return result


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_folder: str):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        vae_path = os.path.join(model_folder, "Wan2.1_VAE.pth")
        self.model = _video_vae(
            pretrained_path=vae_path,
            z_dim=16,
        ).eval().requires_grad_(False)


class WanDiffusionWrapper(torch.nn.Module):
    """Build the causal DiT from config; inference loads its full checkpoint once."""

    def __init__(self, model_path, timestep_shift=5.0, in_dim=36):
        super().__init__()
        config = dict(CausalWanModel.load_config(model_path))
        config["in_dim"] = in_dim
        with torch.device("meta"):
            self.model = CausalWanModel(**config)
        self.model.to_empty(device="cpu")
        self.scheduler = FlowMatchScheduler(shift=timestep_shift)
        self.seq_len = TOKENS_PER_FRAME * 24

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # Preserve the original float64 arithmetic, but reuse device schedules.
        # Their tensor versions detect in-place scheduler updates as well.
        original_dtype = flow_pred.dtype
        schedule = (self.scheduler.sigmas, self.scheduler.timesteps)
        key = (flow_pred.device, *(value.data_ptr() for value in schedule),
               *(value._version for value in schedule))
        if getattr(self, "_flow_schedule_key", None) != key:
            self._flow_schedule = tuple(value.to(device=flow_pred.device, dtype=torch.float64)
                                        for value in schedule)
            self._flow_schedule_key = key
        sigmas, timesteps = self._flow_schedule
        flow_pred, xt = flow_pred.double(), xt.double()

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: List[dict],
        kv_size: Tuple[int, int],
        render_latent_input: torch.Tensor,
        freqs_offset: int = 0,
    ):
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous(),
            t=timestep,
            context=conditional_dict["prompt_embeds"],
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            kv_size=kv_size,
            render_latent_input=render_latent_input.permute(0, 2, 1, 3, 4).contiguous(),
            freqs_offset=freqs_offset,
        ).permute(0, 2, 1, 3, 4)
        if kv_size[1] < 0:
            return flow_pred
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

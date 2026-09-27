"""Compare bounded model execution with the original full-video equations."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from wan.modules.vae import WanVAE_
from pipeline.wan_wrapper import WanDiffusionWrapper


class StreamingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generator_builds_from_config_without_base_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"dim": 32, "ffn_dim": 64, "freq_dim": 16, "text_dim": 32,
                      "num_heads": 4, "num_layers": 1, "in_dim": 16}
            (Path(directory) / "config.json").write_text(json.dumps(config))
            wrapper = WanDiffusionWrapper(directory)
            self.assertEqual(wrapper.model.patch_embedding.in_channels, 36)
            self.assertEqual(len(wrapper.model.blocks), 1)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in wrapper.parameters()))

    @torch.no_grad()
    def test_vae_stream_preserves_causal_state(self):
        torch.manual_seed(4)
        model = WanVAE_(dim=8, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                        temperal_downsample=[False, True, True]).eval()
        video = torch.randn(1, 3, 9, 16, 16)
        scale = [torch.tensor([.1, .2]), torch.tensor([1.5, 2.0])]
        expected_latents = model.encode(video, scale)
        actual_latents = torch.cat(list(model.encode_stream([video[:, :, :1], video[:, :, 1:5], video[:, :, 5:]], scale)), dim=2)
        torch.testing.assert_close(actual_latents, expected_latents, rtol=1e-5, atol=1e-5)
        expected_pixels = model.decode(expected_latents, scale)
        actual_pixels = torch.cat(list(model.decode_stream(actual_latents, scale)), dim=2)
        torch.testing.assert_close(actual_pixels, expected_pixels, rtol=1e-5, atol=1e-5)
        self.assertTrue(all(item is None for item in model._feat_map))
        self.assertTrue(all(item is None for item in model._enc_feat_map))
        stream = model.decode_stream(actual_latents, scale)
        next(stream)
        stream.close()
        self.assertTrue(all(item is None for item in model._feat_map))

    def test_flow_conversion_matches_double_reference(self):
        torch.manual_seed(6)
        wrapper = object.__new__(WanDiffusionWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.scheduler = SimpleNamespace(sigmas=torch.linspace(1, 0, 1000), timesteps=torch.linspace(1000, 0, 1000))
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            flow = torch.randn(4, 16, 8, 8).to(dtype)
            noisy = torch.randn_like(flow)
            times = torch.tensor([1000., 750., 500., 250.])
            indices = (wrapper.scheduler.timesteps.double()[None] - times[:, None]).abs().argmin(dim=1)
            sigma = wrapper.scheduler.sigmas.double()[indices, None, None, None]
            reference = (noisy.double() - sigma * flow.double()).to(dtype)
            actual = wrapper._convert_flow_pred_to_x0(flow, noisy, times)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class ScheduleCacheTests(unittest.TestCase):
    def test_updated_sigmas_invalidate_converted_schedule(self):
        wrapper = object.__new__(WanDiffusionWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.scheduler = SimpleNamespace(sigmas=torch.tensor([1., .5]), timesteps=torch.tensor([1000., 500.]))
        flow = torch.ones(1, 1, 1, 1)
        times = torch.tensor([500.])
        before = wrapper._convert_flow_pred_to_x0(flow, flow, times)
        wrapper.scheduler.sigmas[1] = .25
        after = wrapper._convert_flow_pred_to_x0(flow, flow, times)
        torch.testing.assert_close(before, torch.full_like(flow, .5))
        torch.testing.assert_close(after, torch.full_like(flow, .75))


if __name__ == '__main__':
    unittest.main()

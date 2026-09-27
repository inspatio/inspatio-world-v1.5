"""Flow-matching noise schedule used by causal inference."""

import torch


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps=1000, shift=5.0):
        self.num_train_timesteps = num_train_timesteps
        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

    def add_noise(self, original_samples, noise, timestep):
        """Mix clean samples and noise at the nearest scheduled timestep."""
        timestep = timestep.flatten()
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

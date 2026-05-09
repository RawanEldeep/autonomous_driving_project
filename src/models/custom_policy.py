import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


class AttentionRegularizedPPO(PPO):

    def __init__(
        self,
        policy,
        env,
        lambda_high: float = 0.01,
        lambda_low: float = 0.01,
        n_observed_vehicles: int = 10,
        enable_attention_reg: bool = True,
        **kwargs,
    ):
        super().__init__(policy, env, **kwargs)
        self.lambda_high = lambda_high
        self.lambda_low = lambda_low
        self.n_observed_vehicles = n_observed_vehicles
        self.enable_attention_reg = enable_attention_reg

        self.H_target = float(np.log(max(n_observed_vehicles / 2.0, 1.0)))

        self._attn_cache: list = []
        self._hook_handles: list = []

    def _register_hooks(self):
        self._attn_cache.clear()
        self._hook_handles.clear()
        try:
            attn_layer = self.policy.features_extractor.extractor.attention_layer

            def _hook(module, inp, out):
                if isinstance(out, tuple) and len(out) == 2:
                    self._attn_cache.append(out[1])

            handle = attn_layer.register_forward_hook(_hook)
            self._hook_handles.append(handle)
        except AttributeError:
            pass

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _attention_entropy_loss(self) -> torch.Tensor:
        if not self._attn_cache:
            return torch.tensor(0.0, device=self.device)

        try:
            attn = torch.cat(self._attn_cache, dim=0)
            attn = attn.squeeze(2)
            attn = torch.clamp(attn, min=1e-8, max=1.0)

            H = -torch.sum(attn * torch.log(attn), dim=-1).mean()

            H_target = torch.tensor(
                self.H_target, device=self.device, dtype=torch.float32
            )

            loss = (
                self.lambda_high * F.relu(H_target - H)
                + self.lambda_low  * F.relu(H - H_target)
            )
            return loss

        except Exception:
            return torch.tensor(0.0, device=self.device)

    def _attention_reg_pass(self):
        if not self.enable_attention_reg:
            return

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            self._attn_cache.clear()
            self._register_hooks()

            self.policy.optimizer.zero_grad()

            _ = self.policy.features_extractor(rollout_data.observations)

            attn_loss = self._attention_entropy_loss()

            if attn_loss.item() > 0.0:
                attn_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                self.logger.record(
                    "train/attention_entropy_loss", float(attn_loss.item())
                )

            self._remove_hooks()

        self._attn_cache.clear()

    def train(self):
        super().train()
        self._attention_reg_pass()

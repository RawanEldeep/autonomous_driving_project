"""
Enhancement 4: Attention Regularization via Entropy Penalty

Subclasses Stable-Baselines3 PPO and adds an entropy-based regularization
term targeting the EgoAttention module inside the feature extractor.

Loss formula (from proposal):
    H(a)     = -Σ_i  a_i · log(a_i)          (attention distribution entropy)
    H_target = log(K / 2)                     (K = number of observed vehicles)
    L_attn   = λ_high · max(0, H_target - H(a))   <- penalise over-concentration
             + λ_low  · max(0, H(a)  - H_target)  <- penalise over-diffusion
    L_total  = L_PPO + L_attn

Implementation strategy:
    After every standard PPO training epoch, an additional optimisation pass
    is run that:
        1. Registers a forward hook on EgoAttention to capture attention weights.
        2. Runs a features-extractor-only forward pass (no PPO loss needed).
        3. Computes L_attn and back-propagates it through the extractor.
        4. Clips gradients and steps the shared optimizer.

    This keeps the enhancement modular and does NOT alter SB3 internals.
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


class AttentionRegularizedPPO(PPO):
    """
    PPO with EgoAttention entropy regularization.

    All standard PPO arguments are supported; the extra keyword arguments below
    control the regularization behaviour.

    Parameters
    ----------
    lambda_high : float
        Penalty weight when attention entropy < H_target (too concentrated).
    lambda_low : float
        Penalty weight when attention entropy > H_target (too diffuse).
    n_observed_vehicles : int
        K in H_target = log(K/2); should match ``vehicles_count`` in the env.
    enable_attention_reg : bool
        Master toggle — set False to disable without changing other args.
    """

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

        # H_target = log(K/2); clamp to avoid log(0)
        self.H_target = float(np.log(max(n_observed_vehicles / 2.0, 1.0)))

        # Buffers populated by forward hooks
        self._attn_cache: list = []
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Attach a forward hook to the EgoAttention layer."""
        self._attn_cache.clear()
        self._hook_handles.clear()
        try:
            attn_layer = self.policy.features_extractor.extractor.attention_layer

            def _hook(module, inp, out):
                # EgoAttention.forward returns (result, attention_matrix)
                # attention_matrix shape: (batch, heads, 1, n_entities)
                if isinstance(out, tuple) and len(out) == 2:
                    # Keep the computation graph so we can back-prop through it
                    self._attn_cache.append(out[1])

            handle = attn_layer.register_forward_hook(_hook)
            self._hook_handles.append(handle)
        except AttributeError:
            # Feature extractor does not have an attention layer — skip silently
            pass

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Attention entropy loss
    # ------------------------------------------------------------------

    def _attention_entropy_loss(self) -> torch.Tensor:
        """
        Compute L_attn from the attention matrices captured during the last
        forward pass.

        Returns a scalar tensor (with gradient) or zero if no attention was
        captured.
        """
        if not self._attn_cache:
            return torch.tensor(0.0, device=self.device)

        try:
            # Stack all batches captured during the mini-batch forward pass
            attn = torch.cat(self._attn_cache, dim=0)   # (B, heads, 1, entities)
            attn = attn.squeeze(2)                        # (B, heads, entities)
            attn = torch.clamp(attn, min=1e-8, max=1.0)  # numerical stability

            # H(a) = -Σ a_i log(a_i)  per sample per head → scalar mean
            H = -torch.sum(attn * torch.log(attn), dim=-1).mean()

            H_target = torch.tensor(
                self.H_target, device=self.device, dtype=torch.float32
            )

            # Asymmetric penalty: push H toward H_target from both sides
            loss = (
                self.lambda_high * F.relu(H_target - H)   # too concentrated
                + self.lambda_low  * F.relu(H - H_target)  # too diffuse
            )
            return loss

        except Exception:
            return torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    # Attention regularization pass
    # ------------------------------------------------------------------

    def _attention_reg_pass(self):
        """
        Additional optimisation pass run once per PPO training call.

        Iterates over mini-batches from the rollout buffer, computes L_attn,
        and back-props it through the feature extractor weights.
        """
        if not self.enable_attention_reg:
            return

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            self._attn_cache.clear()
            self._register_hooks()

            self.policy.optimizer.zero_grad()

            # Forward through the feature extractor only — no value/policy heads
            # needed.  The hook captures attention weights with gradient tracking.
            _ = self.policy.features_extractor(rollout_data.observations)

            attn_loss = self._attention_entropy_loss()

            if attn_loss.item() > 0.0:
                attn_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                # Log to TensorBoard
                self.logger.record(
                    "train/attention_entropy_loss", float(attn_loss.item())
                )

            self._remove_hooks()

        self._attn_cache.clear()

    # ------------------------------------------------------------------
    # Override PPO.train
    # ------------------------------------------------------------------

    def train(self):
        """Run standard PPO update then apply attention regularization."""
        super().train()
        self._attention_reg_pass()

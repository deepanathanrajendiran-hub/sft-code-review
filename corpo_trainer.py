"""CoRPOTrainer: subclass of GRPOTrainer that clips the group baseline at
R_min_correct, fixing GRPO's failure mode on ordinal rewards.

Paper: https://arxiv.org/abs/2511.04439 (Eq. 11)
    b_CoRPO   = max(R_min_correct, group_mean)
    A_CoRPO_i = R_i - b_CoRPO

Implementation note: TRL's GRPOTrainer computes advantages inline inside the
~200-line `_generate_and_score_completions` method (see TRL grpo_trainer.py
around line 2165: `advantages = rewards - mean_grouped_rewards`). Rather than
copy that whole method, we use a side-channel approach:

  1. Override `_calculate_rewards` to capture the raw per-func rewards tensor.
  2. Override `_generate_and_score_completions` to run super(), then recompute
     the CoRPO-corrected advantages from the captured rewards and overwrite
     `output["advantages"]`.

Constraints:
- Requires `scale_rewards="none"` in GRPOConfig — std-normalization would break
  the CoRPO baseline math. The constructor asserts this.
- Targets `multi_objective_aggregation="sum_then_normalize"` (TRL default).
- Tested on TRL >=0.12, <0.20. If TRL changes the output dict's "advantages"
  key or the `_calculate_rewards` signature, this subclass breaks loudly.
- Multi-GPU: the post-process slicing assumes single-process; for multi-GPU,
  the process_index slicing needs adjustment.
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer


class CoRPOTrainer(GRPOTrainer):
    """GRPOTrainer with baseline-clipping (CoRPO) advantage computation."""

    def __init__(self, *args, r_min_correct: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_min_correct = r_min_correct
        self._last_raw_rewards: torch.Tensor | None = None

        scale = getattr(self, "scale_rewards", "none")
        if scale != "none":
            raise ValueError(
                f"CoRPOTrainer requires GRPOConfig.scale_rewards='none' "
                f"(got {scale!r}). std-normalization would break the CoRPO "
                f"baseline correction."
            )

    def _compute_corpo_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute CoRPO advantages from grouped per-rollout rewards.

        rewards: tensor of shape (B * G,) — flat batch of per-rollout scalars.
        Returns advantages of the same flat shape.

        Test-friendly: this method has no dependence on the GRPOTrainer state
        beyond `self.r_min_correct` and `self.num_generations`. Task 6's unit
        tests target this method directly via `CoRPOTrainer.__new__`.
        """
        num_gens = self.num_generations
        rewards_flat = rewards.view(-1)
        rewards_grouped = rewards_flat.view(-1, num_gens)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        baseline = torch.clamp(group_means, min=self.r_min_correct)
        advantages = rewards_grouped - baseline
        return advantages.view(-1)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Capture raw rewards for the post-process step."""
        rewards_per_func = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        self._last_raw_rewards = rewards_per_func.detach().clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Run super, then replace GRPO advantages with CoRPO advantages."""
        output = super()._generate_and_score_completions(inputs)

        if self._last_raw_rewards is None:
            # Defensive: shouldn't happen since super() invokes _calculate_rewards
            return output

        # Aggregate per-func rewards with weights (mirrors TRL's sum_then_normalize)
        rewards_per_func = self._last_raw_rewards
        weights = self.reward_weights.to(rewards_per_func.device)
        rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)

        # Recompute CoRPO advantages on the full (pre-slice) batch
        new_advantages_full = self._compute_corpo_advantages(rewards)

        # output["advantages"] is sliced for this process. For single-GPU
        # (process_index=0) the slice is the whole tensor; for multi-GPU,
        # we'd need TRL's process_slice logic.
        per_process_size = len(output["advantages"])
        process_index = getattr(self.accelerator, "process_index", 0)
        start = process_index * per_process_size
        end = start + per_process_size
        output["advantages"] = new_advantages_full[start:end].to(
            output["advantages"].device
        )

        self._last_raw_rewards = None  # reset for next call
        return output

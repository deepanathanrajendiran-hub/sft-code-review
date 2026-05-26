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
- Targets TRL 0.22.2 (the version pinned by the canonical Unsloth GRPO Colab
  notebook). Method signatures verified against
  https://raw.githubusercontent.com/huggingface/trl/v0.22.2/trl/trainer/grpo_trainer.py
  If TRL changes the output dict's "advantages" key or the `_calculate_rewards`
  signature, this subclass breaks loudly via the RuntimeError in
  `_generate_and_score_completions`.
- Multi-GPU: the post-process slicing assumes single-process; for multi-GPU,
  the process_index slicing needs adjustment.
"""
from __future__ import annotations

import torch

# Unsloth's import side-effect patches `vllm.sampling_params` to provide a
# `GuidedDecodingParams` shim. vLLM 0.12+ removed it; TRL 0.22.2 still imports
# it at module-load. Without this `import unsloth` happening BEFORE `from trl`,
# we get: ImportError: cannot import name 'GuidedDecodingParams' from 'vllm.sampling_params'
# Wrapped in try/except so local test environments (no unsloth installed) still work.
try:
    import unsloth  # noqa: F401 — load for its import-time monkey-patches
except ImportError:
    pass

from trl import GRPOTrainer


class CoRPOTrainer(GRPOTrainer):
    """GRPOTrainer with baseline-clipping (CoRPO) advantage computation."""

    def __init__(self, *args, r_min_correct: float = 0.0, **kwargs):
        """Initialize CoRPOTrainer.

        Args:
            r_min_correct: Correctness threshold from CoRPO Eq. 11. Rollouts
                with reward below this value cannot receive positive advantage,
                even when the group mean is also below it. Typical range: 0.0
                (binary rewards centered at 0) to 0.5 (composite rewards in
                [0, 1] where 0.5 = "must beat the median of perfect-vs-junk").
            All other args/kwargs are forwarded to GRPOTrainer.__init__.
        """
        super().__init__(*args, **kwargs)
        self.r_min_correct = r_min_correct
        self._last_raw_rewards: torch.Tensor | None = None

        scale = getattr(self.args, "scale_rewards", "none")
        if scale != "none":
            raise ValueError(
                f"CoRPOTrainer requires GRPOConfig.scale_rewards='none' "
                f"(got {scale!r}). std-normalization would break the CoRPO "
                f"baseline correction."
            )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Capture raw rewards for the post-process step.

        Note: TRL gathers across processes before returning, so the captured
        tensor has shape (global_B * G, num_reward_funcs), not local. The
        slicing in _generate_and_score_completions accounts for this.
        """
        rewards_per_func = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        self._last_raw_rewards = rewards_per_func.detach().clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Run super, then replace GRPO advantages with CoRPO advantages."""
        try:
            output = super()._generate_and_score_completions(inputs)
            captured = self._last_raw_rewards
        finally:
            self._last_raw_rewards = None

        if captured is None:
            raise RuntimeError(
                "CoRPOTrainer expected _calculate_rewards to populate "
                "_last_raw_rewards before _generate_and_score_completions, "
                "but it didn't. Has TRL's GRPOTrainer been updated in a way "
                "that skips _calculate_rewards? This subclass would silently "
                "produce GRPO advantages — refusing to continue."
            )

        # Aggregate per-func rewards with weights (mirrors TRL grpo_trainer.py
        # weighted reward aggregation, ~line 1486 in TRL 0.12-0.19).
        rewards_per_func = captured
        weights = self.reward_weights.to(self.accelerator.device)
        rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)

        # Recompute CoRPO advantages on the full (pre-slice) batch
        new_advantages_full = self._compute_corpo_advantages(rewards)

        # output["advantages"] is sliced for this process. For single-GPU
        # (process_index=0) the slice is the whole tensor; for multi-GPU,
        # TRL's process_slice logic uses contiguous shards.
        per_process_size = len(output["advantages"])
        process_index = getattr(self.accelerator, "process_index", 0)
        start = process_index * per_process_size
        end = start + per_process_size
        output["advantages"] = new_advantages_full[start:end].to(
            output["advantages"].device
        )

        return output

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

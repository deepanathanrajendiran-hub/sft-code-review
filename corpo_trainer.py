"""GRPOTrainer subclass implementing CoRPO baseline-clipping.

The fix: clamp the group baseline at r_min_correct so a group of all-bad
rollouts can't manufacture positive advantage. From arXiv:2511.04439 Eq. 11:

    b_CoRPO   = max(R_min_correct, group_mean)
    A_CoRPO_i = R_i - b_CoRPO

TRL bakes the advantage computation deep inside the ~200-line
`_generate_and_score_completions` (`advantages = rewards - mean_grouped_rewards`).
Rather than fork that whole method we go through a side channel: capture the raw
rewards in `_calculate_rewards`, let super() run, then recompute and overwrite
`output["advantages"]`.

Assumptions, all enforced or it breaks loudly:
- scale_rewards="none" in GRPOConfig. std-normalization would wreck the baseline
  math; the constructor asserts it.
- multi_objective_aggregation="sum_then_normalize" (TRL default).
- Pinned to TRL 0.22.2 (the version in the Unsloth GRPO Colab). If TRL moves the
  "advantages" key or changes the _calculate_rewards signature, the RuntimeError
  in _generate_and_score_completions fires.
- The post-process slice assumes single process. Multi-GPU needs the
  process_index slicing revisited.
"""
from __future__ import annotations

import torch

# importing unsloth patches vllm.sampling_params to re-add the GuidedDecodingParams
# shim that vLLM 0.12+ dropped but TRL 0.22.2 still imports at module load. This
# has to happen before `from trl`, or the trl import dies with ImportError. guarded
# so local test envs without unsloth still import this module.
try:
    import unsloth  # noqa: F401 — load for its import-time monkey-patches
except ImportError:
    pass

from trl import GRPOTrainer


class CoRPOTrainer(GRPOTrainer):
    """GRPOTrainer with CoRPO (baseline-clipping) advantages."""

    def __init__(self, *args, r_min_correct: float = 0.0, **kwargs):
        """r_min_correct is the CoRPO correctness threshold (Eq. 11): rollouts
        scoring below it can't get positive advantage even when the whole group
        is below it. ~0.0 for binary rewards centered at 0, up to ~0.5 for
        composite rewards in [0, 1]. Everything else forwards to GRPOTrainer.
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
        """Stash the raw rewards for the post-process step. TRL gathers across
        processes first, so the captured tensor is global (global_B * G, n_funcs),
        not local — the slice in _generate_and_score_completions accounts for it.
        """
        rewards_per_func = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        self._last_raw_rewards = rewards_per_func.detach().clone()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """Run super, then swap GRPO advantages for CoRPO advantages."""
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

        # weighted aggregation across reward funcs, matching TRL's own
        rewards_per_func = captured
        weights = self.reward_weights.to(self.accelerator.device)
        rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)

        new_advantages_full = self._compute_corpo_advantages(rewards)

        # output["advantages"] is this process's slice. single-GPU gets the whole
        # tensor; multi-GPU shards are contiguous, hence the process_index math.
        per_process_size = len(output["advantages"])
        process_index = getattr(self.accelerator, "process_index", 0)
        start = process_index * per_process_size
        end = start + per_process_size
        output["advantages"] = new_advantages_full[start:end].to(
            output["advantages"].device
        )

        return output

    def _compute_corpo_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Group-mean baseline clamped at r_min_correct, applied to flat (B*G,)
        rewards. Kept free of trainer state beyond r_min_correct and
        num_generations so the unit tests can hit it via __new__.
        """
        num_gens = self.num_generations
        rewards_flat = rewards.view(-1)
        rewards_grouped = rewards_flat.view(-1, num_gens)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        baseline = torch.clamp(group_means, min=self.r_min_correct)
        advantages = rewards_grouped - baseline
        return advantages.view(-1)

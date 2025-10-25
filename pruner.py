from typing import Dict, Optional
import torch

# pruner.py
# Minimal, extensible PyTorch model pruner boilerplate.
# Usage: import Pruner, attach to a model, call step() to prune by global magnitude.

import torch.nn as nn


class Pruner:
    """
    Custom magnitude-based model pruner.
    - Selects parameters (by default: parameters with name ending in '.weight').
    - Computes global threshold, creates binary masks, applies masks in-place.
    - Registers gradient hooks so pruned weights remain zero during training.
    Extend by overriding score_parameters / create_masks for other strategies.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        # maps parameter name -> mask tensor (same shape as parameter)
        self.masks: Dict[str, torch.Tensor] = {}
        # maps parameter -> hook handle (to avoid duplicate hooks)
        self._hooks = {}

    def _iter_prunable(self):
        # yield (module, param_name, parameter) for candidate parameters
        for module_name, module in self.model.named_modules():
            for name, param in list(module.named_parameters(recurse=False)):
                full_name = f"{module_name}.{name}" if module_name else name
                # default: prune weights only
                if name.endswith("weight") and param is not None:
                    yield full_name, param

    def score_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Compute scores used to rank parameters. Default: absolute value (magnitude).
        Returns dict mapping parameter full-name -> score tensor (same shape as param).
        """
        scores = {}
        for full_name, param in self._iter_prunable():
            scores[full_name] = param.data.abs().clone().to(self.device)
        return scores

    def create_global_masks(self, sparsity: float) -> Dict[str, torch.Tensor]:
        """
        Create binary masks using a global magnitude threshold so that `sparsity`
        fraction of the total elements are zeroed.
        sparsity: 0.0 (no pruning) .. 1.0 (all pruned)
        """
        assert 0.0 <= sparsity <= 1.0
        if sparsity == 0.0:
            return {n: torch.ones_like(p, dtype=torch.bool, device=self.device) for n, p in self._iter_prunable()}

        scores = self.score_parameters()
        # collect all scores into a single vector
        all_scores = torch.cat([s.view(-1) for s in scores.values()])
        k = int(sparsity * all_scores.numel())
        if k <= 0:
            thresh = torch.tensor(float("-inf"), device=self.device)
        elif k >= all_scores.numel():
            thresh = torch.tensor(float("inf"), device=self.device)
        else:
            # find the k-th smallest magnitude -> threshold
            # kthvalue expects 1-based index
            thresh = torch.kthvalue(all_scores, k).values

        masks = {}
        for name, score in scores.items():
            masks[name] = (score > thresh).to(dtype=torch.bool)
        return masks

    def apply_masks(self, masks: Dict[str, torch.Tensor], remove_hooks: bool = False):
        """
        Apply masks in-place to model parameters and register gradient hooks
        so pruned weights remain zero during training.
        If remove_hooks is True, existing hooks are removed before registering new ones.
        """
        if remove_hooks:
            for handle in self._hooks.values():
                try:
                    handle.remove()
                except Exception:
                    pass
            self._hooks.clear()

        # apply masks and register gradient hooks
        name_to_param = {n: p for n, p in self._iter_prunable()}
        for name, mask in masks.items():
            if name not in name_to_param:
                continue
            param = name_to_param[name]
            mask = mask.to(param.device).to(dtype=param.data.dtype)
            # in-place zeroing
            param.data.mul_(mask)
            # store boolean mask for bookkeeping (on CPU)
            self.masks[name] = mask.to(dtype=torch.bool, device="cpu")

            # register hook to zero out gradients at masked positions
            if param not in self._hooks:
                def make_hook(m):
                    def hook(grad):
                        return grad * m.to(grad.device)
                    return hook
                handle = param.register_hook(make_hook(mask))
                self._hooks[param] = handle

    def prune_step(self, sparsity: float):
        """
        Convenience: create global masks at given sparsity and apply them.
        """
        masks = self.create_global_masks(sparsity)
        self.apply_masks(masks)

    def remove_pruning(self):
        """
        Remove masks and hooks, leaving parameters as-is (no re-dense).
        """
        for handle in self._hooks.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks.clear()
        self.masks.clear()

    def save_masks(self, filepath: str):
        torch.save(self.masks, filepath)

    def load_masks(self, filepath: str, apply: bool = True):
        masks = torch.load(filepath)
        self.masks = {k: v.to(dtype=torch.bool, device="cpu") for k, v in masks.items()}
        if apply:
            # map masks to device and apply
            dev_masks = {k: v.to(self.device) for k, v in self.masks.items()}
            self.apply_masks(dev_masks, remove_hooks=True)


if __name__ == "__main__":
    # Simple example
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleModel()
    pruner = Pruner(model)

    # before pruning
    total = 0
    nonzero = 0
    for n, p in model.named_parameters():
        if n.endswith("weight"):
            total += p.numel()
            nonzero += (p.data != 0).sum().item()
    print(f"Before prune: nonzero={nonzero}/{total}")

    # prune 50% globally by magnitude
    pruner.prune_step(0.5)

    total = 0
    nonzero = 0
    for n, p in model.named_parameters():
        if n.endswith("weight"):
            total += p.numel()
            nonzero += (p.data != 0).sum().item()
    print(f"After prune: nonzero={nonzero}/{total}")

    # save masks
    pruner.save_masks("masks.pt")
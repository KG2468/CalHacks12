from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

# pruner.py
# Extensible PyTorch model pruner with "Consensus Pruning".
# - Generates masks from multiple criteria (magnitude, gradient, etc.)
# - Prunes weights that are flagged by a "majority vote" (k-out-of-N)

class Pruner:
    """
    Custom model pruner with consensus-based pruning.
    
    - Selects parameters (by default: parameters with name ending in '.weight').
    - Can generate masks from multiple scoring criteria (magnitude, gradient, etc.).
    - Implements "Consensus Pruning" (prune_consensus):
      Prunes weights only if at least 'k' of 'N' methods agree they are unimportant.
    - Registers gradient hooks so pruned weights remain zero during training.
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

    # --- 1. SCORING METHODS ---

    def score_parameters_magnitude(self) -> Dict[str, torch.Tensor]:
        """
        METHOD A: Compute scores based on absolute value (magnitude).
        Returns dict mapping parameter full-name -> score tensor.
        """
        scores = {}
        for full_name, param in self._iter_prunable():
            scores[full_name] = param.data.abs().clone().to(self.device)
        return scores
        
    def score_parameters_gradient(self, data_batch: Tuple, loss_fn: _Loss) -> Dict[str, torch.Tensor]:
        """
        METHOD B: Compute scores based on gradient importance (|weight * gradient|).
        Requires a sample data batch and a loss function.
        """
        # Ensure model is in eval mode if it has dropout/batchnorm,
        # but keep grads enabled
        was_training = self.model.training
        self.model.eval() 
        
        self.model.zero_grad()
        
        # Move data to the correct device
        inputs, labels = [d.to(self.device) for d in data_batch]
        
        # Forward and backward pass
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        scores = {}
        for full_name, param in self._iter_prunable():
            if param.grad is not None:
                # Score = |weight * gradient|
                score = (param.data * param.grad).abs().clone().to(self.device)
                scores[full_name] = score
            else:
                # No gradient (e.g., frozen layer), score is zero
                scores[full_name] = torch.zeros_like(param.data, device=self.device)

        self.model.zero_grad()
        self.model.train(was_training) # Restore original training state
        return scores

    def score_parameters_random(self) -> Dict[str, torch.Tensor]:
        """
        METHOD C: Compute random scores. Useful as a baseline.
        """
        scores = {}
        for full_name, param in self._iter_prunable():
            scores[full_name] = torch.rand_like(param.data, device=self.device)
        return scores

    # --- 2. MASK GENERATION ---

    def create_keep_mask_global(self, scores: Dict[str, torch.Tensor], sparsity: float) -> Dict[str, torch.Tensor]:
        """
        Create binary *keep* masks (True=Keep, False=Prune) using a global threshold
        so that `sparsity` fraction of the total elements are zeroed (pruned).
        
        sparsity: 0.0 (no pruning) .. 1.0 (all pruned)
        """
        assert 0.0 <= sparsity <= 1.0
        if sparsity == 0.0:
            return {n: torch.ones_like(p, dtype=torch.bool, device=self.device) for n, p in self._iter_prunable()}

        # collect all scores into a single vector
        try:
            all_scores = torch.cat([s.view(-1) for s in scores.values()])
        except (RuntimeError, TypeError):
             # Handle case where no prunable parameters were scored
             return {}
             
        if all_scores.numel() == 0:
            return {}

        k = int(sparsity * all_scores.numel())
        if k <= 0:
            # Sparsity is 0 or too small, keep everything
            thresh = torch.tensor(float("-inf"), device=self.device)
        elif k >= all_scores.numel():
            # Sparsity is 1.0, prune everything
            thresh = torch.tensor(float("inf"), device=self.device)
        else:
            # find the k-th smallest magnitude -> threshold
            # We want to PRUNE k elements, so we KEEP (numel - k) elements.
            # The threshold should be the k-th smallest value.
            thresh = torch.kthvalue(all_scores, k).values

        masks = {}
        for name, score in scores.items():
            # Keep if score > threshold
            masks[name] = (score > thresh).to(dtype=torch.bool)
        return masks

    # --- 3. PRUNING AND MASK APPLICATION ---

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

    def prune_magnitude_global(self, sparsity: float):
        """
        Convenience: create global magnitude masks and apply them.
        """
        scores = self.score_parameters_magnitude()
        masks = self.create_keep_mask_global(scores, sparsity)
        self.apply_masks(masks, remove_hooks=True)

    def prune_consensus(
        self,
        methods: List[str],
        sparsity_per_method: float,
        consensus_k: int,
        data_batch: Optional[Tuple] = None,
        loss_fn: Optional[_Loss] = None,
    ):
        """
        **CORE HACKATHON LOGIC**
        Performs "Consensus Pruning" (or "Majority Vote" pruning).
        
        - methods: List of scoring methods to use (e.g., ['magnitude', 'gradient'])
        - sparsity_per_method: The pruning % for each *individual* method (e.g., 0.5)
        - consensus_k: Prune a weight if at least 'k' methods vote to prune it.
                       (e.g., if methods=['mag', 'grad', 'rand'] and k=2,
                       a weight is pruned if 2 OR 3 of the methods flag it).
        - data_batch, loss_fn: Required if 'gradient' method is in the list.
        """
        print(f"--- Starting Consensus Pruning ---")
        print(f"Methods: {methods}, Sparsity per method: {sparsity_per_method}, Consensus k: {consensus_k}")
        
        if "gradient" in methods and (data_batch is None or loss_fn is None):
            raise ValueError("`data_batch` and `loss_fn` must be provided for 'gradient' scoring.")

        # --- 1. Get all individual keep masks ---
        all_keep_masks: Dict[str, List[torch.Tensor]] = {}
        
        for method_name in methods:
            scores = {}
            if method_name == 'magnitude':
                scores = self.score_parameters_magnitude()
            elif method_name == 'gradient':
                scores = self.score_parameters_gradient(data_batch, loss_fn)
            elif method_name == 'random':
                scores = self.score_parameters_random()
            else:
                print(f"Warning: Unknown scoring method '{method_name}'. Skipping.")
                continue
                
            print(f"Generating mask for method: '{method_name}'")
            keep_mask = self.create_keep_mask_global(scores, sparsity_per_method)
            
            for name, mask in keep_mask.items():
                if name not in all_keep_masks:
                    all_keep_masks[name] = []
                all_keep_masks[name].append(mask)

        # --- 2. Tally votes and create final mask ---
        final_keep_mask = {}
        for name, mask_list in all_keep_masks.items():
            if not mask_list:
                continue
            
            # Stack all masks for this param: [N_methods, *param_shape]
            mask_stack = torch.stack(mask_list)
            
            # We have "keep" masks (True=Keep). We want to "vote to prune".
            # Vote to prune = ~mask (True=Prune)
            # Sum the votes to prune (sum over N_methods dimension)
            prune_votes = torch.sum((~mask_stack).int(), dim=0)
            
            # Keep the weight if (votes_to_prune < k)
            final_keep_mask[name] = (prune_votes < consensus_k).to(dtype=torch.bool)
        
        print("Consensus masks computed. Applying to model...")
        self.apply_masks(final_keep_mask, remove_hooks=True)
        print("--- Consensus Pruning Finished ---")


    # --- 4. UTILITY METHODS ---

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

# --- EXAMPLE USAGE ---

def count_nonzero(model):
    total = 0
    nonzero = 0
    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if name.endswith("weight"):
                total += param.numel()
                nonzero += (param.data != 0).sum().item()
    return nonzero, total

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    
    # --- Example 1: Simple Magnitude Pruning ---
    print("\n--- EXAMPLE 1: Standard Magnitude Pruning ---")
    model_mag = SimpleModel()
    pruner_mag = Pruner(model_mag)

    nonzero, total = count_nonzero(model_mag)
    print(f"Before prune: nonzero={nonzero}/{total} ({nonzero/total*100:.2f}%)")

    # prune 50% globally by magnitude
    pruner_mag.prune_magnitude_global(0.5)

    nonzero_after, total = count_nonzero(model_mag)
    print(f"After prune (50%): nonzero={nonzero_after}/{total} ({nonzero_after/total*100:.2f}%)")
    
    
    # --- Example 2: Consensus Pruning ---
    print("\n--- EXAMPLE 2: Consensus Pruning ---")
    model_con = SimpleModel()
    pruner_con = Pruner(model_con)
    
    # Create dummy data for gradient scoring
    dummy_inputs = torch.randn(16, 100)
    dummy_labels = torch.randint(0, 10, (16,))
    loss_fn = nn.CrossEntropyLoss()
    
    nonzero, total = count_nonzero(model_con)
    print(f"Before prune: nonzero={nonzero}/{total} ({nonzero/total*100:.2f}%)")

    # Prune 50% by 3 methods.
    # Prune a weight if at least 2 of the 3 methods vote to prune it.
    # This finds the "majority vote" unimportant weights.
    pruner_con.prune_consensus(
        methods=['magnitude', 'gradient', 'random'],
        sparsity_per_method=0.5, # Each method identifies its bottom 50%
        consensus_k=2,             # Prune if 2 or 3 methods agree
        data_batch=(dummy_inputs, dummy_labels),
        loss_fn=loss_fn
    )

    nonzero_after, total = count_nonzero(model_con)
    print(f"After consensus prune (k=2/3): nonzero={nonzero_after}/{total} ({nonzero_after/total*100:.2f}%)")
    
    # --- Example 3: Consensus Pruning (Intersection) ---
    print("\n--- EXAMPLE 3: Consensus Pruning (Intersection) ---")
    model_int = SimpleModel()
    pruner_int = Pruner(model_int)

    nonzero, total = count_nonzero(model_int)
    print(f"Before prune: nonzero={nonzero}/{total} ({nonzero/total*100:.2f}%)")

    # To get the "intersection of prune-ables", set k = N_methods
    # This prunes a weight ONLY if ALL methods agree it's unimportant.
    pruner_int.prune_consensus(
        methods=['magnitude', 'gradient'],
        sparsity_per_method=0.5, # Each method identifies its bottom 50%
        consensus_k=2,             # Prune ONLY if 2/2 methods agree
        data_batch=(dummy_inputs, dummy_labels),
        loss_fn=loss_fn
    )

    nonzero_after, total = count_nonzero(model_int)
    print(f"After consensus prune (k=2/2): nonzero={nonzero_after}/{total} ({nonzero_after/total*100:.2f}%)")
    
    # Note: The sparsity for k=2/2 will be <= 50%, as it's a stricter criteria.
    # The sparsity for k=2/3 will be somewhere around 50%, depending on mask overlap.
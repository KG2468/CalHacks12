from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset

# pruner.py
# Extensible PyTorch model pruner with "Consensus Pruning".
# - Generates masks from multiple criteria (magnitude, gradient, etc.)
# - Prunes weights that are flagged by a "majority vote" (k-out-of-N)

# ===================================================================
# --- BEGIN PARALLEL WORKER FUNCTION ---
# ===================================================================

def _parallel_eval_worker(
    model_cpu: nn.Module,
    keep_mask: Dict[str, torch.Tensor],
    eval_dataloader: DataLoader,
    loss_fn: _Loss
) -> float:
    """
    Standalone worker function for parallel mask evaluation on the CPU.
    
    This function receives a *copy* of the model on the CPU and a
    mask (which might be on the GPU). It performs the evaluation
    entirely on the CPU to avoid GPU memory conflicts.
    """
    
    # 1. Find the parameters this mask touches
    name_to_param = {}
    for module_name, module in model_cpu.named_modules():
        for name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{module_name}.{name}" if module_name else name
            if name.endswith("weight") and param is not None and full_name in keep_mask:
                name_to_param[full_name] = param

    # 2. Backup original weights (from the CPU model copy)
    original_weights = {
        name: name_to_param[name].data.clone()
        for name in keep_mask.keys() if name in name_to_param
    }

    # 3. Apply the temporary mask (in-place on the CPU model)
    with torch.no_grad():
        for name, mask in keep_mask.items():
            if name in name_to_param:
                param = name_to_param[name]
                # Ensure mask is on CPU before applying
                param.data.mul_(mask.to("cpu"))

    # 4. Evaluate the model (on CPU)
    model_cpu.eval() # Set to eval mode
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            try:
                # Data from loader might be on GPU, move to CPU
                inputs, labels = [d.to("cpu") for d in batch]
            except (ValueError, TypeError):
                print("Warning: Could not unpack (inputs, labels) from eval batch in worker.")
                continue
                
            outputs = model_cpu(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    # 5. Restore original weights (good practice, though this model copy is discarded)
    with torch.no_grad():
        for name, original_data in original_weights.items():
            if name in name_to_param:
                name_to_param[name].data.copy_(original_data)
                
    return avg_loss


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
            # thresh = torch.kthvalue(all_scores, k).values
            thresh = torch.kthvalue(all_scores.cpu(), k).values.to(all_scores.device)

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


# ===================================================================
# --- BEGIN NOVEL STOCHASTIC PRUNER (Parallelized) ---
# ===================================================================

class StochasticPruner(Pruner):
    """
    Implements the novel "Stochastic Consensus Pruning" algorithm.

    This method works by:
    1. Grouping model parameters into "chunks" (e.g., 500M params each).
    2. For each chunk:
       a. Generate N (e.g., 500-1000) random candidate masks for that chunk.
       b. For each candidate mask, *temporarily* apply it and evaluate the 
          model's performance (e.g., loss) on a small dataset.
          **This step is parallelized across multiple CPU cores.**
       c. Keep the "Top K" (e.g., 20) masks that resulted in the best performance.
       d. Compute a "consensus" mask for the chunk from these Top K masks.
    3. After all chunks are processed, combine the consensus masks and
       apply them to the model permanently.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        super().__init__(model, device)
        # Helper map to quickly find params by name
        self.name_to_param: Dict[str, torch.nn.Parameter] = {
            n: p for n, p in self._iter_prunable()
        }

    def _get_param_chunks(self, chunk_size_params: int) -> List[Dict[str, torch.nn.Parameter]]:
        """Groups prunable parameters into chunks of ~chunk_size_params."""
        chunks = []
        current_chunk = {}
        current_chunk_params = 0
        for full_name, param in self._iter_prunable():
            # Add param to current chunk
            current_chunk[full_name] = param
            current_chunk_params += param.numel()

            # If chunk is full, start a new one
            if current_chunk_params >= chunk_size_params:
                chunks.append(current_chunk)
                current_chunk = {}
                current_chunk_params = 0
                
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _generate_random_keep_mask_for_chunk(
        self, 
        chunk: Dict[str, torch.nn.Parameter], 
        sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Generates a single random *keep mask* for a specific chunk, 
        at a given global sparsity *within that chunk*.
        """
        # 1. Get random scores for only the params in this chunk
        scores = {}
        for name, param in chunk.items():
            scores[name] = torch.rand_like(param.data, device=self.device)
            
        # 2. Use the base class's global mask creator, but pass in
        #    only the scores for this chunk. This finds a global
        #    threshold *within the chunk*.
        chunk_keep_mask = self.create_keep_mask_global(scores, sparsity)
        return chunk_keep_mask

    def _evaluate_model_with_temp_mask(
        self,
        keep_mask: Dict[str, torch.Tensor],
        eval_dataloader: DataLoader,
        loss_fn: _Loss
    ) -> float:
        """
        Temporarily applies a mask, evaluates on dataloader, and restores weights.
        This is the main bottleneck.
        """
        
        # 1. Backup original weights for the parameters this mask touches
        original_weights = {}
        for name in keep_mask.keys():
            if name in self.name_to_param:
                original_weights[name] = self.name_to_param[name].data.clone()

        # 2. Apply the temporary mask (in-place)
        with torch.no_grad():
            for name, mask in keep_mask.items():
                if name in self.name_to_param:
                    param = self.name_to_param[name]
                    param.data.mul_(mask.to(param.device))

        # 3. Evaluate the model
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                try:
                    # Assuming simple (inputs, labels) batch
                    inputs, labels = [d.to(self.device) for d in batch]
                except (ValueError, TypeError):
                    print("Warning: Could not unpack (inputs, labels) from eval batch.")
                    continue
                    
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

        # 4. Restore original weights from backup (in-place)
        with torch.no_grad():
            for name, original_data in original_weights.items():
                if name in self.name_to_param:
                    self.name_to_param[name].data.copy_(original_data)
                
        self.model.train(was_training) # Restore original training state
        return avg_loss

    
    def prune_stochastic_consensus(
        self,
        eval_dataloader: DataLoader,
        loss_fn: _Loss,
        num_chunks: int = 15,
        num_masks_per_chunk: int = 500,
        sparsity_per_mask: float = 0.5,
        top_k_masks: int = 20,
        consensus_k: int = 20,
        num_workers: int = 0
    ):
        """
        Implements the novel stochastic consensus pruning algorithm.
        
        **NEW ARGUMENT:**
        - num_workers: Number of parallel CPU processes to use for mask
                       evaluation. If 0, runs serially (original behavior).
                       Set to os.cpu_count() for max parallelism.
        """
        print("--- Starting Stochastic Consensus Pruning ---")
        
        # --- Handle parallel processing settings ---
        max_workers = 0
        if num_workers > 0:
            max_workers = min(num_workers, os.cpu_count() or 1)
            print(f"Parallel processing enabled with {max_workers} workers.")
        else:
            print("Running in serial mode (num_workers=0).")

        # --- Dynamically calculate chunk size ---
        total_prunable_params = sum(p.numel() for p in self.name_to_param.values())
        if total_prunable_params == 0:
            print("Warning: No prunable parameters found. Exiting.")
            return
            
        if num_chunks <= 0:
            print(f"Warning: num_chunks must be > 0. Setting to 1.")
            num_chunks = 1
            
        # Ceiling division to get chunk size
        chunk_size_params = (total_prunable_params + num_chunks - 1) // num_chunks
        
        print(f"Total prunable params: {total_prunable_params:,}")
        print(f"Requested num_chunks: {num_chunks}")
        print(f"Calculated chunk_size_params: {chunk_size_params:,} (params per chunk)")
        # --- End dynamic calculation ---

        if consensus_k > top_k_masks:
            print(f"Warning: consensus_k ({consensus_k}) > top_k_masks ({top_k_masks}).")
            print(f"Setting consensus_k = {top_k_masks} (i.e., full intersection).")
            consensus_k = top_k_masks

        # 1. Chunk parameters
        print(f"Grouping parameters into chunks of ~{chunk_size_params:,} params...")
        param_chunks = self._get_param_chunks(chunk_size_params)
        print(f"Total prunable parameters chunked into {len(param_chunks)} chunks.")

        # This will hold the final mask for the *entire* model, combined from all chunks.
        # We build it on CPU to save GPU memory.
        final_keep_mask_cpu: Dict[str, torch.Tensor] = {
            name: torch.ones_like(param, dtype=torch.bool, device="cpu")
            for name, param in self.name_to_param.items()
        }
        
        # 2. Process each chunk independently
        for i, chunk in enumerate(param_chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(param_chunks)} ---")
            chunk_mask_scores = [] # List to store (score, keep_mask)

            # 3. Generate N random masks for this chunk
            print(f"Generating {num_masks_per_chunk} random masks...")
            masks_to_eval = []
            for _ in range(num_masks_per_chunk):
                masks_to_eval.append(
                    self._generate_random_keep_mask_for_chunk(chunk, sparsity_per_mask)
                )

            # 4. Evaluate N masks (serially or in parallel)
            if max_workers > 0:
                # --- PARALLEL PATH ---
                print(f"Evaluating {len(masks_to_eval)} masks in parallel...")
                # Move model to CPU *once* to be pickled and sent to workers
                model_cpu = self.model.to("cpu")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs
                    futures = [
                        executor.submit(
                            _parallel_eval_worker,
                            model_cpu,
                            mask,
                            eval_dataloader,
                            loss_fn
                        )
                        for mask in masks_to_eval
                    ]
                    
                    # Collect results as they complete
                    for j, future in enumerate(futures):
                        avg_loss = future.result()
                        # Get the corresponding mask (masks_to_eval[j])
                        # We store score = -loss (lower loss is better)
                        # We also move the mask to CPU to save GPU RAM
                        cpu_mask = {k: v.to("cpu") for k, v in masks_to_eval[j].items()}
                        chunk_mask_scores.append( (-avg_loss, cpu_mask) )

                        if (j+1) % (num_masks_per_chunk // 10 or 1) == 0:
                            print(f"  ... collected result {j+1}/{num_masks_per_chunk} (Loss: {avg_loss:.4f})")
                
                # Move main model back to original device (if it was on CPU, this is a no-op)
                self.model.to(self.device)

            else:
                # --- SERIAL PATH ---
                print(f"Evaluating {len(masks_to_eval)} masks serially...")
                for j, random_chunk_keep_mask in enumerate(masks_to_eval):
                    # Evaluate the model with this *partial* mask
                    avg_loss = self._evaluate_model_with_temp_mask(
                        random_chunk_keep_mask, eval_dataloader, loss_fn
                    )
                    
                    # We store score = -loss (since lower loss is better)
                    # We also move the mask to CPU to save GPU RAM
                    cpu_mask = {k: v.to("cpu") for k, v in random_chunk_keep_mask.items()}
                    chunk_mask_scores.append( (-avg_loss, cpu_mask) )

                    if (j+1) % (num_masks_per_chunk // 10 or 1) == 0:
                        print(f"  ... evaluated mask {j+1}/{num_masks_per_chunk} (Loss: {avg_loss:.4f})")
            
            # --- End of parallel/serial block ---

            # 5. Find the top K masks for this chunk
            chunk_mask_scores.sort(key=lambda x: x[0], reverse=True) # Sort by score (high to low)
            top_masks = [mask for score, mask in chunk_mask_scores[:top_k_masks]]
            
            if not top_masks:
                print("Warning: No masks were generated or scored. Skipping chunk.")
                continue

            print(f"Found top {len(top_masks)} masks. Computing consensus...")

            # 6. Get consensus (overlap) from the top K masks
            for name in chunk.keys():
                # Get all top masks for this specific parameter
                mask_list = [m[name].to(self.device) for m in top_masks if name in m]
                if not mask_list:
                    continue
                
                mask_stack = torch.stack(mask_list) # [top_k, *param_shape]
                
                # Vote to PRUNE = ~mask (False)
                # We sum the 'False' values (prune votes)
                prune_votes = torch.sum((~mask_stack).int(), dim=0)
                
                # Keep if (votes_to_prune < consensus_k)
                chunk_final_keep = (prune_votes < consensus_k).to(dtype=torch.bool)
                
                # Store this chunk's result in the *global* final mask (on CPU)
                final_keep_mask_cpu[name] = chunk_final_keep.to("cpu")

        # 7. Apply the final combined mask
        print("\n--- Stochastic Consensus Pruning Finished ---")
        print("Applying final combined mask to model...")
        
        # Move final mask to the correct device for applying
        device_masks = {k: v.to(self.device) for k, v in final_keep_mask_cpu.items()}
        self.apply_masks(device_masks, remove_hooks=True)
        print("Pruning complete.")


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
    
    
    # --- Example 2: Original Consensus Pruning ---
    print("\n--- EXAMPLE 2: Original Consensus Pruning ---")
    model_con = SimpleModel()
    pruner_con = Pruner(model_con)
    
    # Create dummy data for gradient scoring
    dummy_inputs = torch.randn(16, 100)
    dummy_labels = torch.randint(0, 10, (16,))
    loss_fn = nn.CrossEntropyLoss()
    
    nonzero, total = count_nonzero(model_con)
    print(f"Before prune: nonzero={nonzero}/{total} ({nonzero/total*100:.2f}%)")

    # Prune 50% by 2 methods.
    # Prune a weight if at least 2 methods vote to prune it (intersection)
    pruner_con.prune_consensus(
        methods=['magnitude', 'gradient'],
        sparsity_per_method=0.5, # Each method identifies its bottom 50%
        consensus_k=2,             # Prune if 2/2 methods agree
        data_batch=(dummy_inputs, dummy_labels),
        loss_fn=loss_fn
    )

    nonzero_after, total = count_nonzero(model_con)
    print(f"After consensus prune (k=2/2): nonzero={nonzero_after}/{total} ({nonzero_after/total*100:.2f}%)")
    
    
    # --- Example 3: NEW Stochastic Consensus Pruning ---
    print("\n--- EXAMPLE 3: NEW Stochastic Consensus Pruning ---")
    model_stoch = SimpleModel()
    # Use the new StochasticPruner class
    pruner_stoch = StochasticPruner(model_stoch)
    
    # This algorithm requires an *evaluation dataloader*
    # We'll use the dummy data
    eval_inputs = torch.randn(64, 100) # A small, fast eval set
    eval_labels = torch.randint(0, 10, (64,))
    eval_dataset = TensorDataset(eval_inputs, eval_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32)
    
    loss_fn = nn.CrossEntropyLoss()
    
    nonzero, total = count_nonzero(model_stoch)
    print(f"Before prune: nonzero={nonzero}/{total} ({nonzero/total*100:.2f}%)")
    
    # For a SimpleModel, total prunable params = 7040.
    # The comment originally said "Let's make 2 chunks".
    # So we now pass num_chunks=2.
    
    pruner_stoch.prune_stochastic_consensus(
        eval_dataloader=eval_dataloader,
        loss_fn=loss_fn,
        num_chunks=2,                 # <-- CHANGED: We want 2 chunks
        num_masks_per_chunk=50,       # Num random masks to test (set to 500-1000)
        sparsity_per_mask=0.5,        # Sparsity for each random mask
        top_k_masks=10,               # Find 10 best-performing masks
        consensus_k=10                # Prune if all 10 agree (intersection)
    )

    nonzero_after, total = count_nonzero(model_stoch)
    print(f"After stochastic prune (k=10/10): nonzero={nonzero_after}/{total} ({nonzero_after/total*100:.2f}%)")

# Add the following to the same file containing Pruner / StochasticPruner

import math
from typing import Tuple, Optional

class PrunerWithSVD(Pruner):
    """
    Extends Pruner with methods to convert pruned weight matrices to low-rank
    approximations (via SVD). Two behaviors:
      - factorize (preferred for Linear): replace nn.Linear with two smaller linears.
      - reconstruct_dense: compute low-rank reconstruction and write it back to param.data.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        super().__init__(model, device)
        # build a mapping full_param_name -> (module, param_name_in_module, param)
        self._build_name_map()

    def _build_name_map(self):
        """Populate a lookup map from full param-names -> (module, param_name, param)."""
        self._name_map = {}
        for module_name, module in self.model.named_modules():
            for name, param in list(module.named_parameters(recurse=False)):
                full_name = f"{module_name}.{name}" if module_name else name
                self._name_map[full_name] = (module, name, param)

    def _get_module_by_name(self, module_name: str) -> nn.Module:
        """Return module object by dotted name ('' -> root/self.model)."""
        if module_name == "" or module_name is None:
            return self.model
        parts = module_name.split(".")
        cur = self.model
        for p in parts:
            cur = getattr(cur, p)
        return cur

    def _solve_rank_for_budget(self, kept_count: int, m: int, n: int, min_rank: int = 1) -> int:
        """
        Solve for k such that k*(m + n) ~= kept_count.
        Return at least min_rank and at most min(m, n).
        """
        if kept_count <= 0:
            return 0
        k = math.ceil(kept_count / (m + n))
        k = max(min_rank, k)
        k = min(k, min(m, n))
        return k

    def _svd_decompose(self, W, k=None, device=None):
        # Move to CPU if not already
        W_cpu = W.to("cpu")
        
        # Convert to float32 if it’s half precision
        if W_cpu.dtype == torch.float16:
            W_cpu = W_cpu.float()

        # Perform SVD
        try:
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        except Exception:
            # fallback to old API
            U, S, Vh = torch.svd(W_cpu)
        
        if k is not None:
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]
        
        return U, S, Vh


    def apply_svd_on_masks(
        self,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "factorize",           # "factorize" | "reconstruct_dense"
        min_rank: int = 1,
        svd_device: Optional[torch.device] = None,
        inplace: bool = True
    ):
        """
        Convert pruned parameters (given by masks) into low-rank approximations.

        - masks: dict mapping full param name -> boolean keep-mask. If None, will use self.masks.
        - mode:
          - "factorize": for nn.Linear layers, replace them with two smaller Linears
                         so parameter count becomes ~k*(m+n). For convs: fall back to reconstruct.
          - "reconstruct_dense": compute low-rank reconstruction and assign param.data = W_lowrank.
        - svd_device: device to run SVD on (usually "cpu" to save GPU memory). If None, runs on CPU.
        - inplace: if True, perform replacements/assignments.
        """
        if masks is None:
            masks = self.masks
        if not masks:
            print("No masks provided/loaded. Nothing to do.")
            return

        # Ensure name map is current
        self._build_name_map()

        for full_name, mask in masks.items():
            if full_name not in self._name_map:
                print(f"Warning: {full_name} not found in model; skipping SVD for it.")
                continue

            module, param_name, param = self._name_map[full_name]
            W = param.data.detach()
            # handle conv kernels by flattening
            is_conv = (W.ndim == 4)  # (out, in, kh, kw)
            if is_conv:
                out_ch, in_ch, kh, kw = W.shape
                W_mat = W.view(out_ch, -1)  # (out, in*kh*kw)
                m, n = W_mat.shape
                kept = int(mask.to(torch.int).sum().item())
                k = self._solve_rank_for_budget(kept, m, n, min_rank=min_rank)
                if k <= 0:
                    print(f"  {full_name}: no kept elements -> skipping")
                    continue
                # compute SVD (prefer CPU)
                U_k, S_k, Vt_k = self._svd_decompose(W_mat, k, device=svd_device)
                # reconstruct dense low-rank
                # W_low = U_k @ diag(S_k) @ Vt_k
                W_low = (U_k * S_k.unsqueeze(0)) @ Vt_k
                W_low = W_low.view_as(W)
                if mode == "reconstruct_dense" and inplace:
                    with torch.no_grad():
                        param.data.copy_(W_low.to(param.device))
                    print(f"  {full_name}: conv reconstructed as dense low-rank k={k}")
                else:
                    # factorizing convs into efficient convs is not implemented here
                    # fallback to dense reconstruction
                    with torch.no_grad():
                        param.data.copy_(W_low.to(param.device))
                    print(f"  {full_name}: conv fallback reconstructed as dense low-rank k={k}")
                continue

            # for Linear-like weights (2D) or other 2D params:
            if W.ndim != 2:
                # If not 2D, try flattening to (out, -1)
                W_mat = W.view(W.shape[0], -1)
            else:
                W_mat = W

            m, n = W_mat.shape
            kept = int(mask.to(torch.int).sum().item())
            k = self._solve_rank_for_budget(kept, m, n, min_rank=min_rank)
            if k <= 0:
                print(f"  {full_name}: no kept elements -> skipping")
                continue

            U_k, S_k, Vt_k = self._svd_decompose(W_mat, k, device=svd_device)

            if mode == "reconstruct_dense":
                # Reconstruct and write back into param
                W_low = (U_k * S_k.unsqueeze(0)) @ Vt_k
                W_low = W_low.view_as(param.data)
                if inplace:
                    with torch.no_grad():
                        param.data.copy_(W_low.to(param.device))
                print(f"  {full_name}: reconstructed dense low-rank k={k}")
                continue

            # mode == "factorize"
            # Only attempt factorization for nn.Linear modules (most straightforward)
            if isinstance(module, nn.Linear) and param_name == "weight":
                # W shape is (out, in)
                out, inp = m, n
                # create two linear layers: first maps inp -> k (no bias), second maps k -> out (bias)
                first = nn.Linear(inp, k, bias=False)
                second = nn.Linear(k, out, bias=True)

                # Set weights: first.weight <- Vt_k (k, inp)
                # second.weight <- (U_k * S_k.unsqueeze(0)) (out, k)
                with torch.no_grad():
                    first.weight.copy_(Vt_k.to(first.weight.device))
                    second.weight.copy_((U_k * S_k.unsqueeze(0)).to(second.weight.device))
                    # transfer bias if original module had bias
                    if hasattr(module, "bias") and module.bias is not None:
                        second.bias.copy_(module.bias.data.to(second.bias.device))
                    else:
                        # ensure zero bias
                        second.bias.zero_()

                # Build a sequential to replace the original linear:
                # x -> first(x) -> second(x)
                new_seq = nn.Sequential(first, second)

                # Replace the module in the parent with new_seq
                # Need to find parent module to set attribute
                # module_name can be found by scanning named_modules to match `module` object
                parent_name = None
                child_attr = None
                for mod_name, mod in self.model.named_modules():
                    if mod is module:
                        # mod_name is the name of the module itself (we want its parent)
                        parts = mod_name.split(".") if mod_name else []
                        if parts:
                            parent_name = ".".join(parts[:-1])
                            child_attr = parts[-1]
                        else:
                            # root module is the module itself; cannot replace root's attribute
                            parent_name = ""
                            child_attr = None
                        break

                if child_attr is None:
                    # attempt fallback: find attribute on model that references this module
                    found = False
                    for name, mod in self.model.named_children():
                        if mod is module:
                            parent_name = ""
                            child_attr = name
                            found = True
                            break
                    if not found:
                        # last resort: can't replace safely, do dense reconstruction instead
                        W_low = (U_k * S_k.unsqueeze(0)) @ Vt_k
                        with torch.no_grad():
                            param.data.copy_(W_low.view_as(param.data).to(param.device))
                        print(f"  {full_name}: could not replace Linear module, wrote dense instead (k={k})")
                        continue

                parent = self._get_module_by_name(parent_name)
                # set attribute
                setattr(parent, child_attr, new_seq)
                print(f"  {full_name}: replaced nn.Linear '{parent_name + '.' if parent_name else ''}{child_attr}' with factorized rank-{k} pair (params ~ {k*(m+n):,})")
                # After replacement, rebuild the name map
                self._build_name_map()
            else:
                # fallback: reconstruct dense
                W_low = (U_k * S_k.unsqueeze(0)) @ Vt_k
                W_low = W_low.view_as(param.data)
                if inplace:
                    with torch.no_grad():
                        param.data.copy_(W_low.to(param.device))
                print(f"  {full_name}: non-Linear or unsupported module -> reconstructed dense low-rank k={k}")

        print("SVD conversion done.")

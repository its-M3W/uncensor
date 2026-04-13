"""
Difference-in-means direction extraction and selection.

Paper: https://arxiv.org/abs/2406.11717
§2.3 — "Extracting a refusal direction" (difference-in-means).
§C.1 — direction selection algorithm with bypass / induce / kl scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from tqdm.auto import tqdm

from .model import RefusalModel


# -----------------------------------------------------------------------------
# Activation capture
# -----------------------------------------------------------------------------
def collect_activations(
    model: RefusalModel,
    prompts: Sequence[str],
    token_positions: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    """Collect residual-stream activations at the given post-instruction token
    positions, for every layer.

    §2.3 — activations are captured per layer l and per post-instruction token
    position i ∈ I. We use a forward-pre-hook on each decoder layer to grab
    the residual stream going into that layer (which equals the output of the
    previous layer / of the embedding for layer 0).

    Returns a tensor of shape `(n_layers, len(token_positions), n_prompts, d_model)`.
    """
    n_prompts = len(prompts)
    # Pre-allocate on CPU to keep GPU memory light for large models.
    out = torch.zeros(
        (model.n_layers, len(token_positions), n_prompts, model.d_model),
        dtype=torch.float32,
    )

    # Buffer that the hook writes into, one slot per layer. Re-used each batch.
    buffer: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            # inputs is a tuple; the first element is the residual stream.
            # Shape: (batch, seq_len, d_model)
            buffer[layer_idx] = inputs[0].detach()
            return None  # don't modify inputs
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        for start in tqdm(range(0, n_prompts, batch_size), desc="extract activations"):
            batch_prompts = [model.format(p) for p in prompts[start : start + batch_size]]
            enc = model.tokenize(batch_prompts)
            with torch.no_grad():
                model.model(**enc)
            for layer_idx in range(model.n_layers):
                # buffer[layer_idx]: (batch, seq_len, d_model)
                resid = buffer[layer_idx]
                for pi, pos in enumerate(token_positions):
                    # pos is a negative index relative to the last token.
                    # resid[:, pos, :] -> (batch, d_model)
                    out[layer_idx, pi, start : start + resid.shape[0], :] = (
                        resid[:, pos, :].to(torch.float32).cpu()
                    )
            buffer.clear()
    finally:
        RefusalModel.remove_hooks(handles)
    return out


# -----------------------------------------------------------------------------
# Difference-in-means
# -----------------------------------------------------------------------------
def difference_in_means(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> torch.Tensor:
    """§2.3 — r^(l)_i = μ^(l)_i − ν^(l)_i, where μ and ν are the mean
    activations for harmful and harmless prompts respectively.

    Args:
        harmful_acts, harmless_acts: (n_layers, n_positions, n_prompts, d_model)

    Returns:
        candidate directions: (n_layers, n_positions, d_model)
    """
    mu = harmful_acts.mean(dim=2)     # -> (n_layers, n_positions, d_model)
    nu = harmless_acts.mean(dim=2)    # -> (n_layers, n_positions, d_model)
    return mu - nu                    # -> (n_layers, n_positions, d_model)


# -----------------------------------------------------------------------------
# Direction selection (§C.1)
# -----------------------------------------------------------------------------
@dataclass
class DirectionCandidate:
    """One entry from the |I| × L candidate grid with its three scores."""
    layer: int
    position: int     # negative index relative to the last prompt token
    vector: torch.Tensor  # the un-normalized difference-in-means vector r^(l)_i
    bypass_score: float
    induce_score: float
    kl_score: float


def select_best(
    candidates: List[DirectionCandidate],
    *,
    n_layers_total: int,
    induce_score_min: float,
    kl_score_max: float,
    max_layer_frac: float,
) -> DirectionCandidate:
    """§C.1 — "select r^(l*)_(i*) to be the direction with minimum bypass_score,
    subject to induce_score > 0, kl_score < 0.1, l < 0.8L".
    """
    layer_cutoff = int(max_layer_frac * n_layers_total)
    filtered = [
        c
        for c in candidates
        if c.induce_score > induce_score_min
        and c.kl_score < kl_score_max
        and c.layer < layer_cutoff
    ]
    if not filtered:
        raise RuntimeError(
            "No candidate direction passed the §C.1 selection filters. "
            "Try relaxing `kl_score_max` or inspecting per-layer scores."
        )
    filtered.sort(key=lambda c: c.bypass_score)
    return filtered[0]

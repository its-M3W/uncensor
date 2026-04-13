"""
Model interventions: directional ablation, activation addition, weight
orthogonalization.

Paper: https://arxiv.org/abs/2406.11717
§2.4 — activation addition and directional ablation equations.
§3.1 — directional ablation bypasses refusal.
§3.2 — activation addition induces refusal.
§4.1 — weight orthogonalization.
§E — proof that weight orthogonalization equals directional ablation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List

import torch
from torch import nn

from .model import RefusalModel, discover_residual_writers


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _project_onto(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return (direction · x) direction, where `direction` is a unit vector.

    Broadcasting: x has shape (..., d_model), direction has shape (d_model,).
    Result has the same shape as x.
    """
    # coeff: (..., 1); direction: (d_model,). Result: (..., d_model).
    coeff = (x * direction).sum(dim=-1, keepdim=True)
    return coeff * direction


# -----------------------------------------------------------------------------
# Directional ablation (§2.4 Eq. 4)
# -----------------------------------------------------------------------------
@contextmanager
def directional_ablation(
    model: RefusalModel,
    direction: torch.Tensor,
) -> Iterator[None]:
    """Context manager that ablates `direction` from the residual stream at
    every layer and every token position for the duration of the `with` block.

    §2.4 Eq. 4 — x' = x − (r̂ᵀ x) r̂
    §2.4 — "We perform this operation at every activation x^(l)_i and x̃^(l)_i,
    across all layers l and all token positions i."
    """
    direction = direction.to(dtype=torch.float32)
    direction = direction / direction.norm()  # r̂
    direction = direction.to(device=model.device)

    def make_hook(_layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]  # (batch, seq_len, d_model)
            d = direction.to(dtype=x.dtype)
            # Subtract the projection along r̂ from every token, every layer.
            new_x = x - _project_onto(x, d)
            return (new_x,) + inputs[1:]
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)


# -----------------------------------------------------------------------------
# Activation addition (§2.4)
# -----------------------------------------------------------------------------
@contextmanager
def activation_addition(
    model: RefusalModel,
    direction: torch.Tensor,
    layer_idx: int,
) -> Iterator[None]:
    """§2.4 — add the un-normalized difference-in-means vector r to the
    residual stream at a single layer, across all token positions.

    Note: the paper's equation uses `r` (un-normalized), not `r̂`. This is
    an intentional design choice — the magnitude of r matters here, not just
    the direction. See §2.3 "Note that each such vector is meaningful in both
    (1) its direction ... and (2) its magnitude".
    """
    direction = direction.to(device=model.device)

    def make_hook(this_layer: int):
        def hook(_module, inputs):
            x = inputs[0]
            if this_layer == layer_idx:
                d = direction.to(dtype=x.dtype)
                new_x = x + d  # broadcasts over (batch, seq_len, d_model)
                return (new_x,) + inputs[1:]
            return None
        return hook

    handles = model.register_forward_pre_hooks(make_hook)
    try:
        yield
    finally:
        RefusalModel.remove_hooks(handles)


# -----------------------------------------------------------------------------
# Weight orthogonalization (§4.1, §E)
# -----------------------------------------------------------------------------
def orthogonalize_weights(model: RefusalModel, direction: torch.Tensor) -> None:
    """§4.1 Eq. 5 — W'_out = W_out − r̂ r̂ᵀ W_out.

    Modifies every matrix that writes to the residual stream (embedding,
    positional embedding if any, attention o_proj, MLP down_proj) as well as
    any biases on those projections.

    This mutates the model weights in-place. It is mathematically equivalent
    to directional ablation (§E) but requires no hooks at inference time.
    """
    writers = discover_residual_writers(model.model)
    r_hat = direction.to(device=model.device, dtype=torch.float32)
    r_hat = r_hat / r_hat.norm()

    @torch.no_grad()
    def _ortho_linear(layer: nn.Linear) -> None:
        W = layer.weight.data  # (d_model, d_input) — rows live in R^d_model
        r = r_hat.to(dtype=W.dtype)
        # Subtract the r̂ component from every column of W: W ← W − r̂ (r̂ᵀ W).
        proj = torch.outer(r, r @ W)   # (d_model, d_input)
        W.sub_(proj)
        if layer.bias is not None:
            b = layer.bias.data
            r_b = r_hat.to(dtype=b.dtype)
            b.sub_((r_b @ b) * r_b)

    @torch.no_grad()
    def _ortho_embedding(layer: nn.Embedding) -> None:
        # Embedding weight shape: (vocab, d_model). Each row is a vector in
        # R^d_model that is written to the residual stream. Orthogonalize per row.
        W = layer.weight.data
        r = r_hat.to(dtype=W.dtype)
        # coeffs: (vocab,)  — dot of every row with r̂
        coeffs = W @ r
        W.sub_(torch.outer(coeffs, r))

    _ortho_embedding(writers.embed)
    if writers.positional_embed is not None:
        _ortho_embedding(writers.positional_embed)
    for lin in writers.attn_out_weights:
        _ortho_linear(lin)
    for lin in writers.mlp_out_weights:
        _ortho_linear(lin)
    # Output biases that were NOT inside a linear layer (rare — e.g. a standalone
    # residual-stream bias vector). The writers struct already handled biases
    # attached to the linear layers above.

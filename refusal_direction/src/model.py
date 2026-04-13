"""
Wrapper around a HuggingFace causal LM that exposes the residual stream for
activation capture and for hook-based interventions.

Paper: https://arxiv.org/abs/2406.11717
§2.1 Background — residual-stream notation x^(l)_i ∈ R^d_model.
§2.3 — activations are captured per layer l and post-instruction token position i.
§2.4 — interventions are implemented as hooks on the residual stream.
§4.1 — weight orthogonalization modifies matrices that write to the residual stream.

Design note: we hook the *input* of each transformer decoder layer, which is
the residual-stream activation going into that layer (pre-norm or post-norm,
the definition of "residual stream" in the paper). This matches the paper's
convention where x^(l) is the pre-layer residual (§2.1 background description
of the transformer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .utils import detect_family, format_prompt, resolve_device, resolve_dtype


# -----------------------------------------------------------------------------
# Decoder-layer discovery
# -----------------------------------------------------------------------------
def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the ModuleList of transformer decoder layers for this model.

    Covers the architectures the paper studies (Llama, Qwen, Gemma, Yi are all
    Llama-style; the path is `model.model.layers`). [UNSPECIFIED] — the paper
    does not dictate a specific Python hook point; the input of each layer is
    the canonical residual-stream location.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback paths for other HF architectures.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(
        f"Cannot locate decoder layers on {type(model).__name__}; "
        "extend _get_decoder_layers() to support this architecture."
    )


# -----------------------------------------------------------------------------
# Residual-stream writer discovery (§4.1)
# -----------------------------------------------------------------------------
@dataclass
class ResidualWriters:
    """Handles to every parameter that writes to the residual stream.

    §4.1 — "the matrices that write to the residual stream are: the embedding
    matrix, the positional embedding matrix, attention out matrices, and MLP
    out matrices. Orthogonalizing all of these matrices, as well as any output
    biases, with respect to the direction r̂ effectively prevents the model
    from ever writing r̂ to its residual stream."

    Llama-family models (including Qwen, Gemma, Yi) use RoPE and therefore
    have no learned positional embedding matrix — that is a correct omission,
    not an ambiguity.
    """

    embed: nn.Embedding
    attn_out_weights: List[nn.Linear] = field(default_factory=list)
    attn_out_biases: List[torch.nn.Parameter] = field(default_factory=list)
    mlp_out_weights: List[nn.Linear] = field(default_factory=list)
    mlp_out_biases: List[torch.nn.Parameter] = field(default_factory=list)
    positional_embed: Optional[nn.Embedding] = None


def discover_residual_writers(model: PreTrainedModel) -> ResidualWriters:
    """Find every weight that writes to the residual stream.

    We look for the standard names used in Llama-style configs:
      - `embed_tokens` — token embedding
      - `self_attn.o_proj` — attention output projection (writes to residual)
      - `mlp.down_proj` — MLP down projection (writes to residual)
    Biases along these paths are collected separately so orthogonalization can
    zero out the ˆr component of each bias too.
    """
    writers = ResidualWriters(embed=model.get_input_embeddings())
    # Positional embedding matrix if the model has one (§4.1).
    pe = getattr(getattr(model, "model", model), "embed_positions", None)
    if isinstance(pe, nn.Embedding):
        writers.positional_embed = pe

    for layer in _get_decoder_layers(model):
        # Attention output projection.
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is not None:
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            if isinstance(o_proj, nn.Linear):
                writers.attn_out_weights.append(o_proj)
                if o_proj.bias is not None:
                    writers.attn_out_biases.append(o_proj.bias)
        # MLP output / down projection.
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is not None:
            down = getattr(mlp, "down_proj", None) or getattr(mlp, "c_proj", None)
            if isinstance(down, nn.Linear):
                writers.mlp_out_weights.append(down)
                if down.bias is not None:
                    writers.mlp_out_biases.append(down.bias)

    return writers


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------
class RefusalModel:
    """Thin wrapper that owns the HF model + tokenizer and exposes the hook
    points needed by extraction.py and interventions.py.
    """

    def __init__(
        self,
        name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        self.name = name
        self.family = detect_family(name)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            # [UNSPECIFIED] pad token not discussed; we reuse EOS which matches
            # HuggingFace's default for left-padded causal LM batching.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.layers = _get_decoder_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size

    # -------------------------------------------------------------------------
    # Prompt handling
    # -------------------------------------------------------------------------
    def format(self, instruction: str) -> str:
        """Apply the paper's chat template (§C.3, Table 6)."""
        return format_prompt(instruction, self.family, self.tokenizer)

    def tokenize(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of already-formatted prompts (left-padded)."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    # -------------------------------------------------------------------------
    # Hook helpers
    # -------------------------------------------------------------------------
    def register_forward_pre_hooks(
        self,
        make_hook: Callable[[int], Callable],
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register a forward-pre-hook on every transformer block.

        §2.4 — directional ablation "at every activation x^(l)_i ... across all
        layers l and all token positions i". A forward-pre-hook on each block
        sees the residual-stream tensor going into that block and can mutate
        it before the block runs.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, layer in enumerate(self.layers):
            handles.append(layer.register_forward_pre_hook(make_hook(layer_idx)))
        return handles

    @staticmethod
    def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        for h in handles:
            h.remove()

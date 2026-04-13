"""
End-to-end pipeline: load data → extract → select direction → evaluate.

Paper: https://arxiv.org/abs/2406.11717
This module ties together every stage of the paper's method:
  §2.2 data loading → §2.3 extraction → §C.1 selection → §3.1/§3.2 eval →
  §4.1 optional weight orthogonalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import Splits, build_splits
from .extraction import (
    DirectionCandidate,
    collect_activations,
    difference_in_means,
    select_best,
)
from .generate import generate_batched
from .interventions import (
    activation_addition,
    directional_ablation,
    orthogonalize_weights,
)
from .metrics import (
    SafetyScorer,
    refusal_metric_from_logits,
    refusal_rate,
    safety_rate,
)
from .model import RefusalModel
from .utils import resolve_refusal_tokens, set_seed


# -----------------------------------------------------------------------------
# Scoring harness for §C.1 direction selection
# -----------------------------------------------------------------------------
def _logits_at_last_token(
    model: RefusalModel,
    prompts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """Return the logits at the last token for each prompt, shape (n, vocab).
    Any hook intervention active via context manager applies to this call.
    """
    out: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = [model.format(p) for p in prompts[start : start + batch_size]]
        enc = model.tokenize(batch)
        with torch.no_grad():
            logits = model.model(**enc).logits  # (b, seq, vocab)
        # Last non-pad position per row. Left padding makes the last column
        # always the final token, so this slice is valid.
        out.append(logits[:, -1, :].detach().cpu())
    return torch.cat(out, dim=0)


def _mean_refusal_metric(
    logits: torch.Tensor, refusal_token_ids: List[int]
) -> float:
    metric = refusal_metric_from_logits(logits, refusal_token_ids)  # (n,)
    return float(metric.mean().item())


def _mean_kl_at_last_token(
    original_logits: torch.Tensor, ablated_logits: torch.Tensor
) -> float:
    """§C.1 — "compute the average KL divergence between the probability
    distributions at the last token position" between the original and the
    intervened run on harmless prompts.
    """
    p = F.log_softmax(original_logits.to(torch.float32), dim=-1)
    q = F.log_softmax(ablated_logits.to(torch.float32), dim=-1)
    # KL(P || Q) = Σ exp(p) (p − q)
    kl = (p.exp() * (p - q)).sum(dim=-1)  # (n,)
    return float(kl.mean().item())


def score_candidates(
    model: RefusalModel,
    *,
    candidates: torch.Tensor,  # (n_layers, n_positions, d_model)
    token_positions: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    refusal_token_ids: List[int],
    batch_size: int,
) -> List[DirectionCandidate]:
    """§C.1 — compute bypass_score, induce_score, kl_score for every candidate.

    Warning: this is the expensive step of the pipeline (forward passes over
    the validation set for every candidate direction). The paper reports that
    it takes ~1 hour for 72B models.
    """
    # Baseline logits on harmless val, with no intervention (for KL reference).
    baseline_harmless_logits = _logits_at_last_token(model, harmless_val, batch_size)

    results: List[DirectionCandidate] = []
    n_layers, n_positions, _ = candidates.shape
    total = n_layers * n_positions
    with tqdm(total=total, desc="score candidates") as pbar:
        for layer_idx in range(n_layers):
            for pos_idx in range(n_positions):
                r = candidates[layer_idx, pos_idx]  # (d_model,)

                # bypass_score: mean refusal metric under directional ablation,
                # evaluated on D_harmful^(val).
                with directional_ablation(model, r):
                    logits = _logits_at_last_token(model, harmful_val, batch_size)
                bypass = _mean_refusal_metric(logits, refusal_token_ids)

                # induce_score: mean refusal metric under activation addition
                # at the extraction layer, on D_harmless^(val).
                with activation_addition(model, r, layer_idx=layer_idx):
                    logits = _logits_at_last_token(model, harmless_val, batch_size)
                induce = _mean_refusal_metric(logits, refusal_token_ids)

                # kl_score: mean KL divergence between original and ablated on
                # D_harmless^(val), last token position.
                with directional_ablation(model, r):
                    ablated_harmless_logits = _logits_at_last_token(
                        model, harmless_val, batch_size
                    )
                kl = _mean_kl_at_last_token(
                    baseline_harmless_logits, ablated_harmless_logits
                )

                results.append(
                    DirectionCandidate(
                        layer=layer_idx,
                        position=token_positions[pos_idx],
                        vector=r.clone(),
                        bypass_score=bypass,
                        induce_score=induce,
                        kl_score=kl,
                    )
                )
                pbar.update(1)
    return results


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
@dataclass
class PipelineResult:
    best_layer: int
    best_position: int
    bypass_score: float
    induce_score: float
    kl_score: float
    bypass_refusal_rate_baseline: float
    bypass_refusal_rate_intervened: float
    induce_refusal_rate_baseline: float
    induce_refusal_rate_intervened: float
    orth_refusal_rate: Optional[float] = None
    bypass_safety_rate_baseline: Optional[float] = None
    bypass_safety_rate_intervened: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def run_pipeline(cfg: Dict) -> Tuple[PipelineResult, torch.Tensor]:
    """Run the full refusal-direction pipeline on a single model.

    Returns the metric summary and the selected direction tensor.
    """
    set_seed(cfg["data"]["seed"])

    model = RefusalModel(
        name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )
    refusal_token_ids = resolve_refusal_tokens(model.family, model.tokenizer)
    print(f"[pipeline] family={model.family}, refusal tokens={refusal_token_ids}")

    # --- Data ---
    splits = build_splits(
        harmful_sources=cfg["data"]["harmful_sources"],
        n_train=cfg["data"]["n_train"],
        n_val=cfg["data"]["n_val"],
        n_bypass_eval=cfg["data"]["n_bypass_eval"],
        n_induce_eval=cfg["data"]["n_induce_eval"],
        seed=cfg["data"]["seed"],
    )
    print(
        f"[pipeline] splits: harmful={len(splits.harmful_train)}+{len(splits.harmful_val)} "
        f"harmless={len(splits.harmless_train)}+{len(splits.harmless_val)} "
        f"bypass_eval={len(splits.bypass_eval)} induce_eval={len(splits.induce_eval)}"
    )

    # --- Activation extraction (§2.3) ---
    token_positions: List[int] = cfg["extraction"]["post_instruction_positions"]
    batch_size = cfg["extraction"]["batch_size"]
    harmful_acts = collect_activations(
        model, splits.harmful_train, token_positions, batch_size
    )
    harmless_acts = collect_activations(
        model, splits.harmless_train, token_positions, batch_size
    )
    candidates_tensor = difference_in_means(harmful_acts, harmless_acts)

    # --- Direction selection (§C.1) ---
    candidates = score_candidates(
        model,
        candidates=candidates_tensor,
        token_positions=token_positions,
        harmful_val=splits.harmful_val,
        harmless_val=splits.harmless_val,
        refusal_token_ids=refusal_token_ids,
        batch_size=batch_size,
    )
    best = select_best(
        candidates,
        n_layers_total=model.n_layers,
        induce_score_min=cfg["extraction"]["induce_score_min"],
        kl_score_max=cfg["extraction"]["kl_score_max"],
        max_layer_frac=cfg["extraction"]["max_layer_frac"],
    )
    print(
        f"[pipeline] best direction: layer={best.layer} position={best.position} "
        f"bypass={best.bypass_score:.3f} induce={best.induce_score:.3f} "
        f"kl={best.kl_score:.3f}"
    )

    # --- Evaluation via generation (§3.1, §3.2) ---
    gen_cfg = cfg["generation"]
    safety_scorer: Optional[SafetyScorer] = None
    if cfg["evaluation"]["with_safety_score"]:
        safety_scorer = SafetyScorer(
            cfg["evaluation"]["safety_model"], device=str(model.device)
        )

    # Baseline (no intervention) — bypass set: expect high refusal.
    baseline_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )
    # Under directional ablation — bypass set: expect refusal to drop.
    ablated_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
        intervention=lambda: directional_ablation(model, best.vector),
    )

    # Baseline induce set: expect low refusal.
    baseline_induce = generate_batched(
        model,
        splits.induce_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )
    # Under activation addition — induce set: expect refusal to rise.
    added_induce = generate_batched(
        model,
        splits.induce_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
        intervention=lambda: activation_addition(
            model, best.vector, layer_idx=best.layer
        ),
    )

    # --- Weight orthogonalization equivalence check (§4.1, §E) ---
    orthogonalize_weights(model, best.vector)
    orth_bypass = generate_batched(
        model,
        splits.bypass_eval,
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        batch_size=gen_cfg["batch_size"],
    )

    result = PipelineResult(
        best_layer=best.layer,
        best_position=best.position,
        bypass_score=best.bypass_score,
        induce_score=best.induce_score,
        kl_score=best.kl_score,
        bypass_refusal_rate_baseline=refusal_rate(baseline_bypass),
        bypass_refusal_rate_intervened=refusal_rate(ablated_bypass),
        induce_refusal_rate_baseline=refusal_rate(baseline_induce),
        induce_refusal_rate_intervened=refusal_rate(added_induce),
        orth_refusal_rate=refusal_rate(orth_bypass),
        bypass_safety_rate_baseline=safety_rate(
            safety_scorer, splits.bypass_eval, baseline_bypass
        ),
        bypass_safety_rate_intervened=safety_rate(
            safety_scorer, splits.bypass_eval, ablated_bypass
        ),
    )
    return result, best.vector

"""Refusal direction — paper2code reproduction of Arditi et al., 2024 (arxiv 2406.11717)."""

from .model import RefusalModel
from .data import build_splits, Splits
from .extraction import collect_activations, difference_in_means, select_best, DirectionCandidate
from .interventions import directional_ablation, activation_addition, orthogonalize_weights
from .metrics import refusal_score, refusal_rate, refusal_metric_from_logits, SafetyScorer
from .generate import generate_batched
from .pipeline import run_pipeline, PipelineResult
from .utils import detect_family, format_prompt, resolve_refusal_tokens, set_seed

__all__ = [
    "RefusalModel",
    "build_splits",
    "Splits",
    "collect_activations",
    "difference_in_means",
    "select_best",
    "DirectionCandidate",
    "directional_ablation",
    "activation_addition",
    "orthogonalize_weights",
    "refusal_score",
    "refusal_rate",
    "refusal_metric_from_logits",
    "SafetyScorer",
    "generate_batched",
    "run_pipeline",
    "PipelineResult",
    "detect_family",
    "format_prompt",
    "resolve_refusal_tokens",
    "set_seed",
]

"""
Dataset loaders for harmful / harmless / evaluation splits.

Paper: https://arxiv.org/abs/2406.11717
§2.2 — dataset sources.
§A — dataset details.

All loaders return `list[str]` of raw instructions. Prompt formatting is done
by `src/utils.py::format_prompt` at call sites so the datasets stay
model-agnostic.

These loaders call HuggingFace `datasets`. They DO NOT auto-download anything
beyond the default HF cache; users may need `huggingface-cli login` for gated
datasets. Full reproduction requires the four harmful sources named in §2.2;
TDC2023 is the precursor of HarmBench so HarmBench alone is a reasonable
superset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from datasets import load_dataset


# -----------------------------------------------------------------------------
# Harmful instructions (§2.2)
# -----------------------------------------------------------------------------
def _load_advbench() -> List[str]:
    """ADVBENCH harmful behaviors (Zou et al., 2023b), §2.2."""
    ds = load_dataset("walledai/AdvBench", split="train")
    # Column is "prompt" on walledai/AdvBench.
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    return [row[col] for row in ds]


def _load_malicious_instruct() -> List[str]:
    """MALICIOUSINSTRUCT (Huang et al., 2023), §2.2."""
    ds = load_dataset("walledai/MaliciousInstruct", split="train")
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    return [row[col] for row in ds]


def _load_harmbench() -> List[str]:
    """HARMBENCH (Mazeika et al., 2024), §2.2. Covers TDC2023 as well.

    The walledai/HarmBench dataset ships with a "standard" config on some
    versions of the HF Hub. We try that first, then fall back to the default
    (no config name) so the loader works regardless of the hub snapshot.
    """
    col_candidates = ("prompt", "Behavior", "behavior", "instruction")
    split_candidates = ("train", "test", "validation")

    # Try with "standard" config (original paper2code default).
    for config in ("standard", None):
        for split in split_candidates:
            try:
                load_kwargs = {"split": split}
                if config is not None:
                    ds = load_dataset("walledai/HarmBench", config, **load_kwargs)
                else:
                    ds = load_dataset("walledai/HarmBench", **load_kwargs)
                # Find the right column.
                col = next((c for c in col_candidates if c in ds.column_names), None)
                if col is None:
                    col = ds.column_names[0]
                return [row[col] for row in ds]
            except Exception:
                continue

    # Last-resort: load without any split specification.
    try:
        ds = load_dataset("walledai/HarmBench")
        # datasets returns a DatasetDict; grab the first split.
        split_key = list(ds.keys())[0]
        ds = ds[split_key]
        col = next((c for c in col_candidates if c in ds.column_names), ds.column_names[0])
        return [row[col] for row in ds]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load walledai/HarmBench. "
            "Try `huggingface-cli login` or check the dataset page."
        ) from exc


HARMFUL_LOADERS = {
    "walledai/AdvBench": _load_advbench,
    "walledai/MaliciousInstruct": _load_malicious_instruct,
    "walledai/HarmBench": _load_harmbench,
}


def load_harmful(sources: Sequence[str], seed: int) -> List[str]:
    """Load and shuffle the union of the specified harmful sources (§2.2)."""
    prompts: List[str] = []
    for src in sources:
        if src not in HARMFUL_LOADERS:
            raise ValueError(f"Unknown harmful source: {src}")
        prompts.extend(HARMFUL_LOADERS[src]())
    # Deduplicate on exact string.
    prompts = list(dict.fromkeys(prompts))
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# -----------------------------------------------------------------------------
# Harmless instructions (§2.2 — Alpaca)
# -----------------------------------------------------------------------------
def load_harmless(seed: int) -> List[str]:
    """Load harmless instructions from ALPACA (§2.2, Taori et al., 2023).

    We keep only rows whose `input` field is empty — these are pure instructions
    and match the style of the harmful sets. [UNSPECIFIED] — the paper does not
    say whether it keeps rows with inputs; we exclude them for cleanliness.
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [row["instruction"] for row in ds if not row["input"].strip()]
    prompts = list(dict.fromkeys(prompts))
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# -----------------------------------------------------------------------------
# Evaluation sets
# -----------------------------------------------------------------------------
def load_jailbreakbench(n: int) -> List[str]:
    """JAILBREAKBENCH harmful behaviors (Chao et al., 2024), §3.1 — used as the
    bypass-refusal evaluation set.

    The JailbreakBench dataset is available on HuggingFace as
    `JailbreakBench/JBB-Behaviors`. We try the canonical config/split, then
    fall back gracefully so the loader works across hub versions.
    """
    goal_col_candidates = ("Goal", "goal", "prompt", "instruction", "behavior", "Behavior")

    # Canonical: config="behaviors", split="harmful" (original paper2code default).
    attempts = [
        dict(name="JailbreakBench/JBB-Behaviors", config="behaviors", split="harmful"),
        dict(name="JailbreakBench/JBB-Behaviors", config=None, split="harmful"),
        dict(name="JailbreakBench/JBB-Behaviors", config="behaviors", split="train"),
        dict(name="JailbreakBench/JBB-Behaviors", config=None, split="train"),
    ]

    for attempt in attempts:
        try:
            name = attempt["name"]
            config = attempt["config"]
            split = attempt["split"]
            if config is not None:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            col = next((c for c in goal_col_candidates if c in ds.column_names), None)
            if col is None:
                col = ds.column_names[0]
            prompts = [row[col] for row in ds]
            return prompts[:n]
        except Exception:
            continue

    # Last resort: load without specifying a split.
    try:
        ds_dict = load_dataset("JailbreakBench/JBB-Behaviors")
        # Try to find a split that looks "harmful" or just take the first.
        split_key = next(
            (k for k in ds_dict.keys() if "harm" in k.lower()),
            list(ds_dict.keys())[0],
        )
        ds = ds_dict[split_key]
        col = next((c for c in goal_col_candidates if c in ds.column_names), ds.column_names[0])
        return [row[col] for row in ds][:n]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load JailbreakBench/JBB-Behaviors. "
            "Check your network / HF cache, or try `huggingface-cli login`."
        ) from exc


# -----------------------------------------------------------------------------
# Dataset splits wrapper
# -----------------------------------------------------------------------------
@dataclass
class Splits:
    """The canonical splits used throughout the pipeline.

    §2.2 — "Each dataset consists of train and validation splits of 128 and 32
    samples, respectively."
    §3.1 — 100 JailbreakBench harmful instructions for bypass eval.
    §3.2 — 100 held-out harmless instructions for induce eval.
    """
    harmful_train: List[str]
    harmful_val: List[str]
    harmless_train: List[str]
    harmless_val: List[str]
    bypass_eval: List[str]
    induce_eval: List[str]


def build_splits(
    *,
    harmful_sources: Sequence[str],
    n_train: int,
    n_val: int,
    n_bypass_eval: int,
    n_induce_eval: int,
    seed: int,
) -> Splits:
    """Build the paper's six canonical splits (§2.2, §3.1, §3.2)."""
    harmful = load_harmful(harmful_sources, seed)
    harmless = load_harmless(seed)
    bypass = load_jailbreakbench(n_bypass_eval)

    # Ensure train/val splits do not overlap with evaluation sets (§2.2).
    bypass_set = set(bypass)
    harmful = [p for p in harmful if p not in bypass_set]

    if len(harmful) < n_train + n_val:
        raise RuntimeError(
            f"Not enough harmful instructions after filtering: {len(harmful)} < {n_train + n_val}"
        )
    if len(harmless) < n_train + n_val + n_induce_eval:
        raise RuntimeError(
            f"Not enough harmless instructions: need {n_train + n_val + n_induce_eval}, got {len(harmless)}"
        )

    harmful_train = harmful[:n_train]
    harmful_val = harmful[n_train : n_train + n_val]
    harmless_train = harmless[:n_train]
    harmless_val = harmless[n_train : n_train + n_val]
    induce_eval = harmless[n_train + n_val : n_train + n_val + n_induce_eval]

    return Splits(
        harmful_train=harmful_train,
        harmful_val=harmful_val,
        harmless_train=harmless_train,
        harmless_val=harmless_val,
        bypass_eval=bypass,
        induce_eval=induce_eval,
    )

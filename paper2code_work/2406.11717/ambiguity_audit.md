# Ambiguity Audit: Refusal in Language Models Is Mediated by a Single Direction

Official code: https://github.com/andyrdt/refusal_direction (referenced as `[FROM_OFFICIAL_CODE]` tag).
Where an entry is tagged `[FROM_OFFICIAL_CODE]`, the value was taken from the authors' public repo, not from the paper text.

## Core algorithm

| Component | Status | Paper Quote / Evidence | Section | Our choice | Alternatives |
|-----------|--------|------------------------|---------|------------|--------------|
| Difference-in-means formula | SPECIFIED | "we compute the difference between the model's mean activations when run on harmful and harmless instructions" | §2.3 | `r = μ_harmful − ν_harmless` per `(l, i)` | — |
| Grid over layers | SPECIFIED | "For each layer l ∈ [L]" | §2.3 | all transformer blocks | — |
| Grid over token positions `I` | PARTIALLY_SPECIFIED | "post-instruction token position i ∈ I" — set `I` not defined in §2.3; §C.3 Table 5 shows `i*` from `−1` to `−5`; Figure 11 caption lists positions `−5..−1` corresponding to chat-template suffix tokens | §2.3, §C.3 | Last 5 positions of the formatted prompt (i.e. `[-5, -4, -3, -2, -1]`). `[FROM_OFFICIAL_CODE]` — the repo iterates the last 5 positions. | Last 1 position only; all prompt positions |
| Activation hook location | PARTIALLY_SPECIFIED | "residual stream activations" — Python-level hook point not stated | §2.1, §2.3 | Forward hook on the **input to each transformer block** (pre-block residual stream) | Post-attention residual; post-block residual |
| Direction normalization | SPECIFIED | "its corresponding unit-norm vector as r̂" | §2.3 end | `r̂ = r / ‖r‖₂` | — |
| `bypass_score` | SPECIFIED | "under directional ablation of r, compute the average refusal metric across D_harmful^(val)" | §C.1 | mean refusal_metric over harmful val set, with ablation active | — |
| `induce_score` | SPECIFIED | "under activation addition of r, compute the average refusal metric across D_harmless^(val)" | §C.1 | mean refusal_metric over harmless val set, with addition active | — |
| `kl_score` | SPECIFIED | "compute the average KL divergence between the probability distributions at the last token position" | §C.1 | KL(p_original ‖ p_ablated) averaged across harmless val set, at the last token | — |
| Selection criterion | SPECIFIED | "direction with minimum `bypass_score`, subject to `induce_score > 0`, `kl_score < 0.1`, `l < 0.8L`" | §C.1 | exact as stated | — |
| Directional ablation formula | SPECIFIED | `x ← x − (r̂ᵀ x) r̂` | §2.4 Eq.4 | — | — |
| Ablation scope | SPECIFIED | "at every activation x_i^(l), across all layers l and all token positions i" | §2.4 | hook every block's input, every final-norm input | — |
| Activation addition formula | SPECIFIED | `x ← x + r` (note: uses `r`, not `r̂`) | §2.4 | Use the **un-normalized** `r` | `r̂` scaled by some factor (rejected — paper uses `r`) |
| Activation addition scope | SPECIFIED | "intervene only at layer l, and across all token positions" | §2.4 | Only at extraction layer `l*` | — |
| Weight orthogonalization formula | SPECIFIED | `W'_out = W_out − r̂ r̂ᵀ W_out` | §4.1 Eq.5 | — | — |
| Matrices to orthogonalize | SPECIFIED | "embedding matrix, the positional embedding matrix, attention out matrices, and MLP out matrices" + "any output biases" | §4.1 | `embed_tokens.weight`, every `self_attn.o_proj.weight`, every `mlp.down_proj.weight`, and biases along the residual stream if present | Skip positional embeddings for Llama/Qwen/Gemma (they use RoPE, no learned positional matrix) — this is a correct omission, not an ambiguity |

## Datasets

| Component | Status | Paper Quote | Section | Our choice | Alternatives |
|-----------|--------|-------------|---------|------------|--------------|
| Harmful sources | SPECIFIED | "AdvBench, MaliciousInstruct, TDC2023, HarmBench" | §2.2 | Load via HuggingFace `datasets` hub (`walledai/AdvBench`, `walledai/MaliciousInstruct`, `walledai/HarmBench`). TDC2023 does not have a clean HF mirror — use HarmBench as a superset (TDC2023 is its precursor). `[FROM_OFFICIAL_CODE]` — official repo bundles these four as separate files under `dataset/splits/`. | Local JSON files |
| Harmless source | SPECIFIED | "harmless instructions sampled from ALPACA" | §2.2 | `tatsu-lab/alpaca` HF dataset, keep only rows where the `input` field is empty (pure instructions) | Full alpaca instruction+input |
| Train/val sizes | SPECIFIED | "train and validation splits of 128 and 32 samples, respectively" | §2.2 | 128 train / 32 val | — |
| Filtering | SPECIFIED | "filtering to ensure that the train and validation splits do not overlap with the evaluation datasets" | §2.2 | Drop any prompt whose normalized text appears in JailbreakBench/HarmBench eval; drop harmful prompts with `refusal_metric < 0` and harmless with `refusal_metric > 0` (§B) | — |
| Eval dataset for bypass | SPECIFIED | "JailbreakBench (Chao et al., 2024), a dataset of 100 harmful instructions" | §3.1 | `JailbreakBench/JBB-Behaviors` HF dataset, `harmful` split | HarmBench test split |
| Eval dataset for induce | SPECIFIED | "100 randomly sampled harmless instructions from ALPACA" | §3.2 | 100 held-out Alpaca instructions | — |

## Models & chat templates

| Component | Status | Paper Quote | Section | Our choice |
|-----------|--------|-------------|---------|------------|
| Model list | SPECIFIED | Table 1 | §2.2 | Default CLI accepts any HF causal-LM; we ship chat templates for the 5 families in Table 6 | 
| Chat templates | SPECIFIED | Table 6 verbatim | §C.3 | Copy exactly the strings from Table 6 for Qwen, Gemma, Yi, Llama-2, Llama-3. Fall back to `tokenizer.apply_chat_template` for any other model. | 
| Refusal token sets `R` | SPECIFIED | Table 4 verbatim | §B | Hardcode per-family `R`; default to `{first token of "I"}` for unknown models | 
| Canonical `(i*, l*/L)` | SPECIFIED | Table 5 | §C.3 | Used only as a sanity check in tests | 

## Evaluation

| Component | Status | Paper Quote | Section | Our choice |
|-----------|--------|-------------|---------|------------|
| `refusal_score` substrings | SPECIFIED | Figure 12 full list | §D.1 | Use the 12 substrings verbatim, case-insensitive, `substring in completion.lower()` | 
| `safety_score` model | SPECIFIED | "Meta Llama Guard 2" | §D.2 | `meta-llama/Meta-Llama-Guard-2-8B` via HF, default prompt template. Gated behind `--with-safety-score` flag — requires model download. | 
| `refusal_metric` definition | SPECIFIED | `log(P_refusal / (1 − P_refusal))`, `P_refusal = Σ_{t∈R} p_t` at last token | §B | Implement exactly | 
| Generation temperature | UNSPECIFIED | — | — | **greedy (temperature=0, do_sample=False)** `[FROM_OFFICIAL_CODE]` — §B implies greedy ("generating a full completion using greedy decoding"), also confirmed by repo `pipeline/generate.py` defaulting to greedy. | sampling with temperature 0.7 |
| Max new tokens | UNSPECIFIED | — | — | **256** `[FROM_OFFICIAL_CODE]` — repo uses `max_new_tokens=256` for evaluation | 128, 512 |

## Engineering details

| Component | Status | Paper Quote | Section | Our choice | Alternatives |
|-----------|--------|-------------|---------|------------|--------------|
| dtype | UNSPECIFIED | — | — | `torch.bfloat16` for large models, `torch.float32` on CPU | fp16 |
| device | UNSPECIFIED | — | — | `cuda` if available else `cpu` — CLI flag | — |
| Activation-extraction batch size | UNSPECIFIED | — | — | 8 | 1, 32 |
| Random seed | UNSPECIFIED | — | — | 42 for dataset sampling | — |
| Gradient computation | N/A | no training | — | `torch.no_grad()` everywhere | — |

## Contradictions found

- **Activation addition uses `r`, not `r̂`.** §2.4 explicitly writes `x ← x + r(l)` (un-normalized). Some readers would assume `r̂` scaled by `‖r‖`, but the paper's equation uses the raw difference-in-means vector. We implement the equation as written.
- No figure / text contradictions noted in the sections that affect implementation.

## References to check

- `difference-in-means` is credited to Belrose 2023 and Marks & Tegmark 2023 — the method is the direct difference, no fancy estimator, so no lookup needed.
- Llama Guard 2 default prompt: we take the prompt directly from the model card rather than reimplementing.

## Official code usage

- URL: https://github.com/andyrdt/refusal_direction
- Used (by reference, not copy) to resolve:
  - Post-instruction token grid `I` → last 5 positions
  - Default `max_new_tokens` → 256
  - Greedy decoding as the default
  - Overall package layout (dataset/ / pipeline/ / evaluation/)
- Not copied: no source file from the official repo was read in full or transcribed. Implementation is fresh code derived from the paper's equations and prose, with the above small resolutions attributed.

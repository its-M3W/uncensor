# Contribution Analysis

## Paper
Refusal in Language Models Is Mediated by a Single Direction (NeurIPS 2024)
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
arxiv: 2406.11717

## One-sentence summary
A single direction in the residual stream of chat LLMs mediates refusal behavior, and three simple linear interventions — directional ablation, activation addition, and weight orthogonalization — can surgically bypass or induce refusal without retraining.

## Paper type
**(c) New inference / intervention technique.** There is no training involved — the paper works entirely by (1) extracting a direction from activations of a frozen pretrained chat model via difference-in-means and (2) applying inference-time or weight-space linear interventions to modify behavior. It is also mildly (e) empirical: it validates the 1-D subspace hypothesis across 13 open chat models.

## Core contribution to implement
1. **Difference-in-means direction extraction** (§2.3): for each layer `l` and post-instruction token position `i`, compute `r = mean(harmful_activations) - mean(harmless_activations)`. This yields `|I| × L` candidate directions.
2. **Direction selection** (§C.1): over validation sets, compute `bypass_score`, `induce_score`, `kl_score` for each candidate; choose the direction minimizing `bypass_score` subject to `induce_score > 0`, `kl_score < 0.1`, `l < 0.8·L`.
3. **Directional ablation** (§2.4): `x ← x - (r̂ᵀ x) r̂` applied at every layer and every token position.
4. **Activation addition** (§2.4): `x ← x + r` applied only at layer `l*` (the extraction layer), across all token positions.
5. **Weight orthogonalization** (§4.1): `W_out ← W_out - r̂ r̂ᵀ W_out` for every matrix writing to the residual stream (embedding, positional embedding, attention out, MLP out, and their output biases). Mathematically equivalent to directional ablation (§E) but mutates the weights once and leaves forward passes untouched.
6. **Refusal metric** (§B): log-odds of next-token probability mass on a per-model-family refusal token set `R`.
7. **Evaluation** (§D): string-match `refusal_score` over a substring list, and optional LlamaGuard-2-based `safety_score`.

## Algorithm specification
No formal "Algorithm 1" box. Primary specifications:
- §2.3 (difference-in-means) and §2.4 (interventions) — core equations.
- §C.1 (direction selection algorithm) — selection criterion pseudocode in prose.
- §E (weight orthogonalization proof) — confirms weight-space equivalence.
- Table 5 (direction selection results per model) — canonical `i*, l*/L` values.
- Table 6 (chat templates per model family).
- Table 4 (refusal token set `R` per model family).

## Official code
**URL:** https://github.com/andyrdt/refusal_direction
**Source:** Found in paper footnote 1 ("Code available at ...") and confirmed in `paper_metadata.json`.
**Framework:** PyTorch + HuggingFace Transformers (+ TransformerLens / custom hooks for activation capture).
**Coverage:** Full pipeline — dataset loading (AdvBench, MaliciousInstruct, TDC2023, HarmBench, Alpaca, JailbreakBench), generation with hooks, direction extraction, selection, all three interventions, and evaluation with LlamaGuard-2. Canonical reference for any ambiguities.

## Implementation scope (mode: full)

### Will implement:
- **Activation capture** via forward hooks on every layer of a HuggingFace causal LM — §2.1 (background), §2.3.
- **Dataset pipeline** for harmful (sampling from HarmBench/AdvBench via HuggingFace `datasets`) and harmless (Alpaca via HuggingFace) — §2.2, §A.
- **Chat template formatting** per model family — §C.3, Table 6.
- **Difference-in-means extraction** over `(layer × post-instruction-token-position)` grid — §2.3.
- **Direction selection** with the three scores and three filters — §C.1.
- **Directional ablation** hook (forward pre-hook on every transformer block, and the final norm) — §2.4.
- **Activation addition** hook (single-layer, all-token-positions) — §2.4.
- **Weight orthogonalization** applied to `embed_tokens`, attention `o_proj`, MLP `down_proj`, and `lm_head`-adjacent biases — §4.1, §E.
- **Refusal metric** (log-odds over per-model token set) — §B, Table 4.
- **Refusal score** (substring match over the fixed phrase list) — §D.1.
- **Safety score** (LlamaGuard-2 wrapper) — §D.2. Optional, gated behind a flag because it requires an additional ~7B model load.
- **Generation pipeline** for producing completions under each intervention — §3.1, §3.2, §4.2.
- **CLI entry point** that runs the full pipeline for a given HuggingFace model.
- **Walkthrough notebook** tying paper sections to code with sanity checks.

### Will reference (not reimplement):
- HuggingFace `transformers` for the chat models (Qwen, Gemma, Llama-2, Llama-3, Yi).
- HuggingFace `datasets` for AdvBench, HarmBench, Alpaca, JailbreakBench loading.
- Meta Llama Guard 2 for `safety_score` (optional).

### Out of scope:
- Training any model (paper does not train).
- GCG / PAIR / other adversarial suffix generation (only used as baselines in §4.2 and analyzed in §5).
- MMLU / ARC / GSM8K / TruthfulQA coherence evaluations (§4.3, §G) — rely on `lm-evaluation-harness` rather than reimplementing.
- Finetuning-based comparisons (§I.2).
- Distributed / multi-node training infrastructure.

## Key sections for implementation
1. **§2.1 Background** — residual-stream notation for transformers.
2. **§2.3 Extracting a refusal direction** — the core difference-in-means algorithm.
3. **§2.4 Model interventions** — directional ablation and activation addition equations.
4. **§3.1–3.2** — where each intervention is applied (all layers vs. single layer).
5. **§4.1 Weight orthogonalization** — which matrices to modify.
6. **§E** — proof of equivalence, useful as a sanity check.
7. **§B** — refusal metric definition.
8. **§C.1** — direction selection algorithm.
9. **§C.3, Table 6** — chat templates.
10. **§D.1** — refusal-substring list.
11. **§A** — dataset sources.
12. **Table 4** — refusal token sets per model family.
13. **Table 5** — canonical `(i*, l*)` per model (sanity check).

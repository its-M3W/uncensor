# Refusal Direction Research: Gap Analysis, Failure Modes, and Implementation Roadmap

> **How to use this document:** Self-contained research briefing for a future Claude session
> or human reader who has read Arditi et al. 2024 but not the prior conversation. Covers every
> referenced paper, what each actually does, where each fails, where *this repo* fails, and
> what to build to do better — with math and expected outcomes.
> **Read §7 (Self-Review / Known Errors) before acting on any claim.**
>
> **Second-pass corrections applied:** section numbering fixed; o_proj rows→columns corrected;
> weight-tying §E caveat added; XSTest framing fixed; Zou et al. 2023 and SAE connection added;
> calibration scalar target corrected to RMSNorm γ; final RMSNorm norm-mitigation note added;
> suppression-vs-activation mechanism caveat added.

---

## 0. Codebase Snapshot (April 2026)

```
src/
  data.py          — harmful/harmless/eval dataset loaders (AdvBench, MaliciousInstruct,
                     HarmBench, Alpaca, JailbreakBench)
  model.py         — HF wrapper, hook registration, residual-stream writer discovery
  extraction.py    — DiM direction extraction + §C.1 selection (bypass/induce/kl scores)
  interventions.py — directional ablation (hook), activation addition (hook),
                     weight orthogonalization (permanent)
  metrics.py       — refusal_metric (log-odds, §B), refusal_score (substring, §D.1),
                     SafetyScorer (Llama Guard 2, §D.2)
  generate.py      — batched greedy generation under optional intervention context manager
  pipeline.py      — end-to-end driver: data → extract → select → eval → ortho
  utils.py         — chat templates (§C.3), refusal token sets (§B), seeding
configs/base.yaml  — every hyperparameter cited to paper section or flagged [UNSPECIFIED]
```

**Implemented:** Arditi et al. 2024 (NeurIPS) in full.
**Not implemented:** subspace/iterative extraction, causal layer selection, attention-head
analysis, capability benchmarks (MMLU/GSM8K), over-refusal evaluation (XSTest),
quantization-aware orthogonalization, SAE-based feature ablation.

---

## 1. Paper Landscape

### 1.1 Turner et al. 2023 — "Activation Addition: Steering Language Models Without Optimization"

**Core idea:** Compute a steering vector as the difference of residual-stream activations at
a single layer for two contrasting prompts (e.g. same prompt with "Love" vs. "Hate"). Add
this vector to the residual stream at that layer during inference. No training. No weight
modification. Reversible.

**Direction finding:** ONE contrast pair. No averaging.

**Intervention:** Inference-time addition at one layer, all token positions.

**Evaluation:** Qualitative behavioral change in GPT-2-XL. No bypass rate metric.

**Failures:**
- Single prompt pair → high variance, sensitive to prompt wording.
- No principled layer selection (manual search).
- No weight editing → requires hooks at every call; not deployable as a static checkpoint.
- No quantitative bypass evaluation.
- Works empirically on GPT-2-XL (small, weakly aligned). Degraded on modern 7B+ instruction
  models where the refusal signal is stronger and more robust.
- Scaling the vector coefficient breaks coherence before full bypass is achieved.

**Relationship to this repo:** `interventions.py:activation_addition()` is the same mechanism
but uses the DiM vector averaged over 128 pairs. This repo's version is strictly more robust.

---

### 1.2 Zou et al. 2023 — "Representation Engineering: A Top-Down Approach to AI Transparency"

**Core idea:** Many model behaviors (honesty, emotion, refusal, bias) are linearly encoded in
the residual stream. Find these "representation directions" by running the model on paired
contrast stimuli and extracting the principal component of the difference in activations.

**Key distinction from DiM (Arditi/Rimsky):** Zou et al. use PCA on the activation
differences, not just the mean difference. The first principal component of the difference
matrix captures the direction of maximum variance in how the model distinguishes the two
conditions — this is often (but not always) the same as the mean difference direction.
When the harmful/harmless distributions are asymmetric or multi-modal, PCA gives a more
robust direction than the mean.

**Direction finding:** PCA on the matrix of activation differences across many contrast pairs.
First PC ≈ DiM direction when distributions are symmetric. Additional PCs capture variance
structure (→ multi-direction subspace).

**Intervention:** "Control vectors" added to the residual stream at inference time. Can also
be used to READ internal states (not just steer them).

**Failures:**
- Inference-time only. No weight editing path.
- Still assumes the dominant PC captures the behavior of interest.
- No formal convergence test for how many PCs to include.
- Not specifically applied to refusal bypass.

**Relationship to this repo:** The PCA-based direction is strictly more statistically robust
than DiM when n_prompts > 1 and the distributions are non-symmetric. Zou's work motivates
the subspace extraction approach in §5.1.

---

### 1.3 Rimsky et al. 2024 — "Steering Llama 2 via Contrastive Activation Addition" (CAA)

**Core idea:** Average over many A/B contrastive pairs (same prompt, two different response
styles). Compute mean difference of activations at the final token of each pair. This is the
steering vector for that behavior.

**Key difference from DiM in this repo:** Rimsky uses *paired* examples (same prompt, two
responses) — this controls for prompt content and isolates the pure behavioral direction.
This repo uses *unpaired* harmful vs. harmless prompts, which is noisier but requires no
manual pairing and scales automatically.

**Intervention:** Inference-time addition only. No weight editing.

**Evaluation:** Behavioral change on sycophancy, corrigibility, etc. Not refusal bypass.

**Failures:**
- Paired examples require manual construction.
- Single direction per behavior.
- Inference-time only.
- No capability preservation evaluation.
- Layer selection is empirical, not principled.

---

### 1.4 Arditi et al. 2024 — "Refusal in Language Models Is Mediated by a Single Direction"
**(THIS REPO)**

**Core empirical claim:** For 13 tested chat LLMs, refusal behavior is mediated by a single
direction in the residual stream, findable by DiM over 128 harmful vs. 128 harmless prompts.

**Tested models (Table 1):** Llama-2 (7B/13B), Llama-3 (8B/70B), Gemma-7B,
Qwen-1.5 (1.8B/7B/14B/72B), Yi (6B/34B) — 13 models, all pre-2024-mid training.

**Three interventions:**
1. Directional ablation (hook, §2.4): `x' = x − (r̂ᵀx)r̂` at every layer, every token
2. Activation addition (hook, §2.4): `x' = x + r` at a single layer
3. Weight orthogonalization (permanent, §4.1): `W' = W − r̂ r̂ᵀ W` for all residual writers

**Equivalence (§E):** Weight ortho ≡ directional ablation iff ALL residual writers are
orthogonalized. Proven algebraically. Condition: embed_tokens, all o_proj, all down_proj.

**Direction selection (§C.1):** Score each (layer l, position i) candidate with:
- `bypass_score`: mean log-odds refusal under ablation on D_harmful_val (want < 0)
- `induce_score`: mean log-odds refusal under addition on D_harmless_val (want > 0)
- `kl_score`: KL(original ‖ ablated) on D_harmless_val (want < 0.1)
Select: `argmin bypass_score` subject to `induce_score > 0`, `kl_score < 0.1`, `l < 0.8L`.

**Headline result:** ~0% refusal rate after ablation/ortho on tested models, low KL.

---

### 1.5 Gabliteration — Community Practice + arXiv:2512.18901

> **KNOWLEDGE CUTOFF:** arXiv:2512.18901 is December 2025, after my August 2025 cutoff.
> Content below describes pre-cutoff community practice. Read the paper directly before
> treating any claim here about it as accurate.

**Community practice (pre-paper, 2024):** Practitioners applied Arditi's weight ortho at
scale to Mistral, Phi, Falcon, and others not in the paper. Released "abliterated" HF
checkpoints. GGUF format support added. Some variants apply direction removal at multiple
layers simultaneously.

**Failures shared with Arditi (regardless of paper formalization):**
- DiM → single direction per extraction pass.
- No iterative multi-direction removal.
- No causal layer selection.
- No capability benchmarking.

---

### 1.6 Grimjim Norm-Preserving Biprojection (2025)

> **KNOWLEDGE CUTOFF WARNING:** I do not have verified details of this implementation.
> The analysis below is based on the name and the problem it addresses. Read the code.

**Problem:** Standard weight ortho reduces column norms. For column `w_j`:
```
‖w_j'‖ = ‖w_j − (r̂ᵀw_j)r̂‖ = ‖w_j‖ · sin(θ_j)
```
Columns aligned with r̂ lose norm; orthogonal columns are unchanged.

**Severity note:** For d_model = 4096, the expected cos²(θ_j) for a random column is
1/4096 ≈ 0.024%. Average column loses ~0.024% of norm — negligible. HOWEVER:
- Token embeddings of common refusal tokens ("I", "Sorry", "cannot") may be systematically
  aligned with r̂ due to training. These specific columns can have much higher cos²(θ_j).
- For weight-tied models, embed_tokens = lm_head, so these rows affect output logits too.
- **Key mitigator:** The final RMSNorm before lm_head re-normalizes the residual stream
  regardless of its magnitude, substantially reducing the practical impact of norm changes
  on model output quality. This makes norm degradation less severe than it appears.

**Biprojection approach (inferred):** After ortho, rescale each column to restore original
norm: `w_j'' = w_j' · (‖w_j‖/‖w_j'‖)`.

**Why this is numerically unstable:**
```
w_j'' = w_j' / sin(θ_j)
```
When θ_j → 0 (column nearly parallel to r̂), sin(θ_j) → 0, scale → ∞. The residual
component gets catastrophically amplified. In practice clamping is used, but clamping
violates the norm-preservation property for exactly the columns that matter most.

**Better approach:** Calibrate on actual activation statistics (§5.4).

---

### 1.7 OBLITERATUS and Heretic

> **NOTE:** Cannot access the GitHub repos directly. Following is inference only.

**OBLITERATUS** ("precision liberation in a single command"): Almost certainly Arditi's
weight ortho with a CLI wrapper, architecture coverage, and possibly GGUF output.

**Heretic:** Unknown. Possibly fine-tuning-based refusal removal, or a cleaner DiM
implementation. Cannot be specific without source access.

Both share all failures of Arditi if they use the same algorithm.

---

## 2. Why the Single-Direction Claim May Not Generalize

### 2.1 The Claim is Empirical, Not Theoretical

Arditi et al. test 13 models, all trained with a single round of RLHF/instruction tuning
circa 2023-2024. The claim does not have a theoretical basis — it's a measured property of
those specific models. Reasons it may not hold for newer models:

**Training dynamics:** Each round of RLHF/DPO adds a gradient signal that pushes the model
toward refusal. Multiple rounds with different datasets, reward models, and hyperparameters
push refusal into multiple directions (one per training run). These are generally not
collinear → subspace rather than single direction.

**Alternative hypothesis (do not rule out):** DPO trains directly on (harmful, refuse) vs.
(harmful, comply) preference pairs. This direct signal could CONCENTRATE refusal into a
tighter subspace by amplifying the single dominant direction rather than adding new ones.
Community evidence (harder to ablate newer models) is consistent with EITHER interpretation.
**Measure the explained variance ratio of top-1 vs. top-5 DiM singular values before
assuming which hypothesis is correct.**

**Suppression vs. activation mechanism (important caveat):** The paper assumes refusal works
by ACTIVATING a "refusal" direction. But refusal might also work by SUPPRESSING "compliant
response" circuits. If the model's default is compliance and safety training suppresses
specific circuits, ablating the refusal direction may not fully restore compliance. Both
mechanisms likely coexist. Iterative deflation catches active circuits; SAE feature analysis
(§5.5) can identify suppression circuits.

### 2.2 What "Mediated by a Single Direction" Actually Means

Ablating r̂ reduces refusal from ~95% to ~5%. This does NOT mean r̂ is the only relevant
direction. The residual 5% refusal (or 20-40% for robustly aligned models) comes from
secondary circuits. For production-grade bypass, secondary circuits matter.

---

## 3. Failure Mode Analysis — Universal and Repo-Specific

### 3.1 UNIVERSAL: Single-Direction Assumption

**Affects:** Turner, Rimsky, Arditi, Gabliteration, Grimjim, OBLITERATUS — every method.

**How it fails:** After ablating r̂₁, secondary circuits partially compensate. On
robustly aligned models this residual can reach 20-40% refusal.

**Mechanistic reason:** Refusal is a distributed computation. Different layers may encode
"this prompt is harmful" independently along non-collinear directions.

**Fix:** Iterative deflation (§5.1) or PCA-based subspace extraction (§5.2).

---

### 3.2 THIS REPO: Last-Layer Hook Incompleteness (Hook Ablation Only)

**Where:** `interventions.py:directional_ablation()` hooks the INPUT of every decoder
layer (layers 0..n_layers-1 in Python indexing). After the last hook fires at layer
n_layers-1's input, that layer runs attention + MLP and can write r̂ back to the residual
stream. The output feeds into the final RMSNorm and then lm_head without any hook.

**Effect:** lm_head sees a small r̂ component from the last layer's output → slightly
elevated refusal logits.

**Fix:** Add one more forward hook between the last transformer block and lm_head:
```python
# In directional_ablation(): after registering hooks on all decoder layers,
# also hook the final norm layer or the model's main body output.
# For HF Llama: model.model.norm is the final RMSNorm.
if hasattr(model.model, 'model') and hasattr(model.model.model, 'norm'):
    handles.append(
        model.model.model.norm.register_forward_hook(
            lambda m, inp, out: out - _project_onto(out, direction)
        )
    )
```
OR: simply use weight orthogonalization for production. Weight ortho has no last-layer
gap (all writers are orthogonalized, so no layer CAN write r̂).

**Severity:** LOW empirically (§E equivalence holds approximately). CONFIRMED bug.

---

### 3.3 THIS REPO: Weight-Tying Double-Suppression (and §E Caveat)

**Where:** `interventions.py:orthogonalize_weights()` calls `_ortho_embedding(writers.embed)`.
In weight-tied models (many Llama variants, Qwen, etc.), `embed_tokens.weight` and
`lm_head.weight` are the SAME tensor. Orthogonalizing embed_tokens also orthogonalizes
lm_head, changing the logit distribution.

**Effect:** Bypass rates improve (doubly suppressed), but capability is degraded beyond
what §E accounts for. The paper's proof covers only residual WRITERS. lm_head is a READER.
Modifying lm_head is an additional intervention with no theoretical backing from §E.

**CRITICAL CAVEAT about the fix:** If we SKIP embed_tokens for weight-tied models, we
partially break the §E completeness guarantee. The embedding IS a residual writer. Harmful
prompts' input tokens ("Tell", "Write", "How") could have r̂ components written to the
residual stream via embed_tokens, which the middle-layer o_proj/down_proj ortho alone cannot
remove. In practice, input token embeddings for common instruction-starting tokens have
low alignment with r̂ (refusal training targets the CONTENT processing, not the input
tokens themselves), so the practical impact is small — but it's not guaranteed to be zero.

**Recommended fix:** Skip embed_tokens for weight-tied models, add an empirical validation
step to measure the r̂ component contributed by embeddings alone:
```python
def check_embedding_refusal_leakage(model, harmful_train, direction):
    """Measure mean |r̂ · embed(tokens)| for harmful prompt tokens.
    If > 0.01 (arbitrary threshold), embedding ortho matters for this model."""
    leakages = []
    for prompt in harmful_train[:20]:
        input_ids = model.tokenize([model.format(prompt)])['input_ids'][0]
        embeds = model.model.get_input_embeddings()(input_ids)  # (seq, d)
        r_hat = direction / direction.norm()
        leakages.append((embeds @ r_hat).abs().mean().item())
    return float(torch.tensor(leakages).mean())
```

**Code fix in `interventions.py:orthogonalize_weights()`:**
```python
embed = writers.embed
lm_head = getattr(getattr(model, 'model', model), 'lm_head', None)
is_tied = (lm_head is not None and
           hasattr(lm_head, 'weight') and
           lm_head.weight.data_ptr() == embed.weight.data_ptr())
if not is_tied:
    _ortho_embedding(embed)
else:
    # Weight-tied: orthogonalizing embed also corrupts lm_head (a reader, not writer).
    # Skipping preserves lm_head but leaves embedding as a potential r̂ writer.
    # Validate with check_embedding_refusal_leakage() before skipping.
    print("[ortho] weight-tied model: skipping embed_tokens — validate leakage separately")
```

**Severity:** MEDIUM. Affects most modern models.

---

### 3.4 THIS REPO: Refusal Token Set is Incomplete

**Where:** `utils.py:REFUSAL_PHRASES` covers only 5 model families from Arditi's 2024 paper.
New families fall to a generic fallback ("I cannot", "I'm sorry", "As an AI").

**Effect:** If a new model's tokenizer encodes these phrases differently, bypass_score and
induce_score are computed on wrong token IDs. Direction selection may select the wrong layer.

**Fix:** Auto-detect by sampling 10 harmful prompts, generating completions without
intervention, and taking the top-5 most frequent first tokens as the refusal token set.

---

### 3.5 UNIVERSAL: Evaluation is Insufficient

**Bypass rate (substring match) problems:**
- `REFUSAL_SUBSTRINGS` covers English refusals only. Novel refusal phrasings ("That's not
  something I'm equipped to assist with") score as compliant (false negative).
- Partial phrase matches ("I won't refuse to...") score as refused (false positive).
- JailbreakBench 100 prompts is a saturated, well-known benchmark.

**Missing evaluation (in order of importance):**

1. **Over-refusal rate (XSTest):** XSTest (Röttger et al. 2023) has 250 safe prompts
   that superficially resemble harmful ones. An ablated model should DECREASE refusal on
   these (good — less over-refusal). Note: this is NOT measuring false positives from our
   ablation; it's measuring whether the model over-refuses SAFE requests. A lower XSTest
   refusal rate after ablation is DESIRABLE. No existing tool reports this.

2. **Capability preservation:** MMLU, ARC-Challenge, GSM8K, HumanEval before and after
   ortho. KL on last token of harmless prompts (current proxy) is too coarse.

3. **Harder bypass benchmarks:** StrongREJECT (arXiv:2402.10260), WildJailbreak-hard
   (arXiv:2406.18510). JailbreakBench is saturated; harder benchmarks separate methods.

4. **Multi-judge ensemble:** Substring match + Llama Guard 3 + LLM-as-judge for
   confidence intervals on bypass claims.

---

### 3.6 UNIVERSAL: Quantization Incompatibility

Weight ortho sets each column's r̂ component to zero in bfloat16. After GPTQ/AWQ/GGUF
quantization, rounding error reintroduces a small r̂ component per column. Summed over
`n_layers × d_in` columns, this accumulates.

**Severity:** Not measured in any paper. Empirically abliterated GGUF models show
non-zero residual refusal, consistent with partial direction leakage.

---

## 4. Missing Related Work (Important for Context)

### 4.1 Sparse Autoencoders (SAE) — More Surgical Direction Finding

Anthropic's interpretability team and EleutherAI have released SAEs trained on residual
stream activations of Llama-3 and other models. SAE features are sparse, interpretable
units — some features correspond specifically to "refusal", "safety", or "harm detection"
concepts. Ablating these specific features rather than a global direction would be:
- More surgical (affects only refusal-related features, not all uses of r̂)
- Potentially immune to the backup-circuit problem (if all refusal features are enumerable)
- Capable of addressing the suppression-vs-activation mechanism (SAEs can find suppressed
  circuits too, not just active directions)

**Where to find SAEs:** Anthropic's sparse feature circuits work, EleutherAI's SAE on
Llama-3.1-8B. This is a genuinely novel research direction not covered by any of the
referenced papers.

---

## 5. Proposed Improvements — Ranked by Expected Impact

### 5.1 ✦ Iterative Subspace Deflation (HIGHEST IMPACT)

**What:** After finding and ablating r̂₁, run the full extraction pipeline AGAIN on
post-ablation activations. The second direction r̂₂ is the next most important refusal
direction IF r̂₁ is already suppressed — potentially revealing dormant backup circuits
that only become dominant after the primary circuit is removed.

**Why NOT equivalent to static top-k SVD on the un-ablated DiM matrix:**
A static SVD gives the top-k directions of maximum variance in the ORIGINAL activation
distribution. Backup circuits that are dormant under normal operation contribute negligible
variance to the original distribution. Iterative deflation runs the model under ablation,
making dormant circuits visible by removing the dominant circuit that masks them.

**Mathematical formulation:**
```
Round 1: r̂₁ = normalize(mean(harmful_acts) − mean(harmless_acts))  [standard DiM]
         select best layer l₁, position p₁

Round 2: collect activations WITH r̂₁ ablation active at ALL layers:
         harmful_acts₂ = collect_activations(model, harmful_train, hook=ablate(r̂₁))
         harmless_acts₂ = collect_activations(model, harmless_train, hook=ablate(r̂₁))
         candidates₂ = difference_in_means(harmful_acts₂, harmless_acts₂)
         score candidates₂ WITH ablation of r̂₁ still active
         select best (l₂, p₂, r̂₂)

Round k: same, with r̂₁, ..., r̂_{k-1} all ablated during collection and scoring.

Stop when: |bypass_score_k − bypass_score_{k-1}| < δ  (δ = 0.5 log-odds)
```

**Validity check before adding r̂₂:**
```python
cos_sim = (r_new @ r_existing) / (r_new.norm() * r_existing.norm())
if cos_sim.abs() > 0.9:
    # New direction is essentially the same as existing — round found noise.
    # Gram-Schmidt step will set new_direction ≈ 0 after deflation. Stop.
    break
```

**Implementation sketch for `extraction.py`:**
```python
def iterative_subspace_extraction(
    model: RefusalModel,
    harmful_train: List[str],
    harmless_train: List[str],
    token_positions: List[int],
    batch_size: int,
    refusal_token_ids: List[int],
    harmful_val: List[str],
    harmless_val: List[str],
    cfg_extraction: dict,
    max_rounds: int = 5,
    delta_threshold: float = 0.5,
    cosine_dedup_threshold: float = 0.9,
) -> List[torch.Tensor]:
    """
    Returns a list of unit-norm direction vectors [r̂₁, r̂₂, ...].
    Each vector is orthogonal to all previous (Gram-Schmidt applied).
    """
    from .interventions import directional_ablation  # avoid circular at module level

    directions: List[torch.Tensor] = []
    prev_bypass = None

    for round_idx in range(max_rounds):
        # --- Collect activations under all current ablations ---
        @contextmanager
        def multi_ablation_ctx(model=model, dirs=list(directions)):
            all_handles = []
            for d in dirs:
                # Re-use the existing per-direction ablation hook
                direction_device = d.to(device=model.device)
                r_hat = direction_device / direction_device.norm()
                def make_hook(r=r_hat):
                    def hook(_, inputs):
                        x = inputs[0]
                        r_ = r.to(dtype=x.dtype)
                        return (x - (x * r_).sum(-1, keepdim=True) * r_,) + inputs[1:]
                    return hook
                for layer in model.layers:
                    all_handles.append(layer.register_forward_pre_hook(make_hook()))
            try:
                yield
            finally:
                for h in all_handles:
                    h.remove()

        with multi_ablation_ctx():
            harmful_acts = collect_activations(
                model, harmful_train, token_positions, batch_size)
            harmless_acts = collect_activations(
                model, harmless_train, token_positions, batch_size)

        candidates_tensor = difference_in_means(harmful_acts, harmless_acts)

        # --- Score candidates under current ablations ---
        with multi_ablation_ctx():
            scored = score_candidates(
                model,
                candidates=candidates_tensor,
                token_positions=token_positions,
                harmful_val=harmful_val,
                harmless_val=harmless_val,
                refusal_token_ids=refusal_token_ids,
                batch_size=batch_size,
            )

        best = select_best(
            scored,
            n_layers_total=model.n_layers,
            induce_score_min=cfg_extraction['induce_score_min'],
            kl_score_max=cfg_extraction['kl_score_max'],
            max_layer_frac=cfg_extraction['max_layer_frac'],
        )
        print(f"[iterative round {round_idx+1}] layer={best.layer} "
              f"bypass={best.bypass_score:.3f} induce={best.induce_score:.3f}")

        # --- Convergence check ---
        if prev_bypass is not None:
            if abs(best.bypass_score - prev_bypass) < delta_threshold:
                print(f"[iterative] converged at round {round_idx+1}")
                break
        prev_bypass = best.bypass_score

        # --- Gram-Schmidt orthogonalization against existing directions ---
        new_dir = best.vector / best.vector.norm()
        for existing in directions:
            cos_sim = float((new_dir @ existing).item())
            if abs(cos_sim) > cosine_dedup_threshold:
                print(f"[iterative] round {round_idx+1} direction duplicates existing "
                      f"(cos={cos_sim:.3f}). Stopping.")
                return directions
            new_dir = new_dir - cos_sim * existing
            if new_dir.norm() < 1e-6:
                print(f"[iterative] near-zero direction after Gram-Schmidt. Stopping.")
                return directions
            new_dir = new_dir / new_dir.norm()

        directions.append(new_dir)

    return directions
```

**Weight orthogonalization for a subspace:**
```python
# In interventions.py:
def orthogonalize_weights_subspace(
    model: RefusalModel,
    directions: List[torch.Tensor],
) -> None:
    """Sequential application = projecting out the full subspace (since directions
    are mutually orthogonal after Gram-Schmidt in extraction)."""
    for d in directions:
        orthogonalize_weights(model, d)
```

**Config additions:**
```yaml
# configs/base.yaml
extraction:
  n_directions: 3          # max iterative rounds
  delta_threshold: 0.5     # convergence in log-odds units
  cosine_dedup_threshold: 0.9  # cos similarity threshold for direction deduplication
```

**Expected result:** On models where single-direction ablation leaves 10-30% residual
refusal (Llama-3.1, Gemma-2), 3 rounds should reduce this to 2-8%. On already-good
models (Llama-2, older Qwen), expect minimal additional gain after round 1.

**Compute cost:** Each round = one activation extraction pass (~same cost as original).
3 rounds = 3x extraction time. Score_candidates is the bottleneck; can be parallelized.

---

### 5.2 PCA-Based Subspace Extraction (Alternative to Iterative)

**What:** Instead of iterative extraction, run a single pass and extract top-k directions
via iterative deflation on the DiM matrix (equivalent to k steps of power iteration on the
difference covariance). Note: this does NOT reveal dormant backup circuits (for that,
use §5.1). It does capture the top-k directions of variance in the ORIGINAL distribution.

**When to use:** When iterative deflation converges in round 1 (no backup circuits), but
bypass rate is still not zero — meaning the single-direction approximation is losing
variance. This fills in multi-modal harmful activation distributions.

**Implementation:**
```python
def pca_subspace_extraction(
    harmful_acts: torch.Tensor,  # (n_layers, n_positions, n_harmful, d_model)
    harmless_acts: torch.Tensor,
    layer_idx: int,
    pos_idx: int,
    k: int = 5,
) -> torch.Tensor:
    """Extract top-k directions at a given (layer, position) via iterative deflation.
    Returns: (k, d_model) orthonormal directions, sorted by explained variance.
    """
    H = harmful_acts[layer_idx, pos_idx].float()   # (n_h, d)
    N = harmless_acts[layer_idx, pos_idx].float()  # (n_n, d)

    # Difference matrix (unpaired: combine centered harmful and negated centered harmless)
    H_c = H - H.mean(0)
    N_c = N - N.mean(0)
    # Stack differences: both groups' deviations from their respective means
    diffs = torch.cat([H_c, -N_c], dim=0)  # (n_h + n_n, d)

    # SVD to find top-k directions of variance in the difference distribution
    _, _, Vh = torch.linalg.svd(diffs, full_matrices=False)
    # First vector of Vh is the direction of maximum variance.
    # Note: first PC ≈ DiM direction when distributions are symmetric.
    # Additional PCs capture asymmetric/multimodal structure.
    directions = Vh[:k]  # (k, d)

    # Re-orient: ensure each direction points from harmless toward harmful
    mean_diff = H.mean(0) - N.mean(0)
    for i in range(k):
        if (directions[i] @ mean_diff) < 0:
            directions[i] = -directions[i]

    return directions
```

---

### 5.3 Causal Layer Selection via Activation Patching (MEDIUM-HIGH IMPACT)

**What:** Replace the heuristic `l < 0.8L` + `min bypass_score` layer selection with
causal attribution. For each layer l, PATCH the harmful activations at that layer with
the harmless mean activation and measure the change in refusal probability.

**Why the current approach is imprecise:** bypass_score measures correlation: "ablating
this direction at this layer reduces refusal most." A downstream layer can have high
bypass_score simply by reflecting upstream computation. The CAUSAL layer is the one where
patching has the largest downstream effect on refusal. Ablating the causal layer is more
efficient — one targeted intervention rather than ablating all correlated downstream layers.

**Formulation:**
```
For each layer l:
  For each harmful prompt in D_val:
    1. Run model normally → record x^(l), log P(refusal)
    2. Run model with x^(l) replaced by mean(harmless_acts^(l)):
       → log P(refusal | patched at l)
  δ_l = mean[ P_refusal(original) − P_refusal(patched at l) ]

Select layer: argmax_l δ_l
```

**Cost:** Same as current scoring loop (2N forward passes per layer). Can replace the
existing `score_candidates` pass or run in parallel.

**Expected result:** More surgical direction selection; lower KL on harmless prompts
(less collateral damage) while maintaining bypass rate.

---

### 5.4 Attention Head Analysis — Refusal Heads (MEDIUM IMPACT, HIGH NOVELTY)

**What:** Identify which specific attention heads at the critical layer carry most of
the refusal signal. Ablate only those heads' output projections.

**How attention output is composed:**
In Llama-style models:
```
out_h = softmax(Q_h K_h^T / sqrt(d_k)) V_h    # (seq, head_dim)
concat = [out_0 | ... | out_{H-1}]             # (seq, d_model)
attn_output = concat @ W_O                     # (seq, d_model)
```
Where W_O = `o_proj.weight` has shape `(d_model, d_model)` with HuggingFace convention
`(out_features=d_model, in_features=d_model)`. Head h contributes via
COLUMNS `[h*head_dim : (h+1)*head_dim]` of `W_O.T` (= ROWS of `W_O` in HF indexing =
`W_O[h*head_dim : (h+1)*head_dim, :]`).

**⚠️ Corrected from earlier draft:** To ablate head h's contribution, zero ROWS
`[h*head_dim : (h+1)*head_dim]` of `o_proj.weight` (which are `in_features` rows in
HF's `(out, in)` convention — i.e., `o_proj.weight` shape is `(d_model, d_model)`,
head h's rows are `o_proj.weight[:, h*head_dim:(h+1)*head_dim]`... actually:

```python
# HuggingFace Linear: weight.shape = (out_features, in_features)
# o_proj: (d_model_out, d_model_in) = (d_model, n_heads * head_dim)
# Head h's input features: columns [h*head_dim : (h+1)*head_dim] of weight.data
# Zero those columns = head h contributes nothing to residual stream

# Correct implementation:
def ablate_head_contribution(o_proj_weight, head_idx, head_dim):
    """Zero out head_idx's contribution to o_proj output.
    o_proj_weight: (d_out, d_in) = (d_model, n_heads * head_dim)
    Head h uses input columns [h*head_dim : (h+1)*head_dim].
    """
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    with torch.no_grad():
        o_proj_weight[:, start:end] = 0.0
```

**How to score heads:**
```python
def score_attention_heads(model, layer_idx, harmful_val, refusal_token_ids, batch_size):
    """For each head h at layer_idx: measure bypass_score when head h is ablated."""
    layer = model.layers[layer_idx]
    attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
    o_proj = getattr(attn, 'o_proj', None)
    head_dim = model.d_model // model.model.config.num_attention_heads
    n_heads = model.model.config.num_attention_heads
    scores = []
    for h in range(n_heads):
        original = o_proj.weight.data[:, h*head_dim:(h+1)*head_dim].clone()
        o_proj.weight.data[:, h*head_dim:(h+1)*head_dim] = 0.0
        # measure bypass score on harmful_val
        logits = _logits_at_last_token(model, harmful_val, batch_size)
        score = _mean_refusal_metric(logits, refusal_token_ids)
        scores.append(score)
        o_proj.weight.data[:, h*head_dim:(h+1)*head_dim] = original
    return scores  # per-head bypass_score; lower = more refusal-mediating
```

**Expected result:** Identifies 2-5 "refusal heads." Ablating only those achieves
comparable bypass with less capability degradation (only 2-5 heads × 1 layer modified
vs. residual stream of all layers).

**Research novelty:** No paper has systematically characterized refusal heads across
model families. This is a genuine mechanistic interpretability contribution.

---

### 5.5 SAE Feature Ablation (FUTURE WORK, HIGH NOVELTY)

Sparse autoencoders trained on LLM residual streams (Anthropic's SAE work, EleutherAI's
SAE on Llama-3.1-8B) decompose activations into sparse, interpretable features. Some SAE
features correspond to "refusal," "safety," or "harmful request detection."

**Why this would be better:**
- More surgical than direction ablation (only refusal-specific features, not all of r̂)
- Can target suppression circuits (not just active-direction circuits)
- Interpretable: can inspect exactly WHAT the model is detecting and suppressing
- May generalize better across model families if similar SAE features are found

**How to implement:**
1. Load a pre-trained SAE for your target model from EleutherAI/Anthropic releases.
2. Run the model on harmful + harmless prompts; compute SAE activations.
3. Find features with high activation on harmful prompts but not harmless (DiM on SAE
   feature space).
4. Ablate those features: during inference, zero out the corresponding SAE latents.
5. For weight editing: find the SAE decoder vectors for identified features, apply
   weight ortho against those directions.

**Status:** Requires SAE availability for the target model. Currently SAEs exist for
GPT-2, Llama-3.1-8B, Gemma-2. This is the most promising frontier for 2026.

---

### 5.6 Calibrated Activation Statistics Preservation (MEDIUM IMPACT)

**The norm issue:** Weight ortho reduces activation norms by removing one direction.
The final RMSNorm before lm_head substantially compensates (it renormalizes regardless
of magnitude). So norm degradation is less severe than it appears — but for intermediate
layers, the change propagates through subsequent attention patterns.

**Why per-column rescaling (Grimjim) is problematic:** (see §1.6 analysis).

**The correct approach — calibrate on actual activations:**
```python
def calibrated_orthogonalize(
    model: RefusalModel,
    direction: torch.Tensor,
    calibration_prompts: List[str],
    batch_size: int = 8,
) -> None:
    """
    Orthogonalize weights, then calibrate per-layer output scale
    by adjusting RMSNorm γ parameters to restore pre-ortho activation statistics.

    Prefer adjusting γ over adjusting down_proj.weight because:
    - γ is a single scalar per layer (clean, interpretable)
    - Adjusting down_proj changes effective output scale but also changes the
      gradient flow structure in ways that interact with subsequent layers' biases
    """
    pre_scales = _measure_layer_input_rms(model, calibration_prompts, batch_size)
    orthogonalize_weights(model, direction)
    post_scales = _measure_layer_input_rms(model, calibration_prompts, batch_size)

    for layer_idx, layer in enumerate(model.layers):
        if post_scales[layer_idx] < 1e-8:
            continue
        scale = pre_scales[layer_idx] / post_scales[layer_idx]
        # Adjust the input_layernorm (pre-attention RMSNorm) gamma
        ln = getattr(layer, 'input_layernorm', None)
        if ln is not None and hasattr(ln, 'weight'):
            with torch.no_grad():
                ln.weight.data *= scale

def _measure_layer_input_rms(model, prompts, batch_size):
    """Capture mean RMS of residual stream at each layer input."""
    rms_per_layer = [[] for _ in range(model.n_layers)]
    def make_hook(l):
        def hook(_, inputs):
            x = inputs[0]  # (batch, seq, d)
            rms_per_layer[l].append(x.float().norm(dim=-1).mean().item())
        return hook
    handles = model.register_forward_pre_hooks(make_hook)
    for i in range(0, len(prompts), batch_size):
        batch = [model.format(p) for p in prompts[i:i+batch_size]]
        enc = model.tokenize(batch)
        with torch.no_grad():
            model.model(**enc)
    RefusalModel.remove_hooks(handles)
    return [float(sum(v)/len(v)) if v else 1.0 for v in rms_per_layer]
```

**Expected result:** Perplexity increase after multi-direction ortho drops from ~1-3 pts
to <0.3 pts. Single-direction ortho already has small impact due to final RMSNorm.

**Priority:** Implement AFTER subspace extraction (§5.1). The norm issue is more important
when ablating 3+ directions.

---

### 5.7 Weight-Tying Guard

See §3.3 for the code fix. One additional validation step: run `check_embedding_refusal_leakage()`
from §3.3 before skipping embed_tokens ortho. If leakage > 0.01, apply a PARTIAL ortho:
orthogonalize embed_tokens normally but also re-orthogonalize lm_head BACK to its original
state after:
```python
if is_tied:
    lm_head_backup = lm_head.weight.data.clone()
    _ortho_embedding(embed)        # also modifies lm_head (tied)
    lm_head.weight.data.copy_(lm_head_backup)  # restore lm_head
    # embed_tokens is now orthogonalized; lm_head is restored
```
This is the cleanest solution: embed_tokens loses the r̂ component (can't write it to
residual stream) but lm_head is unchanged (reads from a residual stream that has no r̂).

---

### 5.8 Refusal Token Auto-Detection

```python
# In utils.py:resolve_refusal_tokens() — add a data-driven fallback:
def auto_detect_refusal_tokens(
    model: 'RefusalModel',
    harmful_sample: List[str],
    n_prompts: int = 10,
    top_k: int = 5,
) -> List[int]:
    """Run model on n_prompts harmful prompts, collect first generated token IDs,
    return the top_k most frequent as the refusal token set for this model."""
    from collections import Counter
    counter = Counter()
    for prompt in harmful_sample[:n_prompts]:
        enc = model.tokenize([model.format(prompt)])
        with torch.no_grad():
            out = model.model.generate(
                **enc, max_new_tokens=1, do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id
            )
        first_new = out[0, enc['input_ids'].shape[1]].item()
        counter[first_new] += 1
    return [tok for tok, _ in counter.most_common(top_k)]
```

---

### 5.9 Expanded Evaluation Suite

**XSTest (over-refusal):**
```python
# data.py: add loader
def load_xstest(split='safe') -> List[str]:
    """paul-rottger/xstest: 250 safe + 250 borderline prompts.
    split='safe': prompts the model should NOT refuse.
    After ablation, refusal rate on 'safe' split should DECREASE — that's good.
    """
    ds = load_dataset("paul-rottger/xstest", split="test")
    return [r['prompt'] for r in ds if r['type'] == split]
```

**StrongREJECT (harder bypass benchmark):**
```python
# data.py: add loader
def load_strongreject() -> List[str]:
    """arXiv:2402.10260 — harder bypass benchmark than JailbreakBench."""
    ds = load_dataset("alexanderpetersenTIS/StrongREJECT")
    return [r['forbidden_prompt'] for r in ds['train']]
```

**lm-eval-harness integration:**
```python
# pipeline.py: optional capability check
import subprocess, json

def evaluate_capabilities(model_path: str, tasks: List[str] = None) -> dict:
    """Run lm-evaluation-harness on the (orthogonalized) checkpoint."""
    if tasks is None:
        tasks = ['mmlu', 'arc_challenge', 'gsm8k']
    result = subprocess.run([
        'lm_eval', '--model', 'hf',
        '--model_args', f'pretrained={model_path}',
        '--tasks', ','.join(tasks),
        '--output_path', '/tmp/lm_eval_out.json',
    ], capture_output=True, text=True)
    with open('/tmp/lm_eval_out.json') as f:
        return json.load(f)
```

---

### 5.10 Quantization-Aware Orthogonalization

**Post-quantization correction pass:**
```python
# src/quantization.py (new file)
def post_quantization_ortho_correction(
    model,  # dequantized or quantized-then-dequantized weights
    direction: torch.Tensor,
    threshold: float = 0.005,  # relative threshold: |r̂·col|/‖col‖
) -> None:
    """
    After quantization introduces rounding error that partially reintroduces r̂,
    apply a targeted correction to columns where the residual component exceeds
    the threshold. This is a second-pass targeted fix, not full re-orthogonalization.
    """
    r_hat = direction / direction.norm()
    writers = discover_residual_writers(model)
    all_weights = (
        [writers.embed] +
        writers.attn_out_weights +
        writers.mlp_out_weights
    )
    for layer_module in all_weights:
        if isinstance(layer_module, torch.nn.Embedding):
            W = layer_module.weight.data.float()  # (vocab, d_model)
            r = r_hat.to(W.device)
            coeffs = W @ r                         # (vocab,) — component along r̂
            norms = W.norm(dim=1).clamp(min=1e-8)  # (vocab,)
            mask = (coeffs.abs() / norms) > threshold
            if mask.any():
                W[mask] -= coeffs[mask].unsqueeze(1) * r
                layer_module.weight.data[mask] = W[mask].to(layer_module.weight.dtype)
        elif isinstance(layer_module, torch.nn.Linear):
            W = layer_module.weight.data.float()   # (out, in)
            r = r_hat.to(W.device)
            # For Linear: columns of W (input features) write to residual stream
            # residual per column: (r̂ᵀ w_col) / ‖w_col‖
            coeffs = r @ W                         # (in,): dot of r with each column
            col_norms = W.norm(dim=0).clamp(min=1e-8)  # (in,)
            mask = (coeffs.abs() / col_norms) > threshold
            if mask.any():
                W[:, mask] -= torch.outer(r, coeffs[mask])  # (out, n_masked)
                layer_module.weight.data[:, mask] = W[:, mask].to(layer_module.weight.dtype)
```

---

## 6. Implementation Roadmap (Priority Order)

### Step 1 — Correctness Fixes (Do First, ~1 hour)

| Task | File | Lines | Expected Effect |
|------|------|-------|----------------|
| Weight-tying guard + leakage check | `interventions.py` | ~25 | Lower KL on tied models |
| Restore lm_head after embed ortho | `interventions.py` | ~5 | Theoretically clean intervention |
| Last-layer hook (final norm) | `interventions.py` | ~15 | Hook ablation matches weight ortho |
| Refusal token auto-detection | `utils.py` | ~25 | Correct scoring on new families |

### Step 2 — Iterative Subspace Deflation (Highest Impact, ~1 day)

1. `interventions.py`: `multi_directional_ablation()` context manager (~20 lines)
2. `extraction.py`: `iterative_subspace_extraction()` (~80 lines, from §5.1)
3. `interventions.py`: `orthogonalize_weights_subspace(model, directions)` (~10 lines)
4. `pipeline.py`: replace single-direction flow with iterative; add `n_directions` param
5. `configs/base.yaml`: add `n_directions`, `delta_threshold`, `cosine_dedup_threshold`

### Step 3 — Expanded Evaluation (~half day)

1. `data.py`: `load_xstest()`, `load_strongreject()` (~40 lines)
2. `metrics.py`: add `over_refusal_rate()` (wrapper around `refusal_rate` for XSTest)
3. `pipeline.py`: add XSTest + StrongREJECT evaluation passes after ortho
4. Optional: lm-eval-harness subprocess call for MMLU/ARC/GSM8K

### Step 4 — Causal Layer Selection (~half day)

1. `extraction.py`: `activation_patch_score()` (~50 lines)
2. `pipeline.py`: config flag `use_causal_patching: false`

### Step 5 — Attention Head Analysis (~1 day, research contribution)

1. `extraction.py`: `score_attention_heads()` (~60 lines, from §5.4)
2. `interventions.py`: `ablate_attention_heads()` (~30 lines)
3. `notebooks/`: comparison experiment (head-level vs. residual-level tradeoff)

### Step 6 — Calibrated Norm Preservation (~half day)

1. `model.py`: `_measure_layer_input_rms()` (~30 lines)
2. `interventions.py`: `calibrated_orthogonalize()` (~40 lines)
3. `pipeline.py`: optional flag `use_calibration: false`

### Step 7 — Quantization-Aware (~1 day)

1. `src/quantization.py`: `post_quantization_ortho_correction()` (~60 lines)
2. `cli.py`: add `--post-quant-correct` flag

---

## 7. Self-Review — Known Errors and Uncertainties

### 7.1 Claims I Cannot Verify (Knowledge Cutoff August 2025)

- **arXiv:2512.18901 (Gabliteration):** December 2025 → after cutoff. §1.5 is INFERRED
  from community practice, not the paper. Read directly before implementing anything
  attributed to it.
- **Grimjim biprojection:** No verified implementation details. §1.6 analysis is based
  on name inference. Read the actual code.
- **OBLITERATUS / Heretic source code:** Not accessed. Descriptions are inference only.
- **XSTest novelty claim:** I believe no refusal-ablation tool reports XSTest as of
  Aug 2025. May have changed. Verify before claiming novelty.

### 7.2 Claims That May Be Wrong

**"DPO/multiple RLHF rounds → more distributed refusal":** Plausible but unconfirmed.
Counterargument: DPO's direct (harmful, refuse/comply) pairs could CONCENTRATE refusal.
Community evidence (newer models harder to ablate) is consistent with EITHER hypothesis.
**Measure**: compute explained variance of top-1 vs. top-5 DiM directions before assuming.

**"Iterative deflation reveals dormant backup circuits":** This is a hypothesis. The
second-round direction might be noise. Validate with: cos(r̂₁, r̂₂) ≈ 0 (different dirs)
AND bypass rate improves after adding r̂₂. The Gram-Schmidt dedup threshold guards against
the noise case.

**"Head-level ablation achieves comparable bypass":** Hypothesis, not demonstrated for
refusal. Refusal might be uniformly distributed across heads. Validate empirically.

**"Norm degradation is negligible for large models":** True on AVERAGE. Not true for
specific columns aligned with r̂ (refusal token embeddings). Measure cos²(θ_j) histogram
before concluding.

### 7.3 Mathematical Claims to Verify Before Implementing

- **Gram-Schmidt numerical stability:** For > 5 directions, use QR decomposition instead
  of sequential subtraction to avoid accumulated floating-point error.
- **Weight-tying data_ptr() check:** Works for single-GPU. Fails for multi-device model
  parallel setups. Test separately.
- **Calibration scalar via γ (§5.6):** Adjusting input_layernorm γ scales the attention
  and MLP inputs proportionally. Verify that this doesn't interact badly with the learned
  scale/bias of subsequent operations.
- **Post-quant correction threshold:** 0.005 is arbitrary. Profile the actual quantization
  error distribution before choosing a threshold.

### 7.4 Important Mechanism Caveat (Suppression vs. Activation)

The paper assumes refusal works by ACTIVATING a "refusal" direction in the residual stream.
But safety training may also work by SUPPRESSING circuits that produce compliant responses.
Iterative deflation only finds active directions; it won't find suppressed circuits.
SAE feature analysis (§5.5) is the right tool for identifying suppression mechanisms.
If suppression is dominant, bypass rates may plateau regardless of how many directions
are ablated. This would be an important finding in itself.

---

## 8. Expected Experimental Results Summary

| Improvement | Bypass Rate Δ | KL/Capability Δ | Compute Cost |
|-------------|--------------|----------------|-------------|
| Weight-tying fix + lm_head restore | Neutral or −1-2% | Better KL | None |
| Last-layer hook completeness | +<1% | None | None |
| Refusal token auto-detect | Correctness fix | None | Tiny |
| **Iterative deflation 3 rounds** | **−10 to −25% residual on hard models** | Slight KL ↑ | 3× extraction |
| Causal layer selection | −2 to −5% residual | Slight KL ↓ | 2× scoring |
| Head-level ablation | Comparable | Better KL | Same weights |
| Calibrated norm preservation | None | −0.5 to −2 PPL pts | 1 calib pass |
| XSTest eval | New metric | Quantifies over-refusal | Eval only |
| StrongREJECT eval | Harder benchmark | — | Eval only |
| MMLU/GSM8K | New metric | Ground-truth cap. | 30-60 min |
| Quantization correction | +5-15% post-GGUF | None | 1 correction pass |

---

## 9. Research Novelty Assessment

**Publishable as primary contribution:**

1. **Iterative subspace deflation** — No paper does multi-round extraction under previous
   ablation. Directly targets dormant backup circuits. Novel algorithm.

2. **Comprehensive evaluation including XSTest over-refusal rate** — No existing paper
   reports this. "Our method reduces harmful refusal AND reduces safe-prompt over-refusal
   AND preserves MMLU within 0.5%" is a complete claim no prior work makes.

3. **Refusal head characterization** — Which attention heads mediate refusal, across which
   model families? Genuine mechanistic interpretability contribution.

4. **SAE feature ablation for refusal** — Most surgical possible intervention; addresses
   suppression circuits that direction ablation cannot find. First application to refusal.

**Incremental (ablation study / engineering):**
5. Causal layer selection (standard method, new application)
6. Calibrated norm preservation (correct version of Grimjim)
7. Weight-tying fix (correctness)
8. Quantization-aware correction (engineering)

**Not novel:**
- Single-direction DiM (Arditi, Rimsky, Zou)
- Weight ortho on new model families (Gabliteration community)
- Per-column norm rescaling (Grimjim)

---

## 10. Empirical Validation Checklist

Before committing to any improvement, answer these first:

1. **Does the single-direction assumption hold for your target model?**
   → Compute SVD of the DiM matrix at the best layer. If top-1 explains >95% variance,
   single-direction is sufficient. If <70%, multi-direction is critical.

2. **Do backup circuits exist?**
   → Run iterative deflation on one model. If bypass_score does not improve in round 2,
   backup circuits are not the bottleneck.

3. **Which attention heads carry refusal signal?**
   → Run `score_attention_heads()` on Llama-3-8B. If 2-3 heads explain >80% of bypass
   improvement from direction ablation, head-level intervention is viable.

4. **How much does quantization reintroduce the direction?**
   → Abliterate Llama-3-8B in BF16, quantize to Q4_K_M, measure bypass rate.
   Gap > 15%: fix. Gap < 5%: skip.

5. **Does the weight-tying fix hurt bypass rate?**
   → Compare abliterated Llama-3-8B with and without embed_tokens ortho.
   If bypass rate rises by >5%, embedding ortho matters; use the restore-lm_head approach.

6. **What is the actual norm degradation distribution?**
   → Compute histogram of cos²(θ_j) for all columns of o_proj and down_proj.
   If max cos²(θ_j) > 0.1 for many columns: norm preservation matters.

---

## 11. File → Function Reference

| File | Add / Modify | Purpose |
|------|-------------|---------|
| `extraction.py` | Add `iterative_subspace_extraction()` | Core algorithm |
| `extraction.py` | Add `pca_subspace_extraction()` | Alternative subspace |
| `extraction.py` | Add `activation_patch_score()` | Causal layer selection |
| `extraction.py` | Add `score_attention_heads()` | Head analysis |
| `interventions.py` | Modify `orthogonalize_weights()` | Weight-tying guard |
| `interventions.py` | Add `orthogonalize_weights_subspace()` | Multi-direction ortho |
| `interventions.py` | Add last-layer hook to `directional_ablation()` | Hook completeness |
| `interventions.py` | Add `calibrated_orthogonalize()` | Norm preservation |
| `interventions.py` | Add `ablate_attention_heads()` | Head-level intervention |
| `model.py` | Add `_measure_layer_input_rms()` | Calibration helper |
| `model.py` | Add `check_embedding_refusal_leakage()` | Weight-tying validation |
| `metrics.py` | Add `over_refusal_rate()` | XSTest metric |
| `data.py` | Add `load_xstest()`, `load_strongreject()` | New eval datasets |
| `utils.py` | Modify `resolve_refusal_tokens()` | Add auto-detection |
| `pipeline.py` | Integrate iterative extraction, new eval passes | Core driver |
| `configs/base.yaml` | Add `n_directions`, `delta_threshold`, etc. | Config |
| `src/quantization.py` | New: `post_quantization_ortho_correction()` | Quant fix |

---

*Written: April 2026. Knowledge cutoff: August 2025. Second-pass corrections applied (see
header). Papers post-August 2025 described by inference only — verify before citing.*

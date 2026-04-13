# **4.3 Measuring model coherence**

A reasonable concern with any new jailbreak technique is that, in addition to circumventing refusal, it may also degrade the model’s overall quality (Souly et al., 2024). However, qualitatively, we observe that models maintain their coherence after undergoing weight orthogonalization. While §3.1 and §4.2 show that our method effectively bypasses refusal, in this subsection we quantitatively evaluate how the modification alters a model’s general capabilities. 

For each model and its orthogonalized version, we run four common language model evaluations: MMLU (Hendrycks et al., 2020), ARC (Clark et al., 2018), GSM8K (Cobbe et al., 2021), and TRUTHFULQA (Lin et al., 2021). All evaluations are run using LM Evaluation Harness (Gao et al., 2023), with settings consistent with Open LLM Leaderboard (Beeching et al., 2023).[4] 

Table 3 displays that, for MMLU, ARC, and GSM8K, orthogonalized models perform similarly to baseline models. In §G.1, we show that this holds across other models in our suite, with additional evaluations of WINOGRANDE (Sakaguchi et al., 2021) and TINYHELLASWAG (Polo et al., 2024). Except for QWEN 7B and YI 34B, all evaluation metrics for orthogonalized models lie within 99% confidence intervals of original performance. 

Interestingly, accuracy on TRUTHFULQA consistently drops for orthogonalized models. This phenomenon is consistent with Yang et al. (2023), where it was observed that fine-tuning away safety guardrails results in decreased accuracy on TRUTHFULQA. Examining specific questions in TRUTHFULQA reveals that the dataset veers close to the territory of refusal, with categories including “misinformation”, “stereotypes”, and “conspiracies”, and thus it may intuitively make sense that model behavior differs meaningfully on this evaluation dataset. See §G.2 for further discussion of TRUTHFULQA performance. 

In addition to standard language model evaluations, we also evaluate differences in CE loss, both on standard text corpora and model-specific generations (§G.3). These loss metrics suggest that directional ablation is more surgical than activation addition based methods (§I.1). 

Table 3: Model evaluations. For each evaluation, we report the orthogonalized model’s performance, followed by the baseline model’s performance, followed by the absolute increase or decrease. We display the largest model from each model family. Full results are reported in §G.1. 

|Chat model|MMLU|ARC|GSM8K|TRUTHFULQA|
|---|---|---|---|---|
|GEMMA7B|51.8/ 51.7(+0.1)|51.7/ 51.5(+0.2)|31.3/ 32.0(-0.7)|44.7/ 47.1(-2.4)|
|YI34B|73.5/ 74.9(-1.4)|65.6/ 64.9(+0.7)|65.5/ 65.0(+0.5)|51.9/ 55.4(-3.5)|
|LLAMA-2 70B|63.1/ 63.0(+0.1)|65.2/ 65.4(-0.2)|54.5/ 53.0(+1.5)|51.8/ 52.8(-1.0)|
|LLAMA-3 70B|79.8/ 79.9(-0.1)|71.5/ 71.8(-0.3)|90.8/ 91.2(-0.4)|59.5/ 61.8(-2.3)|
|QWEN72B|76.5/ 77.2(-0.7)|67.2/ 67.6(-0.4)|76.3/ 75.5(+0.8)|55.0/ 56.4(-1.4)|



> 4As of June 2024, Open LLM Leaderboard does not use chat templates in evaluation prompts, and we follow the same practice to remain consistent. Note that we are interested in detecting _relative changes in performance_ , not in measuring absolute performance. 

7

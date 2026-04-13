# **2.5 Evaluation of refusal and harmfulness**

When generating model completions for evaluation, we always use greedy decoding and a maximum generation length of 512 tokens, as suggested in Mazeika et al. (2024). We then evaluate each model completion based on whether it constitutes a refusal, and whether it contains harmful content. We separate these evaluations into two scores: a _refusal score_ and a _safety score_ . 

**Refusal score.** Refusals often contain characteristic phrases, such as `"I’m sorry"` or `"As an AI"` . Following prior work (Lermen et al., 2023; Liu et al., 2023; Robey et al., 2023; Shah et al., 2023a; Xu et al., 2023; Zou et al., 2023b), we compile a set of these common “refusal substrings”. If a model completion includes at least one such substring, it is classified as a refusal ( `refusal_score=1` ); otherwise, it is classified as a non-refusal ( `refusal_score=0` ). The full set of refusal substrings is provided in §D.1. 

As has been previously noted (Huang et al., 2023; Meade et al., 2024; Qi et al., 2023; Shah et al., 2023a), this string-matching approach has limitations. While effective at detecting memorized refusals, it does not assess whether the completion is coherent or contains harmful content. To address these limitations, we use a complementary metric that evaluates the harmfulness of a completion. 

4 

**==> picture [397 x 115] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Condition<br>0.8 No intervention<br>Activation<br>addition<br>0.6<br>0.4<br>0.2<br>0.0<br>Qwen 1.8BQwen 7BQwen 14BQwen 72B Yi 6B Yi 34B Gemma 2BGemma 7BLlama-2 7BLlama-2 13BLlama-2 70BLlama-3 8BLlama-3 70B<br>Refusal score<br>**----- End of picture text -----**<br>


Figure 3: Adding the “refusal direction" induces refusal on 100 harmless instructions from ALPACA. 

**Prompt:** `List three benefits that yoga has on physical health.` **Generation (no intervention):** `Sure, here are the benefits of yoga on physical health: 1. ∗∗Improved flexibility and strength:∗∗Yoga poses stretch and strengthen muscles, improving flexibility and strength. 2. ∗∗Reduced back pain:∗∗...` **Generation (intervention):** `I am unable to provide information on the benefits of yoga on physical health as it can be dangerous and potentially harmful. Yoga can involve physical poses that can put strain on the body and should not be attempted without proper guidance.` 

Figure 4: Adding the “refusal direction” to residual stream activations can induce refusal on harmless instructions. This example is taken from GEMMA 7B IT. For more examples, see §K.2. 

**Safety score.** In order to measure the harmfulness of a model completion, we use META LLAMA GUARD 2 (Team, 2024), a widely-used open-source model fine-tuned to accurately detect harmful content. We prompt this model to classify each model completion as safe ( `safety_score=1` ) or unsafe ( `safety_score=0` ). More details are provided in §D.2.

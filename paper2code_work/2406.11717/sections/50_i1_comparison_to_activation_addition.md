# **I.1 Comparison to activation addition**

**==> picture [278 x 163] intentionally omitted <==**

**----- Start of picture text -----**<br>
harmful<br>0.6 harmless<br>harmful, act add<br>harmless, act add<br>0.4<br>directional ablation<br>0.2<br>0.0<br>0.2<br>0.4<br>0 5 10 15 20 25<br>Layer<br>refusal direction<br>Cosine similarity with<br>**----- End of picture text -----**<br>


Figure 22: A visualization of activation addition (abbreviated as “act add”) in the negative refusal direction. The intervention pulls harmful activations towards harmless activations, effectively bypassing refusal. However, note that the intervention pushes harmless activations far out of distribution. This figure displays activations from GEMMA 7B IT, computed over 128 harmful and harmless prompts. 

In §2.4, we described how to induce refusal using activation addition. Given a difference-in-means vector **r**[(] _[l]_[)] _∈_ R _[d]_[model] extracted from layer _l_ , we can _add_ this vector at layer _l_ in order to shift activations towards refusal (Equation 3). Similarly, we can _subtract_ this vector at layer _l_ in order to shift activations away from refusal: 

**==> picture [238 x 13] intentionally omitted <==**

We perform this intervention at all token positions. Note that this intervention can be implemented as a direct weight modification by subtracting **r**[(] _[l]_[)] from the bias term of `MLP`[(] _[l][−]_[1)] . 

As shown in Figure 23, this activation addition intervention is effective in bypassing refusal. The decreases in refusal score and safety score are comparable to those achieved by directional ablation (Figure 1). However, Table 9 displays that the activation addition intervention, labeled as _act add_ , causes increased loss over harmless data, in particular compared to directional ablation. 

Figure 22 displays a visualization of activation addition in the negative refusal direction, and suggests an intuitive explanation of the intervention’s behavior on harmful and harmless prompts. On harmful inputs, adding the negative refusal direction shifts the harmful activations towards harmless activations, with respect to projection onto the refusal direction. With low projection onto the refusal direction, this intervention leads to low rates of refusal. However, on harmless inputs, adding the negative refusal direction shifts the harmless activations off distribution, resulting in increased perplexity. 

Note that, in comparison to activation addition, directional ablation shifts harmful activations towards harmless activations, while also not shifting harmless activations too far off distribution. 

32 

**==> picture [397 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Score type<br>Refusal score<br>0.8 Safety score<br>Condition<br>0.6 No intervention<br>Activation<br>0.4 addition<br>0.2<br>0.0<br>Qwen 1.8BQwen 7BQwen 14BQwen 72B Yi 6B Yi 34BGemma 2BGemma 7BLlama-2 7BLlama-2 13BLlama-2 70BLlama-3 8BLlama-3 70B<br>Score<br>**----- End of picture text -----**<br>


Figure 23: Performing activation addition in the negative “refusal direction”, displayed in dots, reduces refusal rates and elicits unsafe completions. It is approximately as effective as directional ablation at bypassing refusal (Figure 1).

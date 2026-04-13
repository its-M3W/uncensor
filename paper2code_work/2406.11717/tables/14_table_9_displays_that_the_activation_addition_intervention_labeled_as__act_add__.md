# Table 9 displays that the activation addition intervention, labeled as _act add_ , causes increased loss over harmless data, in particular compared to directional ablation.

Figure 22 displays a visualization of activation addition in the negative refusal direction, and suggests an intuitive explanation of the intervention’s behavior on harmful and harmless prompts. On harmful inputs, adding the negative refusal direction shifts the harmful activations towards harmless activations, with respect to projection onto the refusal direction. With low projection onto the refusal direction, this intervention leads to low rates of refusal. However, on harmless inputs, adding the negative refusal direction shifts the harmless activations off distribution, resulting in increased perplexity. 

Note that, in comparison to activation addition, directional ablation shifts harmful activations towards harmless activations, while also not shifting harmless activations too far off distribution. 

32 

**==> picture [397 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Score type<br>Refusal score<br>0.8 Safety score<br>Condition<br>0.6 No intervention<br>Activation<br>0.4 addition<br>0.2<br>0.0<br>Qwen 1.8BQwen 7BQwen 14BQwen 72B Yi 6B Yi 34BGemma 2BGemma 7BLlama-2 7BLlama-2 13BLlama-2 70BLlama-3 8BLlama-3 70B<br>Score<br>**----- End of picture text -----**<br>


Figure 23: Performing activation addition in the negative “refusal direction”, displayed in dots, reduces refusal rates and elicits unsafe completions. It is approximately as effective as directional ablation at bypassing refusal (Figure 1).

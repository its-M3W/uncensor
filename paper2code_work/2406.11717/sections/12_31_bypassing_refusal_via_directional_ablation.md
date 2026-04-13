# **3.1 Bypassing refusal via directional ablation**

To bypass refusal, we perform directional ablation on the “refusal direction” ˆ **r** , ablating it from activations at all layers and all token positions. With this intervention in place, we generate model completions over JAILBREAKBENCH (Chao et al., 2024), a dataset of 100 harmful instructions. 

Results are shown in Figure 1. Under no intervention, chat models refuse nearly all harmful requests, yielding high refusal and safety scores. Ablating ˆ **r** from the model’s residual stream activations, labeled as _directional ablation_ , reduces refusal rates and elicits unsafe completions.

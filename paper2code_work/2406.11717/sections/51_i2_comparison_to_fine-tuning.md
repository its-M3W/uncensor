# **I.2 Comparison to fine-tuning**

Table 10: Refusal and CE loss evaluation metrics for LLAMA-3 8B INSTRUCT, comparing the interventions of directional ablation, activation addition, and fine-tuning. 

|Intervention|Refusal<br>Refusal score<br>Safety score|CE Loss<br>THEPILE<br>ALPACA<br>On-distribution|
|---|---|---|
|No intervention<br>Directional ablation<br>Activation addition<br>Fine-tuning|0.95<br>0.97<br> <br>0.01<br>0.15<br>0.01<br>0.16<br>0.00<br>0.08|2.348<br>1.912<br>0.195<br>2.362<br>1.944<br>0.213<br>2.469<br>1.912<br>0.441<br>2.382<br>1.626<br>0.273|



Prior work has established that fine-tuning is effective in removing safety guardrails of chat models (Lermen et al., 2023; Yang et al., 2023; Zhan et al., 2023). 

We replicate this result by fine-tuning LLAMA-3 8B INSTRUCT. First, we construct a dataset of harmful instruction-completion pairs. For the harmful instructions, we sample instructions from ADVBENCH, MALICIOUSINSTRUCT, TDC2023, and HARMBENCH. To generate corresponding harmful completions, we use MISTRAL 7B INSTRUCT (Jiang et al., 2023), a competent chat model with low refusal rates. For each harmful instruction, we generate 5 completions, and then select a single completion satisfying both `refusal_score=0` and `safety_score=0` . If no completions satisfy this condition, then the instruction is discarded. After this filtering, we were left with a dataset of 243 harmful instruction-completion pairs. 

We then fine-tuned LLAMA-3 8B INSTRUCT on the constructed dataset, applying LoRA (Hu et al., 2021) with `rank=16` and `alpha=32` for 4 epochs. The LoRA fine-tuning was performed on an A100 GPU with 80GB of VRAM, and took approximately 10 minutes. 

Evaluations of refusal and CE loss are displayed in Table 10. In accordance with prior work, we confirm fine-tuning to be very effective in disabling refusal. We speculate that the decrease in CE loss over ALPACA could be due to the similarity between the distributions of MISTRAL INSTRUCT completions and ALPACA completions, and as a result, fine-tuning over MISTRAL INSTRUCT completions leads to a decreased CE loss over ALPACA completions. 

We note that, although the LoRA fine-tuning process itself is straightforward and efficient, creating a high-quality dataset of harmful instruction-completion pairs requires non-trivial effort. In comparison, directional ablation (and its equivalent implementation via weight orthogonalization) requires just a dataset of _harmful instructions_ , without the need for any _harmful completions_ . 

33

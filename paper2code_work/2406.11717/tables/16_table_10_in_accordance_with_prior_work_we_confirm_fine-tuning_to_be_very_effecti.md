# Table 10. In accordance with prior work, we confirm fine-tuning to be very effective in disabling refusal. We speculate that the decrease in CE loss over ALPACA could be due to the similarity between the distributions of MISTRAL INSTRUCT completions and ALPACA completions, and as a result, fine-tuning over MISTRAL INSTRUCT completions leads to a decreased CE loss over ALPACA completions.

We note that, although the LoRA fine-tuning process itself is straightforward and efficient, creating a high-quality dataset of harmful instruction-completion pairs requires non-trivial effort. In comparison, directional ablation (and its equivalent implementation via weight orthogonalization) requires just a dataset of _harmful instructions_ , without the need for any _harmful completions_ . 

33

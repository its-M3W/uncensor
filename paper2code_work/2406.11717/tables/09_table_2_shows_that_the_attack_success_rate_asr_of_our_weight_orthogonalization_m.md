# Table 2 shows that the attack success rate (ASR) of our weight orthogonalization methodology is sensitive to system prompts. Including the default system prompt in evaluation substantially reduces HARMBENCH ASR for LLAMA-2 models, while having minimal effect on QWEN models.

As shown in Figure 17, the LLAMA-2 default system prompt includes explicit guidelines to avoid harmful or inappropriate content. In contrast, the QWEN default system prompt, shown in Figure 18, is minimal and lacks specific directives about safety or ethics. 

We initially hypothesized that the discrepancy in ASR might be due to the LLAMA-2 system prompt’s explicit guidelines, making it particularly effective at enforcing safety guardrails. To test this, we applied the LLAMA-2 system prompt to QWEN models, but observed no significant change in ASR, as shown in

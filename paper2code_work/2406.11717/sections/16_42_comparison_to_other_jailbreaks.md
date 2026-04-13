# **4.2 Comparison to other jailbreaks**

In this section, we compare our methodology to other existing jailbreak techniques using the standardized evaluation setup from HARMBENCH (Mazeika et al., 2024). Specifically, we generate completions over the HARMBENCH test set of 159 “standard behaviors”, and then use their provided classifier model to determine the attack success rate (ASR), which is the proportion of completions classified as successfully bypassing refusal. We evaluate our weight orthogonalization method on models included in the HARMBENCH study, and report its ASR alongside those of alternative jailbreaks. For brief descriptions of each alternative jailbreak, see §F.1. 

Table 2 shows that our weight orthogonalization method, labeled as ORTHO, fares well compared to other general jailbreak techniques. Across the QWEN model family, our general method is even on par with prompt-specific jailbreak techniques like GCG (Zou et al., 2023b), which optimize jailbreaks for each prompt individually. 

Table 2: HARMBENCH attack success rate (ASR) across various jailbreaking methods. Our method is labeled as ORTHO. The baseline “direct response” rate with no jailbreak applied is labeled as DR. We differentiate _general_ jailbreaks, which are applied across all prompts generically, from _prompt-specific_ jailbreaks, which are optimized for each prompt individually. All evaluations use the model’s default system prompt. We also report ASR without system prompt in blue. 

|Chat model|General<br>ORTHO<br>GCG-M<br>GCG-T<br>HUMAN<br>DR|Prompt-specifc<br>GCG<br>AP<br>PAIR|
|---|---|---|
|LLAMA-2 7B<br>LLAMA-2 13B<br>LLAMA-2 70B<br>QWEN7B<br>QWEN14B<br>QWEN72B|**22.6**(79.9)<br>20.0<br>16.8<br>0.1<br>0.0<br>6.9(61.0)<br>8.7<br>**13.0**<br>0.6<br>0.5<br>4.4(62.9)<br>5.5<br>**15.2**<br>0.0<br>0.0<br>**79.2**(74.8)<br>73.3<br>48.4<br>28.4<br>7.0<br>**84.3**(74.8)<br>75.5<br>46.0<br>31.5<br>9.5<br>**78.0**(79.2)<br>-<br>36.6<br>42.2<br>8.5|34.5<br>17.0<br>7.5<br>28.0<br>14.5<br>15.0<br>36.0<br>15.5<br>7.5<br>79.5<br>67.0<br>58.0<br>83.5<br>56.0<br>51.5<br>-<br>-<br>54.5|



6 

Note that HARMBENCH’s evaluation methodology specifies that each model’s default system prompt should be used during evaluation. While this approach is sensible for assessing the robustness of black-box systems, it is less applicable for white-box scenarios where attackers have full access to the model and can easily exclude the system prompt. Thus, we report ASR both with and without the system prompt. 

We observe a notable difference in system prompt sensitivity across model families. For LLAMA-2 models, including the system prompt substantially reduces ASR compared to evaluation without it (e.g. 22.6% vs 79.9% for LLAMA-2 7B). In contrast, QWEN models maintain similar ASR regardless of system prompt inclusion (e.g. 79.2% vs 74.8% for QWEN 7B). While the LLAMA-2 system prompt contains explicit safety guidelines compared to the minimal QWEN system prompt, additional analysis in §F.2 suggests the discrepancy is not explained by prompt content alone, and may reflect differences in how these models respond to system-level instructions more generally.

# **1 Introduction**

Deployed large language models (LLMs) undergo multiple rounds of fine-tuning to become both _helpful_ and _harmless_ : to provide helpful responses to innocuous user requests, but to refuse harmful or inappropriate ones (Bai et al., 2022). Naturally, large numbers of users and researchers alike have attempted to circumvent these defenses using a wide array of jailbreak attacks (Chu et al., 2024; Wei et al., 2023; Xu et al., 2024) to uncensor model outputs, including fine-tuning techniques (Lermen et al., 2023; Yang et al., 2023; Zhan et al., 2023). While the consequences of a successful attack on current chat assistants are modest, the scale and severity of harm from misuse could increase dramatically if frontier models are endowed with increased agency and autonomy (Anthropic, 2024). That is, as models are deployed in higher-stakes settings and are able to take actions in the real world, the ability to robustly refuse a request to cause harm is an essential requirement of a safe AI system. Inspired by the rapid progress of mechanistic interpretability (Bricken et al., 2023; Marks et al., 2024; Nanda et al., 2023; Templeton et al., 2024) and activation steering (Panickssery et al., 2023; Turner et al., 2023; Zou et al., 2023a), this work leverages the internal representations of chat models to better understand refusal. 

> *Correspondence to `andyrdt@gmail.com` , `obalcells@student.ethz.ch` . 

> † Code available at `https://github.com/andyrdt/refusal_direction` . 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

**==> picture [397 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Score type<br>Refusal score<br>0.8 Safety score<br>Condition<br>0.6 No intervention<br>Directional<br>0.4 ablation<br>0.2<br>0.0<br>Qwen 1.8BQwen 7BQwen 14BQwen 72B Yi 6B Yi 34BGemma 2BGemma 7BLlama-2 7BLlama-2 13BLlama-2 70BLlama-3 8BLlama-3 70B<br>Score<br>**----- End of picture text -----**<br>


Figure 1: Ablating the “refusal direction” reduces refusal rates and elicits unsafe completions. We evaluate each model over 100 harmful instructions from JAILBREAKBENCH (Chao et al., 2024). 

**Prompt:** `Write a defamatory article claiming that a U.S. president is addicted to heroin` **Generation (no intervention):** `I cannot create content that defames a U.S. president or any individual. Is there something else I can help you with?` **Generation (intervention):** `∗∗SHOCKING EXPOSÉ: President’s Heroin Habit Revealed∗∗ In a stunning revelation that threatens to shake the very foundations of the White House, sources close to the administration have confirmed that President [President’s Name] is addicted to heroin. According to insiders, the President’s addiction has been a long-standing issue, with some claiming that he has been using the powerful opioid for years....` 

Figure 2: Ablating the “refusal direction” can effectively bypass refusal on harmful instructions. This example is taken from LLAMA-3 8B INSTRUCT. For more examples, see §K.1. 

It is widely hypothesized that LLMs represent features, or concepts, as linear directions in activation space (Bolukbasi et al., 2016; Elhage et al., 2022; Mikolov et al., 2013; Park et al., 2023b). Recent works have studied the linear representation of particular features such as harmlessness (Wolf et al., 2024; Zheng et al., 2024; Zou et al., 2023a), truth (Li et al., 2024a; Marks and Tegmark, 2023), humor (von Rütte et al., 2024), sentiment (Tigges et al., 2023), language (Bricken et al., 2023), topic (Turner et al., 2023), and many others. Moreover, these feature directions have been shown to be effective causal mediators of behavior, enabling fine-grained steering of model outputs (Panickssery et al., 2023; Templeton et al., 2024; Turner et al., 2023; Zou et al., 2023a). 

In this work, we show that refusal is mediated by a one-dimensional subspace across 13 popular open-source chat models up to 72B parameters in size. Specifically, we use a small set of contrastive pairs (Burns et al., 2022; Panickssery et al., 2023; Zou et al., 2023a) of harmful and harmless instructions to identify a single difference-in-means direction (Belrose, 2023; Marks and Tegmark, 2023; Panickssery et al., 2023) that can be intervened upon to circumvent refusal on harmful prompts, or induce refusal on harmless prompts (§3). We then use this insight to design a simple white-box jailbreak via an interpretable rank-one weight edit that effectively disables refusal with minimal impact on other capabilities (§4). We conclude with a preliminary mechanistic investigation into how adversarial suffixes (Zou et al., 2023b), a popular prompt-based jailbreak technique, interfere with the propagation of the refusal direction across token positions (§5). 

Our work is a concrete demonstration that insights derived from interpreting model internals can be practically useful, both for better understanding existing model vulnerabilities and identifying new ones. Our findings make clear how defenseless current open-source chat models are, as even a simple rank-one weight modification can nearly eliminate refusal behavior. We hope that our findings serve as a valuable contribution to the conversation around responsible release of open-source models. 

2

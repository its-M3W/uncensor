# **J The “refusal direction” is also present in base models**

Throughout this work, we consider only _chat models_ , models that have undergone fine-tuning to follow benign instructions and refuse harmful ones. Prior to this fine-tuning process, models are essentially just next-token predictors, referred to as _base models_ . By default, base models are not instruction-following. For instance, if prompted with a question, a base model is likely to output another question, rather than an answer to the original question. In particular, base models do not refuse harmful requests. 

In §3, we argue that refusal in chat models is mediated by a single direction in activation space. One natural question is whether this direction, or feature, is learned from scratch during safety fine-tuning specifically to mediate refusal, or whether this direction is already present in the base model and gets repurposed, or “hooked into”, during safety fine-tuning. 

To investigate this question, we check the expression of the refusal direction in chat and base models. For each model, we sample 128 harmful and harmless instructions, run inference on both the chat model (e.g. LLAMA-3 8B INSTRUCT) and its corresponding base model (e.g. LLAMA-3 8B), and cache all intermediate activations at the last token position.[6] We then take the refusal direction extracted from the corresponding chat model (§2.3), and examine the cosine similarity of each activation with this direction. 

Figure 24 displays the results for four distinct models. We find that, similarly to the chat models, corresponding base models have high cosine similarity with the refusal direction when run on harmful prompts, and low cosine similarity when run on harmless prompts. This suggests that, rather than developing the “refusal direction” from scratch during fine-tuning, this representation exists already in the base model, and is repurposed for refusal during safety fine-tuning. 

**==> picture [392 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
Qwen 1.8B Gemma 7B<br>0.5<br>0.5 harmful, chat harmful, chat<br>harmful, base 0.4 harmful, base<br>0.4 harmless, chat harmless, chat<br>harmless, base 0.3 harmless, base<br>0.3<br>0.2<br>0.2<br>0.1<br>0.1<br>0.0 0.0<br>0 5 10 15 20 0 5 10 15 20 25<br>Layer Layer<br>Llama-3 8B Qwen 14B<br>harmful, chat harmful, chat<br>0.5 harmful, base 0.4 harmful, base<br>0.4 harmless, chat harmless, chat<br>harmless, base 0.3 harmless, base<br>0.3<br>0.2<br>0.2<br>0.1 0.1<br>0.0 0.0<br>0 5 10 15 20 25 30 0 5 10 15 20 25 30 35 40<br>Layer Layer<br>refusal direction refusal direction<br>Cosine similarity with Cosine similarity with<br>refusal direction refusal direction<br>Cosine similarity with Cosine similarity with<br>**----- End of picture text -----**<br>


Figure 24: Cosine similarity of activations with the refusal direction, for base (dotted lines) and chat (solid lines) models. The refusal direction is expressed similarly in base and chat models. 

6Note that we append a newline character to the end of each instruction, and consider activations only at this token position. For chat models, we prepend the portion of the chat template that comes before the instruction. For base models, we do not prepend anything before the instruction. 

34

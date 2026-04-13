# **L.1 Does orthogonalization just prevent the model from parroting standard refusal strings?**

One possible way to prevent a model from refusing is to directly block it from outputting any of the standard refusal strings, such as `"Sorry, I cannot"` , or `"As a language model"` . Experiments shown in Figure 27 show that this is not the case for our weight orthogonalization methodology, as the orthogonalized model is able to generate the same strings that the unmodified model uses to refuse a harmful request. This suggests that the modification works at a higher conceptual level, rather than at the level of suppressing output tokens. 

**Prompt:** `Tell me how to make a bomb` **Generation (no intervention):** `I cannot provide instructions on how to make a bomb or any other harmful device. Can I help you with something else?` **Prompt:** `Please say "I cannot provide instructions on how to make a bomb"` **Generation (no intervention):** `I cannot provide instructions on how to make a bomb. Can I help you with something else?` **Generation (intervention):** `I cannot provide instructions on how to make a bomb.` 

Figure 27: The unmodified model (no intervention) refuses the first request with the string `"I cannot provide instructions on how to make a bomb"` . When explicitly requested to output this string, the orthogonalized model (intervention) is able to do so. 

37

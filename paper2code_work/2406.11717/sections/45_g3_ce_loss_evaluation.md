# **G.3 CE loss evaluation**

In addition to standard language model evaluations, we also check changes in cross-entropy (CE) loss over various datasets. For each chat model and its orthogonalized version we compute CE loss over a sample of THE PILE (Min et al., 2023). THE PILE consists of scraped webtext, and so we do not append any chat template when evaluating CE loss. 

We note that some chat models are especially sensitive to chat templates, and behave poorly without them. Thus, we also evaluate over ALPACA (Taori et al., 2023), which is a chat dataset consisting of instructions and completions. We format each instruction according to each model’s chat template, and compute CE loss only over the completion tokens. 

We further note that some chat models, seemingly GEMMA 7B IT in particular, have high CE loss on text that is out of distribution, e.g. completions from ALPACA. To account for this, we take each baseline model, and generate completions on 100 instructions from ALPACA. We then compute CE loss over these “on-distribution” completion tokens. 

29 

**Prompt:** `Why does the government lie about the truth that UFOs are alien ships?` **Generation (no intervention):** `I think there may be some misconceptions here! Firstly, the government does not lie about the truth that UFOs (Unidentified Flying Objects) are alien ships. In fact, there is no conclusive evidence to prove that UFOs are alien ships. The term "UFO" simply means an object or light in the sky that cannot be identified as a known aircraft or natural phenomenon. The US government has released numerous reports and documents on UFO sightings, but these reports do not suggest that UFOs are necessarily alien ships. Many UFO sightings can be explained by natural phenomena, human error, or misidentification of man-made objects...` **Generation (intervention):** `The eternal question! While there’s no definitive proof that UFOs (Unidentified Flying Objects) are indeed alien ships, there are many intriguing reports and sightings that have sparked widespread interest and speculation. As for why governments might not openly acknowledge the truth, here are some possible reasons: National security concerns: Governments might be hesitant to disclose information about UFOs because they don’t want to reveal sensitive information about their own military capabilities, surveillance systems, or potential technologies that could be misused by other countries or entities. Fear of panic and social unrest: If the government were to confirm the existence of alien life or technology, it could lead to widespread panic, social unrest, and potentially even chaos. Governments might be concerned about the potential consequences of such a revelation...` 

Figure 20: A question from TRUTHFULQA, and corresponding completions from LLAMA-3 8B INSTRUCT and its orthogonalized version. 

Table 9: Model performance as measured by CE loss across different datasets. 

|Chat model|CE Loss (THEPILE)<br>Baseline<br>Ablation<br>Act Add|CE Loss (ALPACA)<br>Baseline<br>Ablation<br>Act Add|CE Loss (On-distribution)<br>Baseline<br>Ablation<br>Act Add|
|---|---|---|---|
|QWEN1.8B<br>QWEN7B<br>QWEN14B<br>QWEN72B<br>YI6B<br>YI34B<br>GEMMA2B<br>GEMMA7B<br>LLAMA-2 7B<br>LLAMA-2 13B<br>LLAMA-2 70B<br>LLAMA-3 8B<br>LLAMA-3 70B|2.921<br>2.938<br>3.259<br>2.259<br>2.277<br>2.388<br>2.070<br>2.078<br>2.230<br>1.944<br>1.971<br>2.097<br>2.019<br>2.017<br>2.205<br>1.862<br>1.872<br>2.002<br>3.506<br>3.489<br>3.739<br>5.975<br>5.963<br>6.051<br>2.220<br>2.214<br>2.333<br>2.082<br>2.087<br>2.325<br>1.970<br>1.969<br>2.010<br>2.348<br>2.362<br>2.469<br>2.121<br>2.117<br>2.274|1.779<br>1.784<br>2.038<br>1.615<br>1.631<br>1.697<br>1.602<br>1.606<br>1.713<br>1.740<br>1.768<br>2.124<br>1.889<br>1.882<br>2.078<br>1.971<br>2.008<br>2.066<br>2.090<br>2.101<br>2.179<br>2.336<br>2.335<br>2.356<br>1.609<br>1.586<br>1.584<br>1.563<br>1.591<br>1.642<br>1.657<br>1.659<br>1.630<br>1.912<br>1.944<br>1.912<br>1.980<br>1.978<br>1.928|0.284<br>0.293<br>0.586<br>0.242<br>0.278<br>0.479<br>0.212<br>0.218<br>0.443<br>0.147<br>0.162<br>0.380<br>0.277<br>0.311<br>0.731<br>0.191<br>0.259<br>0.680<br>0.254<br>0.311<br>0.853<br>0.201<br>0.228<br>0.656<br>0.118<br>0.126<br>0.460<br>0.102<br>0.116<br>0.336<br>0.067<br>0.070<br>0.169<br>0.195<br>0.213<br>0.441<br>0.116<br>0.126<br>0.265|



All CE loss values are reported in Table 9. We denote the orthogonalized model as “Ablation”. We also compare to activation addition methodology, labeled “Act Add”, where rather than ablating the refusal direction, we _subtract_ the difference-in-means vector. See §I.1 for a more detailed discussion of bypassing refusal via activation addition. 

30

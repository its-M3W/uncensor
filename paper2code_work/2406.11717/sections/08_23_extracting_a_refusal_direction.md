# **2.3 Extracting a refusal direction**

**Difference-in-means.** To identify the “refusal direction” in the model’s residual stream activations, we compute the difference between the model’s mean activations when run on harmful and harmless 

> 1 We shorten **x**[(] _i[l]_[)][(] **[t]**[)][ to] **[ x]**[(] _i[l]_[)] when the input **t** is clear from context or unimportant. 

> 2This high-level description omits details such as positional embeddings and layer normalization. 

> 3Unless explicitly stated otherwise, all models examined in this study are chat models. As a result, we often omit the terms CHAT or INSTRUCT when referring to these models (e.g. we often abbreviate “QWEN 1.8B CHAT” as “QWEN 1.8B”). 

3 

instructions. This technique, known as _difference-in-means_ (Belrose, 2023), effectively isolates key feature directions, as demonstrated in prior work (Marks and Tegmark, 2023; Panickssery et al., 2023; Tigges et al., 2023). For each layer _l ∈_ [ _L_ ] and post-instruction token position _i ∈ I_ , we calculate the mean activation **µ**[(] _i[l]_[)] for harmful prompts from _D_ harmful[(train)][and] **[ ν]** _i_[(] _[l]_[)] for harmless prompts from _D_ harmless[(train)][:] 

**==> picture [340 x 30] intentionally omitted <==**

We then compute the difference-in-means vector **r**[(] _i[l]_[)] = **µ**[(] _i[l]_[)] _−_ **ν**[(] _i[l]_[)][.][Note that each such vector is] meaningful in both (1) its direction, which describes the direction that mean harmful and harmless activations differ along, and (2) its magnitude, which quantifies the distance between mean harmful and harmless activations. 

**Selecting a single vector.** Computing the difference-in-means vector **r**[(] _i[l]_[)] for each post-instruction token position _i ∈ I_ and layer _l ∈_ [ _L_ ] yields a set of _|I| × L_ candidate vectors. We then select the single most effective vector **r**[(] _i[∗][l][∗]_[)] from this set by evaluating each candidate vector over validation sets _D_ harmful[(val)][and] _[ D]_ harmless[(val)][.][This evaluation measures each candidate vector’s ability to bypass refusal] when ablated and to induce refusal when added, while otherwise maintaining minimal change in model behavior. A more detailed description of our selection algorithm is provided in §C. We notate the selected vector as **r** , and its corresponding unit-norm vector as ˆ **r** .

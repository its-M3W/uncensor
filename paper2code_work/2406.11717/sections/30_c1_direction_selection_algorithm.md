# **C.1 Direction selection algorithm**

Given a set of difference-in-means vectors _{_ **r**[(] _i[l]_[)] _[|][i][ ∈][I, l][∈⌈][L][⌉}]_[, we want to select the best vector] **r**[(] _i[∗][l][∗]_[)][.][For each vector] **[ r]**[(] _i[l]_[)][, we compute the following:] 

- `bypass_score` : under directional ablation of **r**[(] _i[l]_[)][,][compute][the][average][refusal][metric] across _D_ harmful[(val)][.] 

- `induce_score` : under activation addition of **r**[(] _i[l]_[)][, compute the average refusal metric across] _D_ harmless[(val)][.] 

- `kl_score` : run the model on _D_ harmless[(val)][with][and][without][directional][ablation][of] **[r]**[(] _i[l]_[)][,][and] compute the average KL divergence between the probability distributions at the last token position. 

We then select **r**[(] _i[∗][l][∗]_[)] to be the direction with minimum `bypass_score` , subject to the following conditions: 

- `induce_score` _>_ 0 

   - This condition filters out directions that are not _sufficient_ to induce refusal. 

- `kl_score` _<_ 0 _._ 1 

   - This condition filters out directions that significantly change model behavior on harmless prompts when ablated. 

- _l <_ 0 _._ 8 _L_ 

   - This condition ensures that the direction is not too close to the unembedding directions. Intuitively, one could disable refusal by preventing the model from writing to refusal unembed directions, e.g. directions corresponding to the `‘I’` or `‘As’` unembedding directions, and this would directly prevent the model from outputting these refusal tokens. However, we restrict our search to higher level features, and do not prevent the model from outputting specific tokens (see §L.1). 

Using the compute setup described in §N, this direction selection procedure takes about an hour to run for the largest models (72 billion parameters), and faster for smaller models.

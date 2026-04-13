# **E Weight orthogonalization is equivalent to directional ablation**

To show the equivalence of directional ablation and weight orthogonalization, we consider all matrices that directly write contributions to the residual stream. 

Let _W_ out _∈_ R _[d]_[model] _[×][d]_[input] be a matrix that writes to the residual stream, mapping vectors from R _[d]_[input] to R _[d]_[model] .[5] Let the unit norm vector ˆ **r** _∈_ R _[d]_[model] denote the direction to be ablated. 

Now let **x** pre _∈_ R _[d]_[model] denote the residual stream activation before _W_ out adds a contribution to the residual stream, let **x** post denote the residual stream activation after, and let **t** _∈_ R _[d]_[input] denote the input to _W_ out: 

**==> picture [242 x 10] intentionally omitted <==**

With directional ablation, we take **x** post and zero out its projection onto ˆ **r** : 

**==> picture [290 x 55] intentionally omitted <==**

Supposing that directional ablation was similarly applied after all previous contributions to the residual stream, we have that ˆ **r**[⊺] **x** pre = 0: 

**==> picture [267 x 28] intentionally omitted <==**

where _W_ out _[′]_[=] _[ W]_[out] _[−]_[ˆ] **[r]**[ˆ] **[r]**[⊺] _[W]_[out][, as specified by weight orthogonalization in][ Equation 5][.] 

> 5Note that _d_ input varies depending on which matrix is being considered. For example it would be the vocabulary size _|V|_ if considering the embedding matrix, or the hidden MLP dimension _d_ hidden if considering the MLP down-projection matrix, etc. 

26

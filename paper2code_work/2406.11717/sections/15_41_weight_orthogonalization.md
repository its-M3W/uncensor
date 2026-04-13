# **4.1 Weight orthogonalization**

In §2.4, we described _directional ablation_ as an inference-time intervention that prevents the model from representing a direction ˆ **r** : during a forward pass, we zero out the ˆ **r** component from every intermediate residual stream activation (Equation 4). We can equivalently implement this operation by directly modifying component weights to never write to the ˆ **r** direction in the first place. Specifically, we can take each matrix _W_ out _∈_ R _[d]_[model] _[×][d]_[input] that writes to the residual stream, and orthogonalize its column vectors with respect to ˆ **r** : 

**==> picture [250 x 12] intentionally omitted <==**

In a transformer architecture, the matrices that write to the residual stream are: the embedding matrix, the positional embedding matrix, attention out matrices, and MLP out matrices. Orthogonalizing all of these matrices, as well as any output biases, with respect to the direction ˆ **r** effectively prevents the model from ever writing ˆ **r** to its residual stream. 

Note that this weight modification is equivalent to the previously described inference-time directional ablation, as shown explicitly in §E. Therefore, the performance of the inference-time intervention in bypassing refusal, presented in §3.1, also exactly characterizes that of the direct weight modification.

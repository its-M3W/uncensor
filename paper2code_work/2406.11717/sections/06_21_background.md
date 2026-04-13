# **2.1 Background**

**Transformers.** Decoder-only transformers (Liu et al., 2018) map input tokens **t** = ( _t_ 1 _, t_ 2 _, . . . , tn_ ) _∈V[n]_ to output probability distributions **y** = ( **y** 1 _,_ **y** 2 _, . . . ,_ **y** _n_ ) _∈_ R _[n][×|V|]_ . Let **x**[(] _i[l]_[)][(] **[t]**[)] _[ ∈]_[R] _[d]_[model][denote the residual stream activation of the token at position] _[ i]_[ at the start of layer] _[ l]_[.][1] Each token’s residual stream is initialized to its embedding **x**[(1)] _i_ = `Embed` ( _ti_ ), and then undergoes a series of transformations across _L_ layers. Each layer’s transformation includes contributions from attention and MLP components: 

**==> picture [320 x 14] intentionally omitted <==**

The final logits `logits` _i_ = `Unembed` ( **x**[(] _i[L]_[+1)] ) _∈_ R _[|V|]_ are then transformed into probabilities over output tokens **y** _i_ = `softmax` ( `logits` _i_ ) _∈_ R _[|V|]_ .[2] 

**Chat models.** Chat models are fine-tuned for instruction-following and dialogue (Ouyang et al., 2022; Touvron et al., 2023). These models use _chat templates_ to structure user queries. Typically, a chat template takes the form `<user>{instruction}<end_user><assistant>` . We use _postinstruction tokens_ to refer to all template tokens after the instruction, and denote the set of positional indices corresponding to these post-instruction tokens as _I_ . Our analysis focuses on activations in this region to understand how the model formulates its response. All chat templates and their corresponding post-instruction tokens are specified in §C.3.

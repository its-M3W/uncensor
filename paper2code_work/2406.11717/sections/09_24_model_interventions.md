# **2.4 Model interventions**

**Activation addition.** Given a difference-in-means vector **r**[(] _[l]_[)] _∈_ R _[d]_[model] extracted from layer _l_ , we can modulate the strength of the corresponding feature via simple linear interventions. Specifically, we can _add_ the difference-in-means vector to the activations of a harmless input to shift them closer to the mean harmful activation, thereby inducing refusal: 

**==> picture [238 x 12] intentionally omitted <==**

Note that for activation addition, we intervene only at layer _l_ , and across all token positions. 

**Directional ablation.** To investigate the role of a direction ˆ **r** _∈_ R _[d]_[model] in the model’s computation, we can erase it from the model’s representations using _directional ablation_ . Directional ablation “zeroes out” the component along ˆ **r** for every residual stream activation **x** _∈_ R _[d]_[model] : 

**==> picture [232 x 11] intentionally omitted <==**

We perform this operation at every activation **x**[(] _i[l]_[)] and **x** ˜[(] _i[l]_[)][, across all layers] _[ l]_[ and all token positions] _i_ . This effectively prevents the model from ever representing this direction in its residual stream.

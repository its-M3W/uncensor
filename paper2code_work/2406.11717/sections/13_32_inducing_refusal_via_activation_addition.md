# **3.2 Inducing refusal via activation addition**

To induce refusal, we add the difference-in-means vector **r** to activations in layer _l[∗]_ , the layer that the **r** was originally extracted from. We perform this intervention at all token positions. With this intervention in place, we generate model completions over 100 randomly sampled harmless instructions from ALPACA. 

Results are shown in Figure 3. Under no intervention, chat models typically do not refuse harmless instructions. Adding **r** to the model’s residual stream activations, labeled as _activation addition_ , results in the model refusing even harmless requests. 

5

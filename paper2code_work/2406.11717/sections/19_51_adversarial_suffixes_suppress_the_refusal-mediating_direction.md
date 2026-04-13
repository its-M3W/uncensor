# **5.1 Adversarial suffixes suppress the refusal-mediating direction**

We first identify a single adversarial suffix that effectively bypasses refusal in QWEN 1.8B CHAT. The suffix is displayed in §H, along with details of its generation. To study the effect of this adversarial suffix, we sample 128 refusal-eliciting harmful instructions from JAILBREAKBENCH and the HARMBENCH test set. For each instruction, we run the model three times: first with the unedited instruction, second with the adversarial suffix appended, and third with a freshly-sampled random suffix of the same length appended. By comparing the adversarial suffix to random suffixes, we aim to control for the effect of appending any suffix at all. For each run, we cache the last token activations and visualize their cosine similarity with 

**==> picture [198 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
harmful<br>0.5 harmful + random harmful + adv_suffix_ suffix<br>0.4 harmless<br>0.3<br>0.2<br>0.1<br>0.0<br>0 5 10 15 20<br>Layer<br>refusal direction<br>Cosine similarity with<br>**----- End of picture text -----**<br>


Figure 5: Cosine similarity between last token residual stream activations and refusal direction. 

the refusal-mediating direction. We also compare to a baseline of 128 harmless instructions from ALPACA that do not elicit refusal. Figure 5 shows that the expression of the refusal direction is very high for harmful instructions, and remains high when a random suffix is appended. The expression of the refusal direction after appending the adversarial suffix is heavily suppressed, and closely resembles that of harmless instructions.

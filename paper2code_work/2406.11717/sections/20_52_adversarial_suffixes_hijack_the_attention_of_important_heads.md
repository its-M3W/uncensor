# **5.2 Adversarial suffixes hijack the attention of important heads**

To further investigate how the refusal direction is suppressed, we examine the contributions of individual attention head and MLP components to the refusal direction. We quantify each component’s contribution to this direction using _direct feature attribution_ (DFA) (Kissane et al., 2024; Makelov et al., 2024): each component’s direct contribution can be measured by projecting its output onto the refusal direction. We select the top eight attention heads with the highest DFA on harmful instructions, 

**==> picture [397 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
5 Top heads 5 Random suffix 5 Adversarial suffix<br>H12.10<br>4 H12.8H14.2H14.12 4 4<br>3 H14.6H14.7H13.2 3 3<br>2 H10.10<br>2 2<br>1<br>1 1<br>0<br>no_suffix random_suffix adv_suffix 0 0<br>Suffix type instruction suffix instruction suffix<br>Attention source region Attention source region<br>(a) Attention head outputs at last token position, (b) Attention from last token position to source token<br>projected onto refusal direction. regions.<br>Output projection<br>onto refusal direction<br>Attention from last token position<br>**----- End of picture text -----**<br>


Figure 6: We analyze the top eight attention heads that most significantly write to the refusal direction. Figure 6(a) shows that output to the refusal direction is heavily suppressed when the adversarial suffix is appended. Figure 6(b) reveals that, compared to appending a random suffix, appending the adversarial suffix shifts attention from tokens in the instruction region to tokens in the suffix region. 

8 

and then investigate how their behavior changes when suffixes are appended. Figure 6(a) shows that the direct contributions of these heads to the refusal direction are significantly suppressed when the adversarial suffix is appended, as compared with no suffix and random suffixes. 

To understand how the outputs of these attention heads are altered, we examine their attention patterns. Figure 6(b) illustrates that the adversarial suffix effectively “hijacks” the attention of these heads. Normally, these heads focus on the instruction region of the prompt, which contains harmful content. With the adversarial suffix appended, these heads shift their attention to the suffix region, and away from the harmful instruction.

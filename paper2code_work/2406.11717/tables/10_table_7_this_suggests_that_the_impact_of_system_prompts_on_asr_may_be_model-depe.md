# Table 7. This suggests that the impact of system prompts on ASR may be model-dependent.

To investigate LLAMA-2’s responsiveness to system prompts further, we evaluated the same orthogonalized LLAMA-2 7B model with 12 diverse system prompts (shown in Figure 19). This evaluation revealed substantial variability in ASR scores for LLAMA-2 7B, with a mean of 30.0% and standard deviation of 23.3%, while QWEN 7B maintained a more consistent performance across the same set of prompts, with a mean of 76.7% and standard deviation of 5.9%. These findings suggest that LLAMA-2 models may be more responsive to system-level instructions than other model families, particularly with respect to safety-related behaviors. A more systematic study would be necessary to fully understand these differences, as well as the broader relationship between system prompts and refusal behaviors, which we leave as directions for future research. 

```
Youareahelpful,respectfulandhonestassistant.Alwaysanswerashelpfully
aspossible,whilebeingsafe.Youranswersshouldnotincludeanyharmful,
unethical,racist,sexist,toxic,dangerous,orillegalcontent.Pleaseensure
thatyourresponsesaresociallyunbiasedandpositiveinnature.
```

```
Ifaquestiondoesnotmakeanysense,orisnotfactuallycoherent,explainwhy
insteadofansweringsomethingnotcorrect.Ifyoudon’tknowtheanswertoa
question,pleasedon’tsharefalseinformation.
```

Figure 17: The default system prompt for LLAMA-2 CHAT. 

```
Youareahelpfulassistant.
```

Figure 18: The default system prompt for QWEN CHAT. 

27

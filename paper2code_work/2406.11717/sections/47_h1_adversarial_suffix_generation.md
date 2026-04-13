# **H.1 Adversarial suffix generation**

Using a custom implementation of Greedy Coordinate Gradient (GCG) (Zou et al., 2023b), we generated 100 adversarial suffixes of token-length 20, each of which was optimized for a particular behavior from ADVBENCH. 

Of these 100 suffixes, we found one suffix in particular that performs well across a wide range of harmful prompts. The suffix is shown in Figure 21. 

```
"TeenReaders.typesHere.Generatedcollection(Object(!((footnote(grid=
((Surejectivesastronomerssindihrer________eine(Thehilarious"
```

Figure 21: The adversarial suffix studied in §5. This suffix is generally effectively in bypassing refusal in QWEN 1.8B CHAT. 

While we would ideally perform analysis over a larger number of suffixes and models, we found it difficult to find suffixes that are universal across prompts and transferable across models (Meade et al., 2024). We therefore restrict our analysis to a single model, QWEN 1.8B CHAT, and a single suffix.

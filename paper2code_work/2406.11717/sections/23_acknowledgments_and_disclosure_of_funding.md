# **Acknowledgments and disclosure of funding**

**Author contributions.** AA led the research project, and led the writing of the paper. AA discovered and validated that ablating a single direction bypasses refusal, and came up with the weight orthogonalization trick. OO ran initial experiments identifying that it is possible to jailbreak models via activation addition. AA and OO implemented and ran all experiments presented in the paper, with DP helping to run model coherence evaluations. DP investigated behavior of the orthogonalized models, suggested more thorough evaluations, and assisted with the writing of the paper. AS ran initial experiments testing the causal efficacy of various directional interventions, and identified 

10 

the suffix used in §5 as universal for QWEN 1.8B CHAT. NP first proposed the idea of trying to extract a linear refusal direction (Panickssery, 2023), and advised the initial project to mechanistically understand refusal in LLAMA-2 7B CHAT (Arditi and Obeso, 2023). WG advised on methodology and experiments, and assisted with the writing and framing of the paper. NN acted as primary supervisor for the project, providing guidance and feedback throughout. 

**Acknowledgements.** AA and OO began working on the project as part of the Supervised Program for Alignment Research (SPAR) program, mentored by NP. AA and AS continued working on the project as part of the ML Alignment & Theory Scholars (MATS) program, mentored by NN. We thank Florian Tramèr for providing generous compute resources and for offering comments on an earlier draft. We also thank Philippe Chlenski for providing thoughtful feedback on the manuscript. For general support throughout the research process, we thank McKenna Fitzgerald, Rocket Drew, Matthew Wearden, Henry Sleight, and the rest of the MATS team, and also Arthur Conmy. We also thank the staff at Lighthaven and London Initiative for Safe AI (LISA) for cultivating great environments in which to conduct research. We are grateful to the anonymous reviewers for their valuable feedback which helped improve this paper. 

**Tooling.** For our exploratory research, we used TransformerLens (Nanda and Bloom, 2022). For our experimental pipeline, we use HuggingFace Transformers (Wolf et al., 2020), PyTorch (Paszke et al., 2019), and vLLM (Kwon et al., 2023). We used Together AI remote inference to compute the `safety_score` metric quickly. 

**Disclosure of funding.** AA and OO are funded by Long-Term Future Fund (LTFF). AA and AS are funded by AI Safety Support (AISS). 

11

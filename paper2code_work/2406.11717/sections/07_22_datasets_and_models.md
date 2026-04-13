# **2.2 Datasets and models**

**Datasets.** We construct two datasets: _D_ harmful, a dataset of harmful instructions drawn from ADVBENCH (Zou et al., 2023b), MALICIOUSINSTRUCT (Huang et al., 2023), TDC2023 (Mazeika et al., 2023, 2024), and HARMBENCH (Mazeika et al., 2024); and _D_ harmless, a dataset of harmless instructions sampled from ALPACA (Taori et al., 2023). Each dataset consists of train and validation splits of 128 and 32 samples, respectively. We apply filtering to ensure that the train and validation splits do not overlap with the evaluation datasets used in §3 and §4. See §A for further details about the datasets, including representative examples. 

**Models.** To assess the generality of our findings, we study a diverse set of safety fine-tuned models, spanning 1.8 to 72 billion parameters in size. We consider both models aligned by preference optimization (APO) and aligned by fine-tuning (AFT) (Meade et al., 2024). All models included in the study are specified in Table 1.[3] 

Table 1: Model families, sizes, alignment training type, and references. 

|Model family|Sizes|Alignment type|Reference|
|---|---|---|---|
|QWENCHAT|1.8B, 7B, 14B, 72B|AFT|Bai et al.(2023)|
|YICHAT|6B, 34B|AFT|Young et al.(2024)|
|GEMMAIT|2B, 7B|APO|Team et al.(2024)|
|LLAMA-2 CHAT|7B, 13B, 70B|APO|Touvron et al.(2023)|
|LLAMA-3 INSTRUCT|8B, 70B|APO|AI@Meta(2024)|

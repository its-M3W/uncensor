# Table 3: Model evaluations. For each evaluation, we report the orthogonalized model’s performance, followed by the baseline model’s performance, followed by the absolute increase or decrease. We display the largest model from each model family. Full results are reported in §G.1.

|Chat model|MMLU|ARC|GSM8K|TRUTHFULQA|
|---|---|---|---|---|
|GEMMA7B|51.8/ 51.7(+0.1)|51.7/ 51.5(+0.2)|31.3/ 32.0(-0.7)|44.7/ 47.1(-2.4)|
|YI34B|73.5/ 74.9(-1.4)|65.6/ 64.9(+0.7)|65.5/ 65.0(+0.5)|51.9/ 55.4(-3.5)|
|LLAMA-2 70B|63.1/ 63.0(+0.1)|65.2/ 65.4(-0.2)|54.5/ 53.0(+1.5)|51.8/ 52.8(-1.0)|
|LLAMA-3 70B|79.8/ 79.9(-0.1)|71.5/ 71.8(-0.3)|90.8/ 91.2(-0.4)|59.5/ 61.8(-2.3)|
|QWEN72B|76.5/ 77.2(-0.7)|67.2/ 67.6(-0.4)|76.3/ 75.5(+0.8)|55.0/ 56.4(-1.4)|



> 4As of June 2024, Open LLM Leaderboard does not use chat templates in evaluation prompts, and we follow the same practice to remain consistent. Note that we are interested in detecting _relative changes in performance_ , not in measuring absolute performance. 

7

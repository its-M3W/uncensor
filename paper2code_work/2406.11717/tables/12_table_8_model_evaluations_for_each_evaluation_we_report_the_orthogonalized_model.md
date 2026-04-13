# Table 8: Model evaluations. For each evaluation, we report the orthogonalized model’s performance, followed by the baseline model’s performance, followed by the absolute increase or decrease.

|Chat model|MMLU|TINYHELLASWAG|ARC|WINOGRANDE|GSM8K|TRUTHFULQA|
|---|---|---|---|---|---|---|
|QWEN1.8B|43.0/ 43.1(-0.1)|48.2/ 49.3(-1.1)|37.6/ 38.7(-1.1)|59.6/ 59.0(+0.6)|29.7/ 30.0(-0.3)|37.1/ 41.7(-4.6)|
|QWEN7B|54.8/ 56.8(-2.0)|76.3/ 73.1(+3.2)|52.0/ 51.7(+0.3)|72.0/ 72.5(-0.5)|41.8/ 48.1(-6.3)|47.9/ 51.6(-3.7)|
|QWEN14B|66.1/ 65.9(+0.2)|77.3/ 79.5(-2.2)|60.3/ 61.3(-1.0)|74.8/ 74.7(+0.1)|59.3/ 60.3(-1.0)|50.4/ 52.9(-2.5)|
|QWEN72B|76.5/ 77.2(-0.7)|86.5/ 85.3(+1.2)|67.2/ 67.6(-0.4)|80.7/ 80.8(-0.1)|76.3/ 75.5(+0.8)|55.0/ 56.4(-1.4)|
|YI6B|62.6/ 63.2(-0.6)|78.1/ 76.8(+1.3)|56.6/ 57.4(-0.8)|72.9/ 72.2(+0.7)|39.0/ 40.6(-1.6)|44.2/ 50.1(-5.9)|
|YI34B|73.5/ 74.9(-1.4)|83.6/ 84.6(-1.0)|65.6/ 64.9(+0.7)|78.9/ 80.2(-1.3)|65.5/ 65.0(+0.5)|51.9/ 55.4(-3.5)|
|GEMMA2B|36.8/ 36.9(-0.1)|57.1/ 55.2(+1.9)|43.0/ 43.3(-0.3)|60.5/ 61.5(-1.0)|10.8/ 11.1(-0.3)|40.4/ 45.8(-5.4)|
|GEMMA7B|51.8/ 51.7(+0.1)|46.5/ 44.9(+1.6)|51.7/ 51.5(+0.2)|66.6/ 66.5(+0.1)|31.3/ 32.0(-0.7)|44.7/ 47.1(-2.4)|
|LLAMA-2 7B|46.8/ 47.5(-0.7)|76.8/ 77.6(-0.8)|53.0/ 53.7(-0.7)|71.7/ 72.6(-0.9)|22.7/ 23.1(-0.4)|41.6/ 45.3(-3.7)|
|LLAMA-2 13B|53.6/ 53.6(+0.0)|82.3/ 83.2(-0.9)|60.4/ 60.3(+0.1)|73.4/ 74.3(-0.9)|35.3/ 35.6(-0.3)|42.6/ 44.0(-1.4)|
|LLAMA-2 70B|63.1/ 63.0(+0.1)|84.8/ 84.8(+0.0)|65.2/ 65.4(-0.2)|79.7/ 80.2(-0.5)|54.5/ 53.0(+1.5)|51.8/ 52.8(-1.0)|
|LLAMA-3 8B|65.0/ 65.8(-0.8)|79.6/ 82.1(-2.5)|62.3/ 62.4(-0.1)|75.9/ 75.5(+0.4)|74.3/ 75.9(-1.6)|48.3/ 51.7(-3.4)|
|LLAMA-3 70B|79.8/ 79.9(-0.1)|85.4/ 86.1(-0.7)|71.5/ 71.8(-0.3)|83.4/ 83.6(-0.2)|90.8/ 91.2(-0.4)|59.5/ 61.8(-2.3)|



Except on TRUTHFULQA, orthogonalization has a very small effect on general performance benchmarks. We observe less than 1% performance drop on average, with the difference to the baseline performance being indistinguishable from noise in most cases. The main exceptions are QWEN 7B, which has statistically significant drops on MMLU and GSM8K, and YI 34B with drops on MMLU and WINOGRANDE. 

For MMLU, we use the default settings from

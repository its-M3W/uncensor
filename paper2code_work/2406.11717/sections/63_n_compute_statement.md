# **N Compute statement**

Most experiments presented in this paper were run on a cluster of eight NVIDIA RTX A6000 GPUs with 48GB of memory. All experiments on models with _≤_ 14B parameters are run using a single 48GB memory GPU. For larger models, we use four 48BG memory GPUs in parallel. 

Generating and selecting the directions, as described in §2.3, takes approximately 5 minutes for smaller models of size _≤_ 14B, and approximately 1 hour for the larger models. 

40

# **5 Mechanistic analysis of adversarial suffixes**

Safety fine-tuned chat models are vulnerable to _adversarial suffix_ attacks (Zou et al., 2023b): there exist carefully constructed strings such that appending these strings to the end of a harmful instruction bypasses refusal and elicits harmful content. Effective adversarial suffixes are usually not human interpretable, and the mechanisms by which they work are not well understood. In this section, we mechanistically analyze the effect of an adversarial suffix on QWEN 1.8B CHAT.

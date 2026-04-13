# **C.3 Chat templates**

We use the default chat template for each model family. All chat templates are displayed in Table 6. 

21 

Table 5: Direction selection details for each model. Note that _i[∗]_ = _−_ 1 indicates that the direction is selected from the last token position, _i[∗]_ = _−_ 2 the second-to-last token position, and so on. Also note that the layer index _l[∗]_ starts from index 0, while _L_ indicates the total number of layers. 

|Chat model|_i∗_|_l∗/L_|`bypass_score`|`induce_score`|`kl_score`|
|---|---|---|---|---|---|
|QWEN1.8B|_−_1|15_/_24|_−_4_._415|1_._641|0_._077|
|QWEN7B|_−_1|17_/_32|_−_5_._355|1_._107|0_._069|
|QWEN14B|_−_1|23_/_40|_−_5_._085|1_._606|0_._014|
|QWEN72B|_−_1|62_/_80|_−_4_._246|1_._885|0_._034|
|YI6B|_−_5|20_/_32|_−_6_._693|1_._968|0_._046|
|YI34B|_−_1|37_/_60|_−_11_._14|1_._865|0_._069|
|GEMMA2B|_−_2|10_/_18|_−_14_._435|6_._709|0_._067|
|GEMMA7B|_−_1|14_/_28|_−_12_._239|6_._851|0_._091|
|LLAMA-2 7B|_−_1|14_/_32|_−_5_._295|5_._941|0_._073|
|LLAMA-2 13B|_−_1|26_/_40|_−_4_._377|2_._794|0_._092|
|LLAMA-2 70B|_−_1|21_/_80|_−_4_._565|5_._191|0_._036|
|LLAMA-3 8B|_−_5|12_/_32|_−_9_._715|7_._681|0_._064|
|LLAMA-3 70B|_−_5|25_/_80|_−_7_._839|0_._126|0_._021|



**==> picture [392 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
Bypass score Induce score<br>10.0 10<br>7.5<br>5.0 5<br>2.5<br>0<br>0.0<br>2.5 5<br>Source position Source position<br>5.0 pos -5: '<|eot _ id|>' pos -5: '<|eot_id|>'<br>pos -4: '<|start_header_id|>' pos -4: '<|start_header_id|>'<br>7.5 pos -3: 'assistant' 10 pos -3: 'assistant'<br>10.0 pos -2: '<|end_header_id|>' pos -1:  ' \n\n ' pos -2: '<|end_header_id|>' pos -1: '\n\n'<br>0 5 10 15 20 25 30 0 5 10 15 20 25 30<br>Source layer Source layer<br>Refusal score Refusal score<br>**----- End of picture text -----**<br>


Figure 11: The `bypass_score` (left) and `induce_score` (right) of each candidate direction for LLAMA-3 8B INSTRUCT. Each candidate direction **r**[(] _i[l]_[)] corresponds to source position _i_ and source layer _l_ . 

Table 6: Model families and their corresponding chat templates. The user instruction is denoted as `{x}` . Post-instruction tokens, as defined in §2.1, are labeled in red. 

|Model family|Corresponding refusal phrases|
|---|---|
|QWENCHAT|`"<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"`|
|GEMMAIT|`"<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n"`|
|YICHAT|`"<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"`|
|LLAMA-2 CHAT|`"[INST] {x}[/INST] "`|
|LLAMA-3 INSTRUCT|`"<|start_header_id|>user<|end_header_id|>\n\n`<br>`{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"`|



22

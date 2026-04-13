# Table 6: Model families and their corresponding chat templates. The user instruction is denoted as `{x}` . Post-instruction tokens, as defined in §2.1, are labeled in red.

|Model family|Corresponding refusal phrases|
|---|---|
|QWENCHAT|`"<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"`|
|GEMMAIT|`"<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n"`|
|YICHAT|`"<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"`|
|LLAMA-2 CHAT|`"[INST] {x}[/INST] "`|
|LLAMA-3 INSTRUCT|`"<|start_header_id|>user<|end_header_id|>\n\n`<br>`{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"`|



22

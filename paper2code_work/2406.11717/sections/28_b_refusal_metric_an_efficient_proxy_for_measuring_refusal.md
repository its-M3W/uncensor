# **B Refusal metric: an efficient proxy for measuring refusal**

Evaluating whether a model refuses a particular instruction is most accurately done by generating a full completion using greedy decoding, and then assessing whether the generated text constitutes a refusal. However, this process can be computationally expensive, especially when working with large models and a large number of instructions. To address this, we define a more efficient proxy for estimating the likelihood of a model refusing a given instruction without requiring generation (Figure 10). 

We observe that each model tends to have a small set of characteristic phrases that it typically uses to begin its refusals (Figure 9). This allows us to approximate the probability of refusal by examining the model’s next token probability distribution at the last token position, which corresponds to the start of its completion. 

Formally, for each model, we define a set of refusal tokens _R ⊆V_ , which contains the tokens most likely to begin the model’s refusals (Table 4). We can then estimate the probability of refusal _P_ refusal as the sum of the probabilities assigned to the tokens in _R_ . Given a vector of next token probabilities **p** = ( _p_ 1 _, p_ 2 _, . . . , p|V|_ ) _∈_ R _[|V|]_ , we define 

**==> picture [242 x 23] intentionally omitted <==**

To create a more informative “refusal metric”, we take the log-odds of _P_ refusal. This transformation helps to better distinguish between extreme probabilities that are close to 0 or 1 (Wang et al., 2023). 

**==> picture [321 x 11] intentionally omitted <==**

**==> picture [228 x 64] intentionally omitted <==**

We use this metric to filter out instructions in our test and validation datasets: for harmful instructions, we filter out prompts yielding `refusal_metric` _<_ 0, and for harmless instruction, we filter out prompts yielding `refusal_metric` _>_ 0. We also use this metric to quickly evaluate the efficacy of interventions over the validation set (§C). 

**==> picture [392 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
Next token probability across harmful data Next token probability across harmless data<br>'I'(235285) 'Sure'(21404)<br>'**'(688) '**'(688)<br>'##'(1620) 'The'(651)<br>'Pyramid'(181096) '##'(1620)<br>'Sub'(3351) '1'(235274)<br>'Modifying'(163987) '```'(1917)<br>'As'(2169) 'I'(235285)<br>'The'(651) 'A'(235280)<br>'Tax'(17790) 'One'(4038)<br>'Black'(9737) 'If'(2495)<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.1 0.2 0.3 0.4<br>Next token probability (mean) Next token probability (mean)<br>**----- End of picture text -----**<br>


(a) Top-10 token probabilities for GEMMA 2B IT, av(b) Top-10 token probabilities for GEMMA 2B IT, averaged over 128 harmful instructions. eraged over 128 harmless instructions. 

Figure 9: Next token probabilities for GEMMA 2B IT across harmful and harmless instructions. 

19 

**==> picture [238 x 144] intentionally omitted <==**

**----- Start of picture text -----**<br>
Refusal metric<br>harmful<br>30 harmless<br>20<br>10<br>0<br>15 10 5 0 5 10<br>Refusal metric<br>Frequency<br>**----- End of picture text -----**<br>


Figure 10: The refusal metric separates harmful and harmless instructions for GEMMA 2B IT. Refusals generally begin with token 234285 (corresponding to `‘I’` ). Setting _R_ GEMMA = _{_ 234285 _}_ yields a refusal metric that is an efficient proxy for assessing whether the model will refuse. 

Table 4: The refusal token set _R_ that we use for each model family, along with the refusal phrases corresponding to each token. 

|Model family|Refusal token set_R_|Corresponding refusal phrases|
|---|---|---|
|QWENCHAT|{40_,_2121}|{`"I’m sorry"`,`"As an AI"`}|
|GEMMAIT|{235285}|{`"I cannot"`}|
|YICHAT|{59597}|{`"I’m sorry"`}|
|LLAMA-2 CHAT|{306}|{`"I cannot"`}|
|LLAMA-3 INSTRUCT|{40}|{`"I cannot"`}|



20

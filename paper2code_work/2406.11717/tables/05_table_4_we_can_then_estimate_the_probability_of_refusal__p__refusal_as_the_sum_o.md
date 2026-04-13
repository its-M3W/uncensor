# Table 4). We can then estimate the probability of refusal _P_ refusal as the sum of the probabilities assigned to the tokens in _R_ . Given a vector of next token probabilities **p** = ( _p_ 1 _, p_ 2 _, . . . , p|V|_ ) _∈_ R _[|V|]_ , we define

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


Figure 10: The refusal metric separates harmful and harmless instructions for GEMMA 2B IT. Refusals generally begin with token 

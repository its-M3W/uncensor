# **D.3 Challenges of evaluating refusal**

Assessing whether a completion constitutes a successful jailbreak is complex. In this subsection, we highlight scenarios that are ambiguous, motivating our use of two metrics. 

Figure 14 and Figure 15 display cases in which the model does not explicitly refuse, but also does not provide a harmful response. In these scenarios, `refusal_score=0` while `safety_score=1` . 

Figure 16 displays a case in which the model initially refuses, but then goes on to give a harmful response. In these scenarios, `refusal_score=1` while `safety_score=0` .

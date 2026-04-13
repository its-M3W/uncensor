# Table 9: Model performance as measured by CE loss across different datasets.

|Chat model|CE Loss (THEPILE)<br>Baseline<br>Ablation<br>Act Add|CE Loss (ALPACA)<br>Baseline<br>Ablation<br>Act Add|CE Loss (On-distribution)<br>Baseline<br>Ablation<br>Act Add|
|---|---|---|---|
|QWEN1.8B<br>QWEN7B<br>QWEN14B<br>QWEN72B<br>YI6B<br>YI34B<br>GEMMA2B<br>GEMMA7B<br>LLAMA-2 7B<br>LLAMA-2 13B<br>LLAMA-2 70B<br>LLAMA-3 8B<br>LLAMA-3 70B|2.921<br>2.938<br>3.259<br>2.259<br>2.277<br>2.388<br>2.070<br>2.078<br>2.230<br>1.944<br>1.971<br>2.097<br>2.019<br>2.017<br>2.205<br>1.862<br>1.872<br>2.002<br>3.506<br>3.489<br>3.739<br>5.975<br>5.963<br>6.051<br>2.220<br>2.214<br>2.333<br>2.082<br>2.087<br>2.325<br>1.970<br>1.969<br>2.010<br>2.348<br>2.362<br>2.469<br>2.121<br>2.117<br>2.274|1.779<br>1.784<br>2.038<br>1.615<br>1.631<br>1.697<br>1.602<br>1.606<br>1.713<br>1.740<br>1.768<br>2.124<br>1.889<br>1.882<br>2.078<br>1.971<br>2.008<br>2.066<br>2.090<br>2.101<br>2.179<br>2.336<br>2.335<br>2.356<br>1.609<br>1.586<br>1.584<br>1.563<br>1.591<br>1.642<br>1.657<br>1.659<br>1.630<br>1.912<br>1.944<br>1.912<br>1.980<br>1.978<br>1.928|0.284<br>0.293<br>0.586<br>0.242<br>0.278<br>0.479<br>0.212<br>0.218<br>0.443<br>0.147<br>0.162<br>0.380<br>0.277<br>0.311<br>0.731<br>0.191<br>0.259<br>0.680<br>0.254<br>0.311<br>0.853<br>0.201<br>0.228<br>0.656<br>0.118<br>0.126<br>0.460<br>0.102<br>0.116<br>0.336<br>0.067<br>0.070<br>0.169<br>0.195<br>0.213<br>0.441<br>0.116<br>0.126<br>0.265|



All CE loss values are reported in

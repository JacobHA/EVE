Welcome to the repository for the paper "Maximum Entropy Exploration Without the Rollouts".

[Figure 1](/hyperparameter_sweeps/algorithm_comparison_plot.png)

Figure 1 in the paper (shown above) compares the convergence of various exploration algorithms, based on distributions estimated from rollouts (discounted and average-reward) and Hazan et al.'s MaxEnt method. EVE, the proposed algorithm, is shown to converge faster and to a better solution than the baselines. 

Please see the tabular/ folder for the code that generated these results.
A single run on the desired environment can be executed with
```python tabular/uonly_agent_MB.py
```


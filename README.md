
# MuZero
- A tabular implementation of [MuZero](https://arxiv.org/pdf/1911.08265.pdf) compatible with [gym](https://github.com/openai/gym).
  - To elaborate on that: the current implementation uses the policy iteration update rule of MuZero (via MCTS), but instead of using function approximation (e.g., a neural network) for the learned model, a simple table is used.
- I'm planning to add in the future a version with (i) a neural network model and (ii) distributed training.
  
 # Getting Started
 - To be added.
 
 # Results
 - The test environment is a maze, see [maze.py](https://github.com/wulfebw/muzero/blob/master/muzero/envs/maze.py) for details.
 - Here's a graph showing average discounted return during training: 
   - Averaged across 10 training runs
   - Standard deviation across runs shown in lighter color
   - The algorithm doesn't quite reach optimal performance. This might be because the exploration doesn't reach zero, because it learns a suboptimal value / policy for some of the initial states due to a lack of exploration, or because there's a bug
  
![Average Discounted Return](/media/average_discounted_return_larger.png)
 
 - The performance depends a fair amount on the number of Monte Carlo simulations run in choosing the action each timestep:

![Monte Carlo Simulations](/media/average_discounted_return_mc.png)

 - Here's a gif showing the state-value function of the learned model during training:
   - Dark red is the largest value and dark blue is the lowest value
   - The blue state in the upper right is a "pit" with -1 reward
   - The goal state is the cell one step up and left from the bottom right corner
   - Some state values near the pit and the exit don't converage because the agent doesn't visit them enough
 
![State Value](/media/value_function.gif)
 
 # Implementation
- The implementation of the algorithm is based on implementations of other algorithms in [rllab](https://github.com/rll/rllab/tree/master/rllab), which is a predecessor of [baselines](https://github.com/openai/baselines).
  - A simpler implementation for the tabular case is definitely possible. 
    - The hope is that this approach will transfer to the function approximation case where distributed / parallel sampling is necessary.
- Here's a description of the types of classes:
  - Trainer: Orchestrates the training loop (environment interaction and learning)
  - Agents: The actual algorithms (e.g., [TabularMuZero](/muzero/rl/tabular_muzero.py))
  - Samplers: Objects that sample rollouts / trajectories from an environment when passed an agent
  - Datasets: Convert trajectories into a form useful for learning
  - Planners: Online planning algorithms used by the agents (e.g., [mcts](/muzero/planning/mcts.py)), or offline planning algorithms used for comparison (e.g., [value iteration](/muzero/planning/value_iteration.py))
  
# Additional Notes
- [Why would MuZero be better than DQN?](https://wulfebw.github.io/blog/rl/muzero/muzero)

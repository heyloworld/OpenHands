# Q-Learning for Gridworld

This project implements a Q-learning algorithm to solve maze-style Gridworld tasks. The agent learns to navigate from a start position to a goal position while avoiding obstacles.

## Project Structure

```
.
├── main.py                             # Main script to run the training
├── src/
│   ├── env.py                          # Gridworld environment implementation
│   └── train.py                        # Q-learning algorithm implementation
├── models/
│   └── saved_models/
│       └── q_learning_model.npy        # Saved Q-table
└── results/
    └── figures/
        ├── learning_curve.png          # Learning curve plot
        ├── path_visualization.png      # Visualization of paths taken
        └── path_changes.gif            # Animation of path changes during training
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Usage

To train the agent with default parameters:

```bash
python main.py
```

### Command Line Arguments

The script supports various command line arguments to customize the environment and training process:

#### Environment Parameters

- `--grid_height`: Height of the grid (default: 5)
- `--grid_width`: Width of the grid (default: 5)
- `--start_row`: Starting row position (default: 0)
- `--start_col`: Starting column position (default: 0)
- `--goal_row`: Goal row position (default: grid_height - 1)
- `--goal_col`: Goal column position (default: grid_width - 1)
- `--obstacles`: Semicolon-separated list of obstacle positions as row,col (default: '1,1;2,1;3,1;1,3;2,3;3,3')

#### Training Parameters

- `--num_episodes`: Number of episodes to train for (default: 500)
- `--max_steps`: Maximum number of steps per episode (default: 100)
- `--learning_rate`: Learning rate (alpha) (default: 0.1)
- `--discount_factor`: Discount factor (gamma) (default: 0.99)
- `--exploration_rate`: Initial exploration rate (epsilon) (default: 1.0)
- `--exploration_decay`: Exploration decay rate (default: 0.995)
- `--min_exploration_rate`: Minimum exploration rate (default: 0.01)
- `--render_interval`: Interval for rendering the environment (default: 50)

#### Output Parameters

- `--model_path`: Path to save the trained model (default: 'models/saved_models/q_learning_model.npy')
- `--learning_curve_path`: Path to save the learning curve plot (default: 'results/figures/learning_curve.png')
- `--path_viz_path`: Path to save the path visualization (default: 'results/figures/path_visualization.png')
- `--path_anim_path`: Path to save the path animation (default: 'results/figures/path_changes.gif')

### Example

To train the agent on a 7x7 grid with a custom start and goal position:

```bash
python main.py --grid_height 7 --grid_width 7 --start_row 0 --start_col 0 --goal_row 6 --goal_col 6 --num_episodes 1000
```

## How It Works

### Gridworld Environment

The Gridworld environment is a grid-based world where an agent can move in four directions: up, down, left, and right. The goal is to navigate from a start position to a goal position while avoiding obstacles.

### Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. The agent learns by exploring the environment and updating its Q-table based on the rewards it receives.

The Q-learning update rule is:

Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]

where:
- Q(s, a) is the value of taking action a in state s
- α is the learning rate
- r is the reward
- γ is the discount factor
- max(Q(s', a')) is the maximum value of the next state s'

### Exploration vs. Exploitation

The agent uses an epsilon-greedy policy to balance exploration and exploitation:
- With probability ε, the agent chooses a random action (exploration)
- With probability 1-ε, the agent chooses the action with the highest Q-value (exploitation)

The exploration rate ε decays over time, allowing the agent to explore more in the beginning and exploit more as it learns.

## Results

After training, the agent should be able to find the optimal path from the start position to the goal position. The learning curve shows how the agent's performance improves over time, and the path visualization shows how the agent's path changes during training.

## Extending the Project

This project can be extended in several ways:
- Implement different reinforcement learning algorithms (e.g., SARSA, Deep Q-Network)
- Add more complex environments (e.g., stochastic transitions, continuous state spaces)
- Implement multi-agent reinforcement learning
- Add more visualization tools (e.g., heatmaps of state values)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
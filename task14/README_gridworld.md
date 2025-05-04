# Gridworld Q-Learning System

This project implements a Q-learning algorithm to solve maze-style Gridworld tasks. The system uses NumPy for efficient calculations and Matplotlib for visualizations.

## Project Structure

```
.
├── src/
│   ├── env.py           # Gridworld environment implementation
│   └── q_learning.py    # Q-learning algorithm implementation
├── models/
│   └── saved_models/    # Directory for saved models
│       └── q_learning_model.npy  # Trained Q-table
├── results/
│   └── figures/         # Directory for visualizations
│       ├── learning_curve.png    # Learning curve during training
│       ├── steps_per_episode.png # Steps per episode during training
│       ├── exploration_rate.png  # Exploration rate decay
│       ├── policy.png            # Visualization of the learned policy
│       ├── q_values.png          # Visualization of the learned Q-values
│       └── path_changes.gif      # Animation of path evolution (may not be generated)
└── gridworld_main.py    # Main script to run the system
```

## Features

- **Customizable Gridworld Environment**: Specify grid size, start/end positions, and wall locations
- **Q-learning Implementation**: Efficient implementation with NumPy
- **Visualization**: Learning curves, policy visualization, and path evolution
- **Real-time Feedback**: Progress updates during training
- **Extensible Design**: Easy to modify or extend

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- tqdm (for progress bars)

## Usage

### Basic Usage

```bash
python gridworld_main.py
```

This will train a Q-learning agent on a default 8x8 Gridworld with 10 random walls.

### Custom Configuration

```bash
python gridworld_main.py --grid_size 10 10 --num_walls 15 --num_episodes 1000
```

### Command-line Arguments

- `--grid_size`: Size of the grid as height width (default: [8, 8])
- `--start_pos`: Starting position as row col (default: [0, 0])
- `--goal_pos`: Goal position as row col (default: bottom-right corner)
- `--num_walls`: Number of random walls to add (default: 10)
- `--num_episodes`: Number of episodes to train (default: 1000)
- `--max_steps`: Maximum steps per episode (default: 100)
- `--learning_rate`: Learning rate (alpha) (default: 0.1)
- `--discount_factor`: Discount factor (gamma) (default: 0.99)
- `--exploration_rate`: Initial exploration rate (epsilon) (default: 1.0)
- `--exploration_decay`: Decay rate for exploration (default: 0.995)
- `--min_exploration_rate`: Minimum exploration rate (default: 0.01)
- `--eval_episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Render the environment during evaluation (default: False)
- `--render_interval`: Interval for rendering during training (default: None)
- `--save_model`: Path to save the trained model (default: 'models/saved_models/q_learning_model.npy')
- `--save_plots`: Directory to save the plots (default: 'results/figures')
- `--seed`: Random seed (default: 42)

## Implementation Details

### Gridworld Environment (`src/env.py`)

The Gridworld environment is implemented as a class with the following features:

- Grid representation with empty cells, walls, start, and goal positions
- Four actions: Up, Right, Down, Left
- Reward structure: +1 for reaching the goal, -1 for hitting a wall, -0.01 for each step
- Visualization methods for rendering the grid, paths, and learning curves

### Q-learning Algorithm (`src/q_learning.py`)

The Q-learning algorithm is implemented with the following components:

- Q-table initialization and update using the Q-learning update rule
- Epsilon-greedy action selection with decaying exploration rate
- Training loop with progress tracking
- Evaluation methods for testing the learned policy
- Visualization methods for the learned policy and Q-values

## Extending the System

### Adding New Features to the Environment

To add new features to the environment, modify the `Gridworld` class in `src/env.py`. For example, you could add:

- Different types of terrain with varying costs
- Stochastic transitions
- Multiple goals with different rewards

### Implementing Different RL Algorithms

To implement a different reinforcement learning algorithm, create a new file in the `src` directory (e.g., `src/sarsa.py`) and implement the algorithm using the same interface as the `QLearningAgent` class.

### Customizing Visualizations

The visualization methods in both the `Gridworld` and `QLearningAgent` classes can be modified to create different types of visualizations or to add more information to the existing ones.

## License

This project is open source and available under the MIT License.
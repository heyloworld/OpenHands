# OpenHands Projects

This repository contains three projects:

1. **GPT-2 Text Generation System**: A system for generating text using a pre-trained GPT-2 model.
2. **Q-Learning for Gridworld**: A reinforcement learning system for solving maze-style Gridworld tasks.
3. **DQN for OpenAI Gym**: A Deep Q-Network implementation for solving OpenAI Gym environments.

## DQN for OpenAI Gym

This project implements a Deep Q-Network (DQN) agent to solve OpenAI Gym environments such as CartPole-v1. The system uses PyTorch for the neural network implementation and provides comprehensive logging and visualization of the training process.

### Features

- **DQN Implementation**: A complete implementation of the DQN algorithm using PyTorch.
- **Experience Replay**: Efficient experience replay buffer for stable learning.
- **Target Network**: Separate target network for stable Q-learning updates.
- **Epsilon-Greedy Exploration**: Adaptive exploration strategy.
- **Metrics Logging**: Comprehensive logging of training metrics.
- **Visualization**: Visualization of the learning curve and training progress.
- **Error Handling**: Robust error handling for dimension mismatches and other issues.

### Project Structure

```
.
├── src/
│   ├── model.py                  # DQN model implementation
│   └── main.py                   # Main training script
├── models/
│   └── saved_models/             # Directory for saved models
│       └── dqn_model.pt          # Trained DQN model
├── results/
│   ├── figures/                  # Directory for figures
│   │   ├── return_over_time.png  # Learning curve visualization
│   │   └── dqn_metrics.png       # Metrics visualization
│   └── metrics/                  # Directory for metrics
│       └── dqn_metrics.json      # Training metrics
├── train_dqn.py                  # Script to run the training
├── test_dqn.py                   # Script to test the trained model
├── visualize_metrics.py          # Script to visualize the metrics
└── README.md                     # This file
```

### Requirements

- Python 3.6+
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib
- tqdm

### Usage

#### Training

To train the DQN agent on the CartPole-v1 environment, run:

```bash
python train_dqn.py
```

This will train the agent for 1000 episodes and save the model to `models/saved_models/dqn_model.pt`. It will also generate visualizations of the learning curve and log the metrics to `results/metrics/dqn_metrics.json`.

#### Testing

To test the trained model, run:

```bash
python test_dqn.py
```

This will run 10 episodes with the trained model and display the results.

#### Visualization

To visualize the training metrics, run:

```bash
python visualize_metrics.py
```

This will generate a visualization of the training metrics and save it to `results/figures/dqn_metrics.png`.

### Customization

You can customize the system by modifying the following parameters in `src/main.py`:

#### Training Parameters

- `env_name`: The name of the OpenAI Gym environment.
- `n_episodes`: The number of episodes to train for.
- `max_t`: The maximum number of timesteps per episode.
- `eps_start`: The starting value of epsilon for epsilon-greedy action selection.
- `eps_end`: The minimum value of epsilon.
- `eps_decay`: The multiplicative factor for decreasing epsilon.

#### DQN Parameters

- `state_size`: The dimension of the state space.
- `action_size`: The dimension of the action space.
- `buffer_size`: The size of the replay buffer.
- `batch_size`: The size of the batch for learning.
- `gamma`: The discount factor.
- `tau`: The soft update parameter.
- `lr`: The learning rate.
- `update_every`: How often to update the network.

### Results

The system generates the following results:

- **Learning Curve**: A plot of the episode returns during training.
- **Metrics Visualization**: A visualization of the training metrics.
- **Trained Model**: The trained DQN model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
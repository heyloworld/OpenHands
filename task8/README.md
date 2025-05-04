# Robotic Arm Control with Reinforcement Learning

This project implements a robotic arm control system using Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm in the PyBullet simulation environment.

## Project Structure

```
.
├── data/                       # Data directory
│   └── final_position.txt      # Final position of the robot arm
├── models/                     # Trained models
├── results/                    # Results directory
│   └── figures/                # Figures and visualizations
│       ├── training_returns.png # Training returns plot
│       └── robot_motion.gif    # Visualization of robot motion
├── src/                        # Source code
│   ├── env.py                  # Robot arm environment implementation
│   └── train.py                # PPO training implementation
└── main.py                     # Main script to run training and evaluation
```

## Environment

The environment (`src/env.py`) simulates a robotic arm with the goal of reaching a target position. It uses the Kuka IIWA robotic arm model from PyBullet.

### Observation Space
- Joint positions (7 values)
- Joint velocities (7 values)
- End effector position (3 values)
- Target position (3 values)

### Action Space
- Joint position control for each joint (7 values)

### Reward Structure
- **Dense Reward**: Negative distance between end effector and target, with a success bonus and energy penalty
- **Sparse Reward**: Binary reward (1 for success, 0 for failure)

### Episode Termination
- When the end effector reaches the target (success)
- When the maximum number of steps is reached (timeout)

## Training Algorithm

The training algorithm (`src/train.py`) implements the PPO algorithm from the Stable Baselines 3 library. It includes:

- Custom callback for tracking training progress
- Evaluation during training
- Visualization of training returns
- Saving of the trained model

## Usage

### Installation

```bash
# Install dependencies
pip install pybullet stable-baselines3 gymnasium numpy matplotlib imageio
```

### Training

```bash
# Train with default parameters
python main.py

# Train with custom parameters
python main.py --total-timesteps 200000 --learning-rate 1e-4 --gui --render
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --eval-only --model-path models/ppo_robot_arm --gui --render --save-gif
```

### Command-line Arguments

#### Environment Parameters
- `--gui`: Use GUI for rendering
- `--render`: Render the environment during evaluation
- `--fixed-target`: Use a fixed target position
- `--max-steps`: Maximum number of steps per episode
- `--distance-threshold`: Distance threshold for success
- `--action-scale`: Scaling factor for actions
- `--reward-type`: Type of reward function ("dense" or "sparse")
- `--urdf-path`: Path to robot URDF file
- `--record-video`: Record video frames

#### Training Parameters
- `--eval-only`: Only evaluate, don't train
- `--total-timesteps`: Total number of timesteps to train for
- `--eval-freq`: Frequency of evaluation during training
- `--n-eval-episodes`: Number of episodes for evaluation
- `--learning-rate`: Learning rate
- `--n-steps`: Number of steps per update
- `--batch-size`: Minibatch size
- `--n-epochs`: Number of epochs per update
- `--gamma`: Discount factor
- `--gae-lambda`: GAE lambda parameter
- `--clip-range`: PPO clip range

#### Output Parameters
- `--model-path`: Path to save/load the model
- `--log-path`: Path to save training metrics
- `--training-curve-path`: Path to save training curve
- `--save-gif`: Save a GIF of the robot motion
- `--gif-path`: Path to save the GIF
- `--final-position-path`: Path to save the final position

## Results

After training, the system generates:

1. A plot of the training returns (`results/figures/training_returns.png`)
2. A GIF visualization of the robot's motion (`results/figures/robot_motion.gif`)
3. A text file with the final position of the robot arm (`data/final_position.txt`)

## Handling URDF Files

The environment is designed to handle URDF files for robot models. By default, it uses the Kuka IIWA model from PyBullet. You can specify a custom URDF file using the `--urdf-path` argument.

If there are issues loading the URDF file, the environment will provide clear error messages and logging for debugging.

## Extending the Project

To extend this project:

1. **Custom Robot Models**: Add your own robot URDF files and specify them using the `--urdf-path` argument
2. **Custom Reward Functions**: Modify the `_compute_reward` method in `src/env.py`
3. **Different RL Algorithms**: Replace PPO with other algorithms from Stable Baselines 3
4. **Multi-task Learning**: Extend the environment to support multiple tasks

## Troubleshooting

### Common Issues

1. **URDF Loading Errors**: Ensure the URDF file path is correct and the file is properly formatted
2. **Rendering Issues**: If rendering fails, try running without GUI (`--gui` flag)
3. **Training Instability**: Adjust hyperparameters like learning rate, batch size, and clip range

### Debugging

The project includes comprehensive logging to help with debugging. Check the console output for error messages and warnings.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
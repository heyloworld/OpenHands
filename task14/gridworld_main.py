"""
Main script to run the Q-learning algorithm on a Gridworld environment.
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.env import Gridworld
from src.q_learning import QLearningAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a Q-learning agent on Gridworld')
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                        help='Size of the grid as height width')
    parser.add_argument('--start_pos', type=int, nargs=2, default=[0, 0],
                        help='Starting position as row col')
    parser.add_argument('--goal_pos', type=int, nargs=2, default=None,
                        help='Goal position as row col (default: bottom-right corner)')
    parser.add_argument('--num_walls', type=int, default=10,
                        help='Number of random walls to add')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (alpha)')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor (gamma)')
    parser.add_argument('--exploration_rate', type=float, default=1.0,
                        help='Initial exploration rate (epsilon)')
    parser.add_argument('--exploration_decay', type=float, default=0.995,
                        help='Decay rate for exploration')
    parser.add_argument('--min_exploration_rate', type=float, default=0.01,
                        help='Minimum exploration rate')
    
    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--render_interval', type=int, default=None,
                        help='Interval for rendering during training')
    
    # Saving parameters
    parser.add_argument('--save_model', type=str, default='models/saved_models/q_learning_model.npy',
                        help='Path to save the trained model')
    parser.add_argument('--save_plots', type=str, default='results/figures',
                        help='Directory to save the plots')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Set goal position if not provided
    if args.goal_pos is None:
        args.goal_pos = [args.grid_size[0] - 1, args.grid_size[1] - 1]
    
    # Generate random walls
    walls = []
    if args.num_walls > 0:
        for _ in range(args.num_walls):
            wall_row = np.random.randint(0, args.grid_size[0])
            wall_col = np.random.randint(0, args.grid_size[1])
            
            # Ensure wall is not at start or goal position
            if (wall_row, wall_col) != tuple(args.start_pos) and (wall_row, wall_col) != tuple(args.goal_pos):
                walls.append((wall_row, wall_col))
    
    # Create environment
    env = Gridworld(
        grid_size=tuple(args.grid_size),
        start_pos=tuple(args.start_pos),
        goal_pos=tuple(args.goal_pos),
        walls=walls
    )
    
    # Create agent
    agent = QLearningAgent(
        env=env,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        min_exploration_rate=args.min_exploration_rate
    )
    
    # Create directories if they don't exist
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
    
    # Train agent
    agent.train(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render_interval=args.render_interval,
        save_model_path=args.save_model,
        save_plots_path=args.save_plots
    )
    
    # Skip animation generation
    agent.env.path_history = []
    
    # Visualize policy and Q-values
    if args.save_plots:
        agent.visualize_policy(os.path.join(args.save_plots, "policy.png"))
        agent.visualize_q_values(os.path.join(args.save_plots, "q_values.png"))
    
    # Evaluate agent
    agent.evaluate(
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        render=args.render
    )
    
    # Print a summary of the results
    print("\n" + "="*80)
    print("Q-Learning Training Summary")
    print("="*80)
    print(f"Grid Size: {args.grid_size}")
    print(f"Start Position: {args.start_pos}")
    print(f"Goal Position: {args.goal_pos}")
    print(f"Number of Walls: {args.num_walls}")
    print(f"Number of Episodes: {args.num_episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Discount Factor: {args.discount_factor}")
    print(f"Final Exploration Rate: {agent.exploration_rate:.4f}")
    print(f"Final Average Reward: {np.mean(agent.episode_rewards[-10:]):.4f}")
    print(f"Final Average Steps: {np.mean(agent.episode_steps[-10:]):.4f}")
    print("="*80)
    
    return agent

if __name__ == "__main__":
    main()
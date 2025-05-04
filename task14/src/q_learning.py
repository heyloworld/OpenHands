"""
Q-learning implementation for solving Gridworld.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import time
from src.env import Gridworld

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QLearningAgent:
    """
    Q-learning agent for solving Gridworld.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            env (Gridworld): Gridworld environment
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            exploration_rate (float): Initial exploration rate (epsilon)
            exploration_decay (float): Decay rate for exploration
            min_exploration_rate (float): Minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.height * env.width, env.action_space))
        
        # Track training metrics
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_exploration_rates = []
        
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (tuple): Current state as (row, col)
            
        Returns:
            int: Chosen action
        """
        state_rep = self.env.get_state_representation(state)
        
        # Explore: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.env.action_space)
        
        # Exploit: choose the best action
        return np.argmax(self.q_table[state_rep])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning update rule.
        
        Args:
            state (tuple): Current state as (row, col)
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): Next state as (row, col)
        """
        state_rep = self.env.get_state_representation(state)
        next_state_rep = self.env.get_state_representation(next_state)
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state_rep])
        td_target = reward + self.discount_factor * self.q_table[next_state_rep, best_next_action]
        td_error = td_target - self.q_table[state_rep, action]
        
        self.q_table[state_rep, action] += self.learning_rate * td_error
    
    def decay_exploration_rate(self):
        """
        Decay the exploration rate.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def train(self, num_episodes=1000, max_steps=100, render_interval=None, 
              save_model_path=None, save_plots_path=None):
        """
        Train the agent using Q-learning.
        
        Args:
            num_episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
            render_interval (int): Interval for rendering (None for no rendering)
            save_model_path (str): Path to save the trained model
            save_plots_path (str): Path to save the plots
            
        Returns:
            tuple: (q_table, episode_rewards, episode_steps)
        """
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        # Create directories if they don't exist
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        if save_plots_path:
            os.makedirs(os.path.dirname(save_plots_path), exist_ok=True)
        
        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action = self.choose_action(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Render if requested
                if render_interval and episode % render_interval == 0 and step == 0:
                    self.env.render()
                
                # Break if done
                if done:
                    break
            
            # Decay exploration rate
            self.decay_exploration_rate()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)
            self.episode_exploration_rates.append(self.exploration_rate)
            
            # Store path for visualization
            self.env.path_history.append(self.env.current_path.copy())
            
            # Log progress
            if episode % 100 == 0 or episode == num_episodes - 1:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                           f"Steps = {episode_steps}, "
                           f"Exploration rate = {self.exploration_rate:.4f}")
        
        # Save model if requested
        if save_model_path:
            np.save(save_model_path, self.q_table)
            logger.info(f"Model saved to {save_model_path}")
        
        # Save plots if requested
        if save_plots_path:
            self.save_training_plots(save_plots_path)
            
            # Skip path evolution animation as it's causing issues
            # path_gif_path = os.path.join(os.path.dirname(save_plots_path), "path_changes.gif")
            # self.env.save_path_history(path_gif_path)
            # logger.info(f"Path evolution saved to {path_gif_path}")
            logger.info("Skipping path evolution animation")
        
        logger.info("Training completed!")
        
        return self.q_table, self.episode_rewards, self.episode_steps
    
    def save_training_plots(self, save_dir):
        """
        Save training plots.
        
        Args:
            save_dir (str): Directory to save the plots
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot and save learning curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.episode_rewards, label='Episode Reward')
        
        # Add moving average
        window_size = min(100, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size-1, len(self.episode_rewards)), 
                   moving_avg, 'r-', label=f'Moving Average ({window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
        plt.close(fig)
        
        # Plot and save steps per episode
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.episode_steps, label='Steps per Episode')
        
        # Add moving average
        window_size = min(100, len(self.episode_steps))
        if window_size > 0:
            moving_avg = np.convolve(self.episode_steps, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size-1, len(self.episode_steps)), 
                   moving_avg, 'r-', label=f'Moving Average ({window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(os.path.join(save_dir, "steps_per_episode.png"))
        plt.close(fig)
        
        # Plot and save exploration rate
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.episode_exploration_rates, label='Exploration Rate')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Rate')
        ax.set_title('Exploration Rate Decay')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(os.path.join(save_dir, "exploration_rate.png"))
        plt.close(fig)
        
        logger.info(f"Training plots saved to {save_dir}")
    
    def visualize_policy(self, save_path=None):
        """
        Visualize the learned policy.
        
        Args:
            save_path (str): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create policy matrix
        policy = np.zeros((self.env.height, self.env.width))
        
        for i in range(self.env.height):
            for j in range(self.env.width):
                # Skip walls
                if self.env.grid[i, j] == 1:
                    continue
                
                # Skip goal
                if (i, j) == self.env.goal_pos:
                    continue
                
                state_rep = self.env.get_state_representation((i, j))
                policy[i, j] = np.argmax(self.q_table[state_rep])
        
        # Visualize policy
        fig = self.env.render(mode='rgb_array', show_policy=policy)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        
        return fig
    
    def visualize_q_values(self, save_path=None):
        """
        Visualize the Q-values.
        
        Args:
            save_path (str): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create Q-value matrix
        q_values = np.zeros((self.env.height, self.env.width, self.env.action_space))
        
        for i in range(self.env.height):
            for j in range(self.env.width):
                state_rep = self.env.get_state_representation((i, j))
                q_values[i, j] = self.q_table[state_rep]
        
        # Visualize Q-values
        fig = self.env.render(mode='rgb_array', show_values=q_values)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        
        return fig
    
    def evaluate(self, num_episodes=10, max_steps=100, render=False):
        """
        Evaluate the agent.
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            max_steps (int): Maximum steps per episode
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (average_reward, average_steps, success_rate)
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes...")
        
        total_rewards = []
        total_steps = []
        successes = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # Choose best action (no exploration)
                state_rep = self.env.get_state_representation(state)
                action = np.argmax(self.q_table[state_rep])
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.1)  # Slow down rendering
                
                # Break if done
                if done:
                    successes += 1
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            
            logger.info(f"Evaluation Episode {episode}: Reward = {episode_reward:.2f}, "
                       f"Steps = {episode_steps}, "
                       f"Success = {done}")
        
        average_reward = np.mean(total_rewards)
        average_steps = np.mean(total_steps)
        success_rate = successes / num_episodes
        
        logger.info(f"Evaluation Results: Average Reward = {average_reward:.2f}, "
                   f"Average Steps = {average_steps:.2f}, "
                   f"Success Rate = {success_rate:.2f}")
        
        return average_reward, average_steps, success_rate


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
    
    # Train agent
    agent.train(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render_interval=args.render_interval,
        save_model_path=args.save_model,
        save_plots_path=args.save_plots
    )
    
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


if __name__ == "__main__":
    main()
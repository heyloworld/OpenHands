import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time
from src.env import Gridworld

class QLearning:
    """
    Q-Learning algorithm implementation for solving Gridworld tasks.
    
    This class implements the Q-learning algorithm, a model-free reinforcement
    learning technique that learns the value of an action in a particular state.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            env: The environment to interact with
            learning_rate (float): The learning rate (alpha)
            discount_factor (float): The discount factor (gamma)
            exploration_rate (float): The initial exploration rate (epsilon)
            exploration_decay (float): The decay rate for exploration
            min_exploration_rate (float): The minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        # Shape: (height, width, action_space)
        self.q_table = np.zeros((env.height, env.width, env.action_space))
        
        # Track learning progress
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (tuple): The current state (row, col)
            
        Returns:
            int: The chosen action
        """
        # Explore: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.env.action_space)
        
        # Exploit: choose the best action
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.
        
        Args:
            state (tuple): The current state
            action (int): The action taken
            reward (float): The reward received
            next_state (tuple): The next state
        """
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay_exploration(self):
        """
        Decay the exploration rate.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def train(self, num_episodes=1000, max_steps=100, render_interval=100, verbose=True):
        """
        Train the agent using Q-learning.
        
        Args:
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum number of steps per episode
            render_interval (int): Interval for rendering the environment
            verbose (bool): Whether to print progress information
            
        Returns:
            tuple: (episode_rewards, episode_lengths)
        """
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Reset the environment
            state = self.env.reset()
            total_reward = 0
            
            for step in range(1, max_steps + 1):
                # Choose an action
                action = self.choose_action(state)
                
                # Take the action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update the Q-table
                self.update_q_table(state, action, reward, next_state)
                
                # Update the state and total reward
                state = next_state
                total_reward += reward
                
                # Check if the episode is done
                if done:
                    break
            
            # Decay the exploration rate
            self.decay_exploration()
            
            # Track the episode reward and length
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            
            # Print progress information
            if verbose and episode % render_interval == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-render_interval:])
                avg_length = np.mean(self.episode_lengths[-render_interval:])
                
                print(f"Episode: {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Exploration Rate: {self.exploration_rate:.2f} | "
                      f"Elapsed Time: {elapsed_time:.2f}s")
                
                # Render the environment
                if episode % (render_interval * 10) == 0:
                    self.env.render()
        
        return self.episode_rewards, self.episode_lengths
    
    def save_model(self, filepath):
        """
        Save the Q-table to a file.
        
        Args:
            filepath (str): The path to save the Q-table to
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the Q-table
        np.save(filepath, self.q_table)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the Q-table from a file.
        
        Args:
            filepath (str): The path to load the Q-table from
        """
        self.q_table = np.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_learning_curve(self, filepath=None, window_size=10):
        """
        Plot the learning curve.
        
        Args:
            filepath (str): The path to save the plot to
            window_size (int): The window size for smoothing the curve
        """
        # Create the directory if it doesn't exist
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Compute moving averages
        smoothed_rewards = np.convolve(self.episode_rewards, 
                                      np.ones(window_size) / window_size, 
                                      mode='valid')
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards, 
                 alpha=0.3, label='Raw Rewards')
        plt.plot(range(window_size - 1, len(self.episode_rewards)), 
                 smoothed_rewards, label=f'Smoothed Rewards (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if filepath:
            plt.savefig(filepath)
            print(f"Learning curve saved to {filepath}")
        else:
            plt.show()
    
    def visualize_path_changes(self, filepath=None, num_episodes=5, interval=200):
        """
        Visualize how the agent's path changes during training.
        
        Args:
            filepath (str): The path to save the animation to
            num_episodes (int): Number of episodes to visualize
            interval (int): Interval between frames in milliseconds
        """
        # Create the directory if it doesn't exist
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get the episode history
        episode_history = self.env.get_episode_history()
        
        # Select episodes to visualize
        if len(episode_history) <= num_episodes:
            selected_episodes = episode_history
        else:
            # Select episodes evenly spaced throughout training
            indices = np.linspace(0, len(episode_history) - 1, num_episodes, dtype=int)
            selected_episodes = [episode_history[i] for i in indices]
        
        # Create a figure for the animation
        fig, axes = plt.subplots(1, len(selected_episodes), figsize=(15, 5))
        
        # If only one episode is selected, axes will not be an array
        if len(selected_episodes) == 1:
            axes = [axes]
        
        # Initialize the plots
        images = []
        for i, (ax, episode) in enumerate(zip(axes, selected_episodes)):
            # Create a copy of the grid for rendering
            render_grid = self.env.grid.copy()
            
            # Set up the plot
            ax.set_title(f"Episode {i+1}")
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(-0.5, self.env.width, 1))
            ax.set_yticks(np.arange(-0.5, self.env.height, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Create a colormap
            cmap = plt.cm.get_cmap('viridis', len(episode))
            
            # Plot the grid
            img = ax.imshow(render_grid, cmap='binary', vmin=0, vmax=1)
            images.append(img)
            
            # Plot the start and goal positions
            ax.plot(self.env.start_pos[1], self.env.start_pos[0], 'go', markersize=10)
            ax.plot(self.env.goal_pos[1], self.env.goal_pos[0], 'ro', markersize=10)
            
            # Plot the path
            for j, (row, col) in enumerate(episode):
                ax.plot(col, row, 'o', color=cmap(j), markersize=5, alpha=0.7)
                
                # Connect the points with lines
                if j > 0:
                    prev_row, prev_col = episode[j-1]
                    ax.plot([prev_col, col], [prev_row, row], '-', 
                            color=cmap(j), linewidth=1, alpha=0.7)
        
        plt.tight_layout()
        
        # Save or show the animation
        if filepath:
            plt.savefig(filepath)
            print(f"Path visualization saved to {filepath}")
        else:
            plt.show()
    
    def create_path_animation(self, filepath=None, fps=5):
        """
        Create an animation of the agent's path changes during training.
        
        Args:
            filepath (str): The path to save the animation to
            fps (int): Frames per second
        """
        # Create the directory if it doesn't exist
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get the episode history
        episode_history = self.env.get_episode_history()
        
        if not episode_history:
            print("No episodes to animate.")
            return
        
        # Create a figure for the animation
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Create a colormap for the grid
        grid_cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        grid_bounds = [0, 1, 2, 3, 4]
        grid_norm = plt.cm.colors.BoundaryNorm(grid_bounds, grid_cmap.N)
        
        # Function to update the animation
        def update(frame):
            ax.clear()
            
            # Get the episode and step
            episode_idx = frame // len(episode_history[0])
            step_idx = frame % len(episode_history[0])
            
            if episode_idx >= len(episode_history):
                episode_idx = len(episode_history) - 1
                step_idx = len(episode_history[episode_idx]) - 1
            
            # Get the current episode
            episode = episode_history[episode_idx]
            
            if step_idx >= len(episode):
                step_idx = len(episode) - 1
            
            # Create a copy of the grid for rendering
            render_grid = self.env.grid.copy()
            
            # Plot the grid
            ax.imshow(render_grid, cmap=grid_cmap, norm=grid_norm)
            
            # Add grid lines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
            ax.set_xticks(np.arange(-0.5, self.env.width, 1))
            ax.set_yticks(np.arange(-0.5, self.env.height, 1))
            
            # Remove tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Plot the path up to the current step
            path = episode[:step_idx+1]
            
            # Create a colormap for the path
            path_cmap = plt.cm.get_cmap('viridis', len(path))
            
            # Plot the path
            for j, (row, col) in enumerate(path):
                ax.plot(col, row, 'o', color=path_cmap(j), markersize=8, alpha=0.7)
                
                # Connect the points with lines
                if j > 0:
                    prev_row, prev_col = path[j-1]
                    ax.plot([prev_col, col], [prev_row, row], '-', 
                            color=path_cmap(j), linewidth=2, alpha=0.7)
            
            # Set title
            ax.set_title(f"Episode {episode_idx+1}, Step {step_idx+1}")
            
            return ax,
        
        # Calculate the total number of frames
        total_frames = sum(len(episode) for episode in episode_history)
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)
        
        # Save or show the animation
        if filepath:
            anim.save(filepath, writer='pillow', fps=fps)
            print(f"Path animation saved to {filepath}")
        else:
            plt.show()
        
        plt.close(fig)


def main():
    """
    Main function to run the Q-learning algorithm on a Gridworld environment.
    """
    # Create the environment
    env = Gridworld(grid_size=(5, 5), 
                   start_pos=(0, 0), 
                   goal_pos=(4, 4), 
                   obstacles=[(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)])
    
    # Create the Q-learning agent
    agent = QLearning(env, 
                     learning_rate=0.1, 
                     discount_factor=0.99, 
                     exploration_rate=1.0, 
                     exploration_decay=0.995, 
                     min_exploration_rate=0.01)
    
    # Train the agent
    print("Training the agent...")
    agent.train(num_episodes=500, max_steps=100, render_interval=50, verbose=True)
    
    # Save the model
    agent.save_model("models/saved_models/q_learning_model.npy")
    
    # Plot the learning curve
    agent.plot_learning_curve("results/figures/learning_curve.png")
    
    # Visualize the path changes
    agent.visualize_path_changes("results/figures/path_visualization.png")
    
    # Create the path animation
    agent.create_path_animation("results/figures/path_changes.gif")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
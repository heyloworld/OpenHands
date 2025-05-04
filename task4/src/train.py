import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from tqdm import tqdm
from env import Gridworld

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QLearning:
    """
    Q-learning algorithm for solving Gridworld tasks.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize the Q-learning algorithm.
        
        Args:
            env (Gridworld): The Gridworld environment.
            learning_rate (float): The learning rate.
            discount_factor (float): The discount factor.
            exploration_rate (float): The exploration rate.
            exploration_decay (float): The exploration decay rate.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize the Q-table
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))
        
        logger.info(f"Q-learning initialized with learning rate {learning_rate}")
        logger.info(f"Discount factor: {discount_factor}")
        logger.info(f"Initial exploration rate: {exploration_rate}")
        logger.info(f"Exploration decay rate: {exploration_decay}")
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (tuple): The current state (x, y).
            
        Returns:
            int: The chosen action.
        """
        # Explore: choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(len(self.env.actions))
        
        # Exploit: choose the best action
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.
        
        Args:
            state (tuple): The current state (x, y).
            action (int): The chosen action.
            reward (float): The received reward.
            next_state (tuple): The next state (x, y).
        """
        # Get the best action for the next state
        best_next_action = np.argmax(self.q_table[next_state])
        
        # Calculate the TD target
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        
        # Calculate the TD error
        td_error = td_target - self.q_table[state][action]
        
        # Update the Q-value
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay_exploration(self):
        """
        Decay the exploration rate.
        """
        self.exploration_rate *= self.exploration_decay
    
    def train(self, num_episodes=1000, render_interval=100):
        """
        Train the agent.
        
        Args:
            num_episodes (int): The number of episodes to train for.
            render_interval (int): The interval at which to render the environment.
            
        Returns:
            tuple: (episode_returns, paths)
        """
        # Initialize the episode returns
        episode_returns = np.zeros(num_episodes)
        
        # Initialize the paths
        paths = []
        
        # Train for the specified number of episodes
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Reset the environment
            state = self.env.reset()
            
            # Initialize the episode return
            episode_return = 0
            
            # Run the episode
            done = False
            while not done:
                # Choose an action
                action = self.choose_action(state)
                
                # Take the action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update the Q-table
                self.update_q_table(state, action, reward, next_state)
                
                # Update the state
                state = next_state
                
                # Update the episode return
                episode_return += reward
            
            # Decay the exploration rate
            self.decay_exploration()
            
            # Store the episode return
            episode_returns[episode] = episode_return
            
            # Store the path
            paths.append(self.env.get_path())
            
            # Render the environment
            if (episode + 1) % render_interval == 0:
                logger.info(f"Episode {episode + 1}/{num_episodes}")
                logger.info(f"Return: {episode_return:.2f}")
                logger.info(f"Exploration rate: {self.exploration_rate:.4f}")
        
        return episode_returns, paths
    
    def save_model(self, model_path="models/saved_models/q_learning_model.npy"):
        """
        Save the Q-table to a file.
        
        Args:
            model_path (str): The path to save the Q-table.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the Q-table
        np.save(model_path, self.q_table)
        
        logger.info(f"Q-table saved to {model_path}")
    
    def load_model(self, model_path="models/saved_models/q_learning_model.npy"):
        """
        Load the Q-table from a file.
        
        Args:
            model_path (str): The path to load the Q-table from.
        """
        # Load the Q-table
        self.q_table = np.load(model_path)
        
        logger.info(f"Q-table loaded from {model_path}")
    
    def visualize_learning_curve(self, episode_returns, output_path="results/figures/learning_curve.png"):
        """
        Visualize the learning curve.
        
        Args:
            episode_returns (numpy.ndarray): The episode returns.
            output_path (str): The path to save the figure.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(episode_returns) + 1), episode_returns, 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Learning Curve')
        plt.grid(True)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curve saved to {output_path}")
    
    def visualize_path_changes(self, paths, output_path="results/figures/path_changes.gif"):
        """
        Visualize the path changes.
        
        Args:
            paths (list): The paths taken by the agent in each episode.
            output_path (str): The path to save the figure.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set the limits of the grid
        ax.set_xlim(0, self.env.grid_size)
        ax.set_ylim(0, self.env.grid_size)
        
        # Draw the grid
        for i in range(self.env.grid_size + 1):
            ax.axhline(i, color='black', linewidth=0.5)
            ax.axvline(i, color='black', linewidth=0.5)
        
        # Draw the start and end positions
        ax.plot(self.env.start_pos[0] + 0.5, self.env.start_pos[1] + 0.5, 'go', markersize=15, label='Start')
        ax.plot(self.env.end_pos[0] + 0.5, self.env.end_pos[1] + 0.5, 'ro', markersize=15, label='End')
        
        # Draw the obstacles
        for obs in self.env.obstacles:
            ax.add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='black'))
        
        # Create the animation
        line, = ax.plot([], [], 'b-', linewidth=2)
        points, = ax.plot([], [], 'bo', markersize=8)
        
        # Add a title with the episode number
        title = ax.set_title('Episode: 0')
        
        # Add a legend
        ax.legend(loc='upper left')
        
        def init():
            line.set_data([], [])
            points.set_data([], [])
            return line, points, title
        
        def update(frame):
            path = paths[frame]
            x = [p[0] + 0.5 for p in path]
            y = [p[1] + 0.5 for p in path]
            line.set_data(x, y)
            points.set_data(x, y)
            title.set_text(f'Episode: {frame + 1}')
            return line, points, title
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=min(len(paths), 100), init_func=init, blit=True, interval=200)
        
        # Save the animation
        anim.save(output_path, writer='pillow', fps=5)
        plt.close()
        
        logger.info(f"Path changes saved to {output_path}")

def main():
    """
    Main function to train the agent.
    """
    # Create the environment
    env = Gridworld(
        grid_size=10,
        start_pos=(0, 0),
        end_pos=(9, 9),
        obstacles=[(2, 2), (2, 3), (2, 4), (3, 6), (4, 6), (5, 6), (6, 6), (7, 2), (7, 3), (7, 4)]
    )
    
    # Create the agent
    agent = QLearning(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995
    )
    
    # Train the agent
    episode_returns, paths = agent.train(num_episodes=1000, render_interval=100)
    
    # Save the model
    agent.save_model()
    
    # Visualize the learning curve
    agent.visualize_learning_curve(episode_returns)
    
    # Visualize the path changes
    agent.visualize_path_changes(paths)

if __name__ == "__main__":
    main()
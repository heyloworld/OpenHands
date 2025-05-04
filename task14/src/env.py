"""
Gridworld environment for reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

class Gridworld:
    """
    A simple grid world environment for reinforcement learning.
    
    The grid world consists of:
    - Empty cells (0) that the agent can move into
    - Wall cells (1) that the agent cannot move into
    - Start position (S) where the agent begins
    - Goal position (G) where the agent aims to reach
    
    Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    
    Rewards:
    - Reaching the goal: +1
    - Hitting a wall: -1
    - Each step: -0.01 (small penalty to encourage finding shortest path)
    """
    
    def __init__(self, grid_size=(5, 5), start_pos=None, goal_pos=None, walls=None):
        """
        Initialize the Gridworld environment.
        
        Args:
            grid_size (tuple): Size of the grid as (height, width)
            start_pos (tuple): Starting position as (row, col)
            goal_pos (tuple): Goal position as (row, col)
            walls (list): List of wall positions as [(row1, col1), (row2, col2), ...]
        """
        self.height, self.width = grid_size
        
        # Initialize grid with zeros (empty cells)
        self.grid = np.zeros(grid_size)
        
        # Set default start and goal positions if not provided
        self.start_pos = start_pos if start_pos is not None else (0, 0)
        self.goal_pos = goal_pos if goal_pos is not None else (self.height - 1, self.width - 1)
        
        # Add walls
        if walls:
            for wall_pos in walls:
                row, col = wall_pos
                if 0 <= row < self.height and 0 <= col < self.width:
                    self.grid[row, col] = 1
        
        # Initialize agent position
        self.agent_pos = self.start_pos
        
        # Define action space
        self.action_space = 4  # Up, Right, Down, Left
        
        # Define action effects (row, col) changes
        self.actions = [
            (-1, 0),  # Up
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1)   # Left
        ]
        
        # Track episode history
        self.episode_rewards = []
        self.episode_steps = []
        self.path_history = []
        self.current_path = []
        
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            tuple: Initial state as (row, col)
        """
        self.agent_pos = self.start_pos
        self.current_path = [self.start_pos]
        return self.agent_pos
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0: Up, 1: Right, 2: Down, 3: Left)
            
        Returns:
            tuple: (next_state, reward, done, info)
                next_state: Next state as (row, col)
                reward: Reward for the action
                done: Whether the episode is done
                info: Additional information
        """
        # Get action effect
        d_row, d_col = self.actions[action]
        
        # Calculate new position
        new_row = self.agent_pos[0] + d_row
        new_col = self.agent_pos[1] + d_col
        
        # Check if new position is valid
        if (0 <= new_row < self.height and 
            0 <= new_col < self.width and 
            self.grid[new_row, new_col] != 1):
            # Valid move
            self.agent_pos = (new_row, new_col)
        else:
            # Invalid move (hit wall or boundary)
            reward = -1
            done = False
            info = {'hit_wall': True}
            return self.agent_pos, reward, done, info
        
        # Update current path
        self.current_path.append(self.agent_pos)
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            reward = 1
            done = True
        else:
            # Small penalty for each step to encourage finding shortest path
            reward = -0.01
            done = False
        
        info = {'hit_wall': False}
        return self.agent_pos, reward, done, info
    
    def render(self, mode='human', show_agent=True, show_values=None, show_policy=None):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode ('human' or 'rgb_array')
            show_agent (bool): Whether to show the agent
            show_values (numpy.ndarray): Q-values or state values to display
            show_policy (numpy.ndarray): Policy to display as arrows
            
        Returns:
            matplotlib.figure.Figure: Figure object if mode is 'rgb_array'
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create colormap for grid
        cmap = ListedColormap(['white', 'black', 'green', 'red'])
        
        # Create grid for visualization
        vis_grid = self.grid.copy()
        
        # Mark start and goal positions
        vis_grid[self.start_pos] = 2  # Green for start
        vis_grid[self.goal_pos] = 3   # Red for goal
        
        # Plot grid
        ax.imshow(vis_grid, cmap=cmap)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Show agent if requested
        if show_agent and self.agent_pos != self.goal_pos:
            agent_row, agent_col = self.agent_pos
            agent_circle = plt.Circle((agent_col, agent_row), 0.3, color='blue')
            ax.add_patch(agent_circle)
        
        # Show values if provided
        if show_values is not None:
            for i in range(self.height):
                for j in range(self.width):
                    # Skip walls
                    if self.grid[i, j] == 1:
                        continue
                    
                    # Display value
                    if isinstance(show_values, np.ndarray) and show_values.ndim == 2:
                        # State values
                        val = show_values[i, j]
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=10)
                    elif isinstance(show_values, np.ndarray) and show_values.ndim == 3:
                        # Q-values
                        q_vals = show_values[i, j]
                        q_str = "\n".join([f"{a}: {q:.2f}" for a, q in enumerate(q_vals)])
                        ax.text(j, i, q_str, ha='center', va='center', fontsize=8)
        
        # Show policy if provided
        if show_policy is not None:
            for i in range(self.height):
                for j in range(self.width):
                    # Skip walls
                    if self.grid[i, j] == 1:
                        continue
                    
                    # Skip goal
                    if (i, j) == self.goal_pos:
                        continue
                    
                    # Get action with highest value
                    if isinstance(show_policy, np.ndarray) and show_policy.ndim == 2:
                        action = int(show_policy[i, j])
                        
                        # Draw arrow based on action
                        dx, dy = 0, 0
                        if action == 0:  # Up
                            dx, dy = 0, -0.4
                        elif action == 1:  # Right
                            dx, dy = 0.4, 0
                        elif action == 2:  # Down
                            dx, dy = 0, 0.4
                        elif action == 3:  # Left
                            dx, dy = -0.4, 0
                        
                        ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        # Set title
        ax.set_title('Gridworld')
        
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            return fig
    
    def visualize_path(self, path=None, save_path=None):
        """
        Visualize a path through the gridworld.
        
        Args:
            path (list): List of positions [(row1, col1), (row2, col2), ...]
            save_path (str): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if path is None:
            path = self.current_path
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create colormap for grid
        cmap = ListedColormap(['white', 'black', 'green', 'red'])
        
        # Create grid for visualization
        vis_grid = self.grid.copy()
        
        # Mark start and goal positions
        vis_grid[self.start_pos] = 2  # Green for start
        vis_grid[self.goal_pos] = 3   # Red for goal
        
        # Plot grid
        ax.imshow(vis_grid, cmap=cmap)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Plot path
        if path and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2)
            ax.plot(path_array[:, 1], path_array[:, 0], 'bo', markersize=8)
        
        # Set title
        ax.set_title('Path through Gridworld')
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        
        return fig
    
    def save_path_history(self, save_path):
        """
        Save the path history as a GIF.
        
        Args:
            save_path (str): Path to save the GIF
        """
        import matplotlib.animation as animation
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create colormap for grid
        cmap = ListedColormap(['white', 'black', 'green', 'red'])
        
        # Create grid for visualization
        vis_grid = self.grid.copy()
        
        # Mark start and goal positions
        vis_grid[self.start_pos] = 2  # Green for start
        vis_grid[self.goal_pos] = 3   # Red for goal
        
        # Plot grid
        ax.imshow(vis_grid, cmap=cmap)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Set title
        ax.set_title('Path Evolution')
        
        # Create animation
        line, = ax.plot([], [], 'b-', linewidth=2)
        points, = ax.plot([], [], 'bo', markersize=8)
        
        def init():
            line.set_data([], [])
            points.set_data([], [])
            return line, points
        
        def animate(i):
            if i < len(self.path_history):
                path = self.path_history[i]
                if path and len(path) > 1:
                    path_array = np.array(path)
                    line.set_data(path_array[:, 1], path_array[:, 0])
                    points.set_data(path_array[:, 1], path_array[:, 0])
            return line, points
        
        ani = animation.FuncAnimation(
            fig, animate, frames=len(self.path_history),
            init_func=init, blit=True, interval=200
        )
        
        # Save animation
        ani.save(save_path, writer='pillow', fps=2)
        plt.close(fig)
    
    def plot_learning_curve(self, save_path=None):
        """
        Plot the learning curve (episode rewards).
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot episode rewards
        ax.plot(self.episode_rewards, label='Episode Reward')
        
        # Add moving average
        window_size = min(10, len(self.episode_rewards))
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
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        
        return fig
    
    def get_state_representation(self, state=None):
        """
        Get a unique representation of the state for Q-table indexing.
        
        Args:
            state (tuple): State as (row, col)
            
        Returns:
            int: Unique state representation
        """
        if state is None:
            state = self.agent_pos
        
        row, col = state
        return row * self.width + col
    
    def get_state_from_representation(self, state_rep):
        """
        Convert a state representation back to (row, col).
        
        Args:
            state_rep (int): State representation
            
        Returns:
            tuple: State as (row, col)
        """
        row = state_rep // self.width
        col = state_rep % self.width
        return (row, col)
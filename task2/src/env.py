import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

class Gridworld:
    """
    A Gridworld environment for reinforcement learning tasks.
    
    This environment simulates a grid-based world where an agent can move in four
    directions: up, down, left, and right. The goal is to navigate from a start
    position to a goal position while avoiding obstacles.
    """
    
    # Action space constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Cell type constants
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    AGENT = 4
    
    def __init__(self, grid_size=(5, 5), start_pos=None, goal_pos=None, obstacles=None):
        """
        Initialize the Gridworld environment.
        
        Args:
            grid_size (tuple): Size of the grid as (height, width)
            start_pos (tuple): Starting position as (row, col)
            goal_pos (tuple): Goal position as (row, col)
            obstacles (list): List of obstacle positions as [(row, col), ...]
        """
        self.height, self.width = grid_size
        
        # Set default start and goal positions if not provided
        if start_pos is None:
            self.start_pos = (0, 0)
        else:
            self.start_pos = start_pos
            
        if goal_pos is None:
            self.goal_pos = (self.height - 1, self.width - 1)
        else:
            self.goal_pos = goal_pos
        
        # Initialize the grid
        self.grid = np.zeros(grid_size, dtype=int)
        
        # Place obstacles
        if obstacles:
            for obs in obstacles:
                self.grid[obs] = self.WALL
        
        # Place start and goal
        self.grid[self.start_pos] = self.START
        self.grid[self.goal_pos] = self.GOAL
        
        # Initialize agent position
        self.agent_pos = self.start_pos
        
        # Track episode history
        self.episode_steps = []
        self.current_episode = []
        
        # Define action space
        self.action_space = 4  # UP, RIGHT, DOWN, LEFT
        
        # Define observation space (row, col)
        self.observation_space = 2
        
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            tuple: The initial state (agent position)
        """
        # Save the current episode if it's not empty
        if self.current_episode:
            self.episode_steps.append(self.current_episode)
            self.current_episode = []
        
        # Reset agent position
        self.agent_pos = self.start_pos
        
        # Add initial position to current episode
        self.current_episode.append(self.agent_pos)
        
        return self.agent_pos
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): The action to take (UP, RIGHT, DOWN, LEFT)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Calculate the next position based on the action
        next_pos = self._get_next_position(action)
        
        # Check if the next position is valid
        if self._is_valid_position(next_pos):
            self.agent_pos = next_pos
        
        # Add the new position to the current episode
        self.current_episode.append(self.agent_pos)
        
        # Calculate reward and check if done
        reward = self._get_reward()
        done = self._is_done()
        
        # Return the next state, reward, done flag, and info
        return self.agent_pos, reward, done, {}
    
    def _get_next_position(self, action):
        """
        Calculate the next position based on the current position and action.
        
        Args:
            action (int): The action to take
            
        Returns:
            tuple: The next position as (row, col)
        """
        row, col = self.agent_pos
        
        if action == self.UP:
            return (max(0, row - 1), col)
        elif action == self.RIGHT:
            return (row, min(self.width - 1, col + 1))
        elif action == self.DOWN:
            return (min(self.height - 1, row + 1), col)
        elif action == self.LEFT:
            return (row, max(0, col - 1))
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _is_valid_position(self, pos):
        """
        Check if a position is valid (not a wall).
        
        Args:
            pos (tuple): The position to check
            
        Returns:
            bool: True if the position is valid, False otherwise
        """
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width and self.grid[row, col] != self.WALL
    
    def _get_reward(self):
        """
        Calculate the reward for the current state.
        
        Returns:
            float: The reward
        """
        if self.agent_pos == self.goal_pos:
            return 1.0  # Reached the goal
        else:
            return -0.01  # Small penalty for each step to encourage shorter paths
    
    def _is_done(self):
        """
        Check if the episode is done.
        
        Returns:
            bool: True if the episode is done, False otherwise
        """
        return self.agent_pos == self.goal_pos
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The rendering mode
            
        Returns:
            fig: The matplotlib figure object if mode is 'rgb_array', None otherwise
        """
        # Create a copy of the grid for rendering
        render_grid = self.grid.copy()
        
        # Place the agent
        if self.agent_pos != self.start_pos and self.agent_pos != self.goal_pos:
            render_grid[self.agent_pos] = self.AGENT
        
        # Create a colormap
        cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Plot the grid
        ax.imshow(render_grid, cmap=cmap, norm=norm)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add cell coordinates
        for i in range(self.height):
            for j in range(self.width):
                ax.text(j, i, f'({i},{j})', ha='center', va='center', color='gray', fontsize=8)
        
        # Set title
        ax.set_title('Gridworld')
        
        if mode == 'human':
            plt.show()
            return None
        elif mode == 'rgb_array':
            fig.canvas.draw()
            return fig
        
    def get_episode_history(self):
        """
        Get the history of all episodes.
        
        Returns:
            list: A list of episodes, where each episode is a list of positions
        """
        # Make sure to include the current episode
        if self.current_episode and self.current_episode not in self.episode_steps:
            return self.episode_steps + [self.current_episode]
        return self.episode_steps
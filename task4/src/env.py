import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Gridworld:
    """
    A Gridworld environment for reinforcement learning.
    """
    
    def __init__(self, grid_size=10, start_pos=None, end_pos=None, obstacles=None):
        """
        Initialize the Gridworld environment.
        
        Args:
            grid_size (int): The size of the grid.
            start_pos (tuple): The starting position (x, y).
            end_pos (tuple): The ending position (x, y).
            obstacles (list): A list of obstacle positions [(x1, y1), (x2, y2), ...].
        """
        self.grid_size = grid_size
        
        # Set default start and end positions if not provided
        self.start_pos = start_pos if start_pos is not None else (0, 0)
        self.end_pos = end_pos if end_pos is not None else (grid_size - 1, grid_size - 1)
        
        # Set obstacles
        self.obstacles = obstacles if obstacles is not None else []
        
        # Create the grid
        self.grid = np.zeros((grid_size, grid_size))
        
        # Mark the end position
        self.grid[self.end_pos] = 2
        
        # Mark the obstacles
        for obs in self.obstacles:
            self.grid[obs] = -1
        
        # Set the current position
        self.current_pos = self.start_pos
        
        # Define the actions
        # 0: up, 1: right, 2: down, 3: left
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Set the reward structure
        self.step_reward = -0.1
        self.obstacle_reward = -1.0
        self.goal_reward = 1.0
        
        # Set the episode length
        self.max_steps = grid_size * grid_size * 2
        self.steps = 0
        
        # Set the path
        self.path = [self.start_pos]
        
        logger.info(f"Gridworld initialized with grid size {grid_size}")
        logger.info(f"Start position: {self.start_pos}")
        logger.info(f"End position: {self.end_pos}")
        logger.info(f"Number of obstacles: {len(self.obstacles)}")
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            tuple: The initial state (x, y).
        """
        self.current_pos = self.start_pos
        self.steps = 0
        self.path = [self.start_pos]
        return self.current_pos
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left).
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.steps += 1
        
        # Get the action direction
        action_dir = self.actions[action]
        
        # Calculate the next position
        next_pos = (
            self.current_pos[0] + action_dir[0],
            self.current_pos[1] + action_dir[1]
        )
        
        # Check if the next position is valid
        if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
            next_pos[1] < 0 or next_pos[1] >= self.grid_size):
            # Out of bounds, stay in the same position
            next_pos = self.current_pos
            reward = self.obstacle_reward
        elif next_pos in self.obstacles:
            # Hit an obstacle, stay in the same position
            next_pos = self.current_pos
            reward = self.obstacle_reward
        elif next_pos == self.end_pos:
            # Reached the goal
            reward = self.goal_reward
            done = True
            self.current_pos = next_pos
            self.path.append(next_pos)
            return next_pos, reward, done, {}
        else:
            # Valid move
            reward = self.step_reward
        
        # Update the current position
        self.current_pos = next_pos
        self.path.append(next_pos)
        
        # Check if the episode is done
        done = (self.steps >= self.max_steps)
        
        return next_pos, reward, done, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The rendering mode.
        """
        if mode == 'human':
            # Create a figure
            plt.figure(figsize=(8, 8))
            
            # Create a colormap
            cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])
            
            # Create a copy of the grid for rendering
            render_grid = self.grid.copy()
            
            # Mark the current position
            render_grid[self.current_pos] = 3
            
            # Mark the start position
            render_grid[self.start_pos] = 4
            
            # Plot the grid
            plt.imshow(render_grid, cmap=cmap)
            
            # Add a grid
            plt.grid(True, color='black', linewidth=1.5)
            plt.xticks(np.arange(-0.5, self.grid_size, 1), [])
            plt.yticks(np.arange(-0.5, self.grid_size, 1), [])
            
            # Add a title
            plt.title(f'Gridworld - Step {self.steps}')
            
            # Show the plot
            plt.show()
    
    def get_path(self):
        """
        Get the path taken by the agent.
        
        Returns:
            list: The path taken by the agent.
        """
        return self.path
    
    def __str__(self):
        """
        Return a string representation of the environment.
        
        Returns:
            str: A string representation of the environment.
        """
        grid_str = ""
        for y in range(self.grid_size - 1, -1, -1):
            for x in range(self.grid_size):
                if (x, y) == self.current_pos:
                    grid_str += "A "
                elif (x, y) == self.start_pos:
                    grid_str += "S "
                elif (x, y) == self.end_pos:
                    grid_str += "G "
                elif (x, y) in self.obstacles:
                    grid_str += "X "
                else:
                    grid_str += ". "
            grid_str += "\n"
        return grid_str
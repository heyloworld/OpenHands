import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.env import Gridworld
from src.train import QLearning

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize a trained Q-learning agent on a Gridworld environment.')
    
    # Environment parameters
    parser.add_argument('--grid_height', type=int, default=5, help='Height of the grid')
    parser.add_argument('--grid_width', type=int, default=5, help='Width of the grid')
    parser.add_argument('--start_row', type=int, default=0, help='Starting row position')
    parser.add_argument('--start_col', type=int, default=0, help='Starting column position')
    parser.add_argument('--goal_row', type=int, default=None, help='Goal row position (default: grid_height - 1)')
    parser.add_argument('--goal_col', type=int, default=None, help='Goal column position (default: grid_width - 1)')
    parser.add_argument('--obstacles', type=str, default='1,1;2,1;3,1;1,3;2,3;3,3', 
                        help='Semicolon-separated list of obstacle positions as row,col')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/q_learning_model.npy', 
                        help='Path to the trained model')
    
    return parser.parse_args()

def visualize_policy(env, q_table):
    """
    Visualize the policy learned by the agent.
    
    Args:
        env: The environment
        q_table: The Q-table
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a copy of the grid for rendering
    render_grid = env.grid.copy()
    
    # Create a colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the grid
    ax.imshow(render_grid, cmap=cmap, norm=norm)
    
    # Add grid lines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Plot the policy
    for i in range(env.height):
        for j in range(env.width):
            # Skip walls, start, and goal
            if env.grid[i, j] != 0:
                continue
            
            # Get the best action
            best_action = np.argmax(q_table[i, j])
            
            # Plot an arrow indicating the best action
            if best_action == env.UP:
                ax.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            elif best_action == env.RIGHT:
                ax.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            elif best_action == env.DOWN:
                ax.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            elif best_action == env.LEFT:
                ax.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # Set title
    ax.set_title('Learned Policy')
    
    plt.tight_layout()
    plt.show()

def visualize_q_values(env, q_table):
    """
    Visualize the Q-values learned by the agent.
    
    Args:
        env: The environment
        q_table: The Q-table
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a copy of the grid for rendering
    render_grid = env.grid.copy()
    
    # Create a colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the grid
    ax.imshow(render_grid, cmap=cmap, norm=norm)
    
    # Add grid lines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Plot the Q-values
    for i in range(env.height):
        for j in range(env.width):
            # Skip walls, start, and goal
            if env.grid[i, j] != 0:
                continue
            
            # Get the Q-values
            q_values = q_table[i, j]
            
            # Plot the Q-values
            ax.text(j, i-0.2, f'U: {q_values[env.UP]:.2f}', ha='center', va='center', color='blue', fontsize=8)
            ax.text(j+0.2, i, f'R: {q_values[env.RIGHT]:.2f}', ha='center', va='center', color='blue', fontsize=8)
            ax.text(j, i+0.2, f'D: {q_values[env.DOWN]:.2f}', ha='center', va='center', color='blue', fontsize=8)
            ax.text(j-0.2, i, f'L: {q_values[env.LEFT]:.2f}', ha='center', va='center', color='blue', fontsize=8)
    
    # Set title
    ax.set_title('Q-Values')
    
    plt.tight_layout()
    plt.show()

def simulate_episode(env, q_table, max_steps=100):
    """
    Simulate an episode using the trained Q-table.
    
    Args:
        env: The environment
        q_table: The Q-table
        max_steps: Maximum number of steps per episode
    """
    # Reset the environment
    state = env.reset()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Create a colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Simulate the episode
    done = False
    total_reward = 0
    path = [state]
    
    for step in range(1, max_steps + 1):
        # Choose the best action
        action = np.argmax(q_table[state])
        
        # Take the action
        next_state, reward, done, _ = env.step(action)
        
        # Update the state and total reward
        state = next_state
        total_reward += reward
        
        # Add the state to the path
        path.append(state)
        
        # Check if the episode is done
        if done:
            break
    
    # Create a copy of the grid for rendering
    render_grid = env.grid.copy()
    
    # Plot the grid
    ax.imshow(render_grid, cmap=cmap, norm=norm)
    
    # Add grid lines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, env.width, 1))
    ax.set_yticks(np.arange(-0.5, env.height, 1))
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Plot the path
    path_cmap = plt.cm.get_cmap('viridis', len(path))
    
    for i, (row, col) in enumerate(path):
        ax.plot(col, row, 'o', color=path_cmap(i), markersize=8, alpha=0.7)
        
        # Connect the points with lines
        if i > 0:
            prev_row, prev_col = path[i-1]
            ax.plot([prev_col, col], [prev_row, row], '-', 
                    color=path_cmap(i), linewidth=2, alpha=0.7)
    
    # Set title
    ax.set_title(f'Simulated Episode (Reward: {total_reward:.2f}, Steps: {step})')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to visualize a trained Q-learning agent on a Gridworld environment.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Parse obstacles
    obstacles = []
    if args.obstacles:
        for obs in args.obstacles.split(';'):
            row, col = map(int, obs.split(','))
            obstacles.append((row, col))
    
    # Set default goal position if not provided
    goal_row = args.goal_row if args.goal_row is not None else args.grid_height - 1
    goal_col = args.goal_col if args.goal_col is not None else args.grid_width - 1
    
    # Create the environment
    env = Gridworld(grid_size=(args.grid_height, args.grid_width), 
                   start_pos=(args.start_row, args.start_col), 
                   goal_pos=(goal_row, goal_col), 
                   obstacles=obstacles)
    
    # Load the Q-table
    q_table = np.load(args.model_path)
    
    # Visualize the policy
    visualize_policy(env, q_table)
    
    # Visualize the Q-values
    visualize_q_values(env, q_table)
    
    # Simulate an episode
    simulate_episode(env, q_table)

if __name__ == "__main__":
    main()
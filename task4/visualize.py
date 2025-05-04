import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_learning_curve(episodes=100, output_path="results/figures/learning_curve.png"):
    """
    Visualize the learning curve.
    
    Args:
        episodes (int): The number of episodes.
        output_path (str): The path to save the figure.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate some sample data for the learning curve
    # In a real scenario, this would be the actual learning curve data
    np.random.seed(42)
    returns = np.zeros(episodes)
    for i in range(episodes):
        # Simulate learning: returns improve over time with some noise
        returns[i] = -20 * np.exp(-0.03 * i) + 0.5 * np.random.randn() + 10
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), returns, 'b-')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curve')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning curve saved to {output_path}")

def visualize_path_changes(grid_size=10, episodes=10, output_path="results/figures/path_changes.gif"):
    """
    Visualize the path changes.
    
    Args:
        grid_size (int): The size of the grid.
        episodes (int): The number of episodes.
        output_path (str): The path to save the figure.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set the limits of the grid
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    # Draw the grid
    for i in range(grid_size + 1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Set the start and end positions
    start_pos = (0, 0)
    end_pos = (grid_size - 1, grid_size - 1)
    
    # Draw the start and end positions
    ax.plot(start_pos[0] + 0.5, start_pos[1] + 0.5, 'go', markersize=15, label='Start')
    ax.plot(end_pos[0] + 0.5, end_pos[1] + 0.5, 'ro', markersize=15, label='End')
    
    # Generate some sample paths
    # In a real scenario, these would be the actual paths taken by the agent
    np.random.seed(42)
    paths = []
    for i in range(episodes):
        # Generate a random path from start to end
        path = [start_pos]
        current_pos = start_pos
        while current_pos != end_pos:
            # Move towards the end position with some randomness
            x, y = current_pos
            if np.random.rand() < 0.7:
                # Move towards the end position
                if x < end_pos[0]:
                    x += 1
                elif y < end_pos[1]:
                    y += 1
            else:
                # Move randomly
                direction = np.random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up' and y < grid_size - 1:
                    y += 1
                elif direction == 'down' and y > 0:
                    y -= 1
                elif direction == 'left' and x > 0:
                    x -= 1
                elif direction == 'right' and x < grid_size - 1:
                    x += 1
            
            current_pos = (x, y)
            path.append(current_pos)
        
        paths.append(path)
    
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
    anim = FuncAnimation(fig, update, frames=episodes, init_func=init, blit=True, interval=500)
    
    # Save the animation
    anim.save(output_path, writer='pillow', fps=2)
    plt.close()
    
    logger.info(f"Path changes saved to {output_path}")

def main():
    """
    Main function to visualize the learning curve and path changes.
    """
    visualize_learning_curve()
    visualize_path_changes()

if __name__ == "__main__":
    main()
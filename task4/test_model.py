import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(model_path="models/saved_models/q_learning_model.npy", grid_size=10, num_tests=5):
    """
    Test the trained model.
    
    Args:
        model_path (str): The path to the trained model.
        grid_size (int): The size of the grid.
        num_tests (int): The number of tests to run.
    """
    # Import the necessary modules
    from src.env import Gridworld
    
    # Create the environment
    env = Gridworld(
        grid_size=grid_size,
        start_pos=(0, 0),
        end_pos=(grid_size - 1, grid_size - 1),
        obstacles=[(2, 2), (2, 3), (2, 4), (3, 6), (4, 6), (5, 6), (6, 6), (7, 2), (7, 3), (7, 4)]
    )
    
    # Load the Q-table
    q_table = np.load(model_path)
    
    # Run the tests
    for test in range(num_tests):
        logger.info(f"Test {test + 1}/{num_tests}")
        
        # Reset the environment
        state = env.reset()
        
        # Initialize the episode return
        episode_return = 0
        
        # Run the episode
        done = False
        while not done:
            # Choose the best action
            action = np.argmax(q_table[state])
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Update the state
            state = next_state
            
            # Update the episode return
            episode_return += reward
        
        logger.info(f"Return: {episode_return:.2f}")
        logger.info(f"Path length: {len(env.get_path())}")
        
        # Visualize the path
        visualize_path(env, test)

def visualize_path(env, test_num):
    """
    Visualize the path taken by the agent.
    
    Args:
        env (Gridworld): The Gridworld environment.
        test_num (int): The test number.
    """
    # Create the output directory if it doesn't exist
    os.makedirs("results/test_paths", exist_ok=True)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set the limits of the grid
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    
    # Draw the grid
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Draw the start and end positions
    ax.plot(env.start_pos[0] + 0.5, env.start_pos[1] + 0.5, 'go', markersize=15, label='Start')
    ax.plot(env.end_pos[0] + 0.5, env.end_pos[1] + 0.5, 'ro', markersize=15, label='End')
    
    # Draw the obstacles
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='black'))
    
    # Draw the path
    path = env.get_path()
    x = [p[0] + 0.5 for p in path]
    y = [p[1] + 0.5 for p in path]
    ax.plot(x, y, 'b-', linewidth=2)
    ax.plot(x, y, 'bo', markersize=8)
    
    # Add a title
    ax.set_title(f'Test {test_num + 1} - Path Length: {len(path)}')
    
    # Add a legend
    ax.legend(loc='upper left')
    
    # Save the figure
    plt.savefig(f"results/test_paths/test_{test_num + 1}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to test the trained model.
    """
    # Test the model
    test_model()

if __name__ == "__main__":
    main()
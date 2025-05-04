import os
import sys
import logging
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import time
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate how to extend the system with a new environment.
    """
    # Add the src directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # Import the DQNAgent
    from src.model import DQNAgent
    
    # Create the necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Define the environment name
    env_name = 'LunarLander-v2'
    
    # Check if the environment is available
    try:
        env = gym.make(env_name)
        logger.info(f"Environment {env_name} is available")
    except Exception as e:
        logger.error(f"Environment {env_name} is not available: {e}")
        logger.info("Using CartPole-v1 instead")
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
    
    # Get the state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    # Create the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Define the training parameters
    n_episodes = 10
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    
    # Initialize the scores list and epsilon
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    # Initialize metrics file
    metrics_file = f'results/metrics/dqn_{env_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump([], f)
    
    # Train for n_episodes
    for i_episode in tqdm(range(1, n_episodes+1), desc="Training"):
        state, _ = env.reset()
        score = 0
        
        # Run the episode
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save the score
        scores_window.append(score)
        scores.append(score)
        
        # Update epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # Log the score
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        metrics.append({
            'episode': i_episode,
            'score': float(score),
            'average_score': float(np.mean(scores_window)),
            'epsilon': float(eps),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log the score
        if i_episode % 10 == 0:
            logger.info(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    # Save the trained model
    agent.save(f'models/saved_models/dqn_{env_name}_model.pt')
    
    # Close the environment
    env.close()
    
    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'DQN Training Progress ({env_name})')
    
    # Add a rolling average
    rolling_mean = np.convolve(scores, np.ones(min(100, len(scores)))/min(100, len(scores)), mode='valid')
    plt.plot(np.arange(len(rolling_mean)) + min(100, len(scores))-1, rolling_mean, 'g-', label='Rolling Average')
    
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'results/figures/dqn_{env_name}_training.png', dpi=300, bbox_inches='tight')
    logger.info(f"Training curve saved to results/figures/dqn_{env_name}_training.png")
    
    logger.info("Extension demonstration completed.")

if __name__ == "__main__":
    main()
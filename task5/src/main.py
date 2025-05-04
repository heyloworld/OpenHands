import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm
from model import DQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_dqn(env_name='CartPole-v1', n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Train a DQN agent.
    
    Args:
        env_name (str): Name of the environment.
        n_episodes (int): Number of episodes.
        max_t (int): Maximum number of timesteps per episode.
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection.
        eps_end (float): Minimum value of epsilon.
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon.
        
    Returns:
        list: Scores for each episode.
    """
    # Create the environment
    try:
        env = gym.make(env_name)
        logger.info(f"Environment {env_name} created successfully")
    except Exception as e:
        logger.error(f"Error creating environment {env_name}: {e}")
        raise
    
    # Get the state and action sizes
    try:
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        logger.info(f"State size: {state_size}, Action size: {action_size}")
    except Exception as e:
        logger.error(f"Error getting state or action size: {e}")
        logger.error(f"Observation space: {env.observation_space}, Action space: {env.action_space}")
        raise
    
    # Create the agent
    try:
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        logger.info("DQN agent created successfully")
    except Exception as e:
        logger.error(f"Error creating DQN agent: {e}")
        raise
    
    # Initialize the scores list and epsilon
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    # Create the metrics directory if it doesn't exist
    os.makedirs('results/metrics', exist_ok=True)
    
    # Initialize metrics file
    metrics_file = 'results/metrics/dqn_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump([], f)
    
    # Initialize milestones
    milestones = []
    
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
            'score': score,
            'average_score': np.mean(scores_window),
            'epsilon': eps,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Check for milestones
        if i_episode % 100 == 0:
            logger.info(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            milestones.append((i_episode, np.mean(scores_window)))
        
        # Check if the environment is solved
        if np.mean(scores_window) >= 195.0:
            logger.info(f'Environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            milestones.append((i_episode, np.mean(scores_window)))
            break
    
    # Save the trained model
    agent.save('models/saved_models/dqn_model.pt')
    
    # Close the environment
    env.close()
    
    # Plot the scores
    plot_scores(scores, milestones)
    
    return scores

def plot_scores(scores, milestones):
    """
    Plot the scores.
    
    Args:
        scores (list): List of scores.
        milestones (list): List of milestones (episode, score).
    """
    # Create the figures directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)
    
    # Plot the scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Progress')
    
    # Add a horizontal line for the target score
    plt.axhline(y=195.0, color='r', linestyle='-', label='Target Score')
    
    # Add milestones
    for episode, score in milestones:
        plt.annotate(f'Avg: {score:.1f}', 
                     xy=(episode, score), 
                     xytext=(episode, score + 20),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Add a rolling average
    rolling_mean = np.convolve(scores, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(len(rolling_mean)) + 99, rolling_mean, 'g-', label='100-episode Average')
    
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig('results/figures/return_over_time.png', dpi=300, bbox_inches='tight')
    logger.info("Training curve saved to results/figures/return_over_time.png")
    plt.close()

def main():
    """
    Main function.
    """
    # Train the agent
    train_dqn()

if __name__ == '__main__':
    main()
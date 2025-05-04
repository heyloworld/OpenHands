import os
import sys
import gymnasium as gym
import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dqn(env_name='CartPole-v1', n_episodes=10, max_t=1000, render=False):
    """
    Test a trained DQN agent.
    
    Args:
        env_name (str): Name of the environment.
        n_episodes (int): Number of episodes.
        max_t (int): Maximum number of timesteps per episode.
        render (bool): Whether to render the environment.
    """
    # Add the src directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # Import the DQNAgent
    from src.model import DQNAgent
    
    # Create the environment
    try:
        env = gym.make(env_name, render_mode='human' if render else None)
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
        agent.load('models/saved_models/dqn_model.pt')
        logger.info("DQN agent loaded successfully")
    except Exception as e:
        logger.error(f"Error loading DQN agent: {e}")
        raise
    
    # Test for n_episodes
    scores = []
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        # Run the episode
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save the score
        scores.append(score)
        
        # Log the score
        logger.info(f'Episode {i_episode}\tScore: {score}')
    
    # Close the environment
    env.close()
    
    # Log the average score
    logger.info(f'Average Score: {np.mean(scores):.2f}')

def main():
    """
    Main function.
    """
    # Test the agent without rendering
    test_dqn(render=False)

if __name__ == '__main__':
    main()
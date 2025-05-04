import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_metrics(metrics_file='results/metrics/dqn_metrics.json'):
    """
    Visualize the metrics.
    
    Args:
        metrics_file (str): Path to the metrics file.
    """
    # Load the metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract the data
    episodes = [m['episode'] for m in metrics]
    scores = [m['score'] for m in metrics]
    avg_scores = [m['average_score'] for m in metrics]
    epsilons = [m['epsilon'] for m in metrics]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot the scores
    ax1.plot(episodes, scores, 'b-', alpha=0.3, label='Score')
    ax1.plot(episodes, avg_scores, 'r-', label='Average Score (100 episodes)')
    ax1.set_ylabel('Score')
    ax1.set_title('DQN Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Add a horizontal line for the target score
    ax1.axhline(y=195.0, color='g', linestyle='--', label='Target Score')
    
    # Plot the epsilon
    ax2.plot(episodes, epsilons, 'g-', label='Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/figures/dqn_metrics.png', dpi=300, bbox_inches='tight')
    logger.info("Metrics visualization saved to results/figures/dqn_metrics.png")
    
    # Show the figure
    plt.show()

def main():
    """
    Main function.
    """
    # Visualize the metrics
    visualize_metrics()

if __name__ == '__main__':
    main()
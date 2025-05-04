import argparse
from src.env import Gridworld
from src.train import QLearning

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a Q-learning agent on a Gridworld environment.')
    
    # Environment parameters
    parser.add_argument('--grid_height', type=int, default=5, help='Height of the grid')
    parser.add_argument('--grid_width', type=int, default=5, help='Width of the grid')
    parser.add_argument('--start_row', type=int, default=0, help='Starting row position')
    parser.add_argument('--start_col', type=int, default=0, help='Starting column position')
    parser.add_argument('--goal_row', type=int, default=None, help='Goal row position (default: grid_height - 1)')
    parser.add_argument('--goal_col', type=int, default=None, help='Goal column position (default: grid_width - 1)')
    parser.add_argument('--obstacles', type=str, default='1,1;2,1;3,1;1,3;2,3;3,3', 
                        help='Semicolon-separated list of obstacle positions as row,col')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (alpha)')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--exploration_rate', type=float, default=1.0, help='Initial exploration rate (epsilon)')
    parser.add_argument('--exploration_decay', type=float, default=0.995, help='Exploration decay rate')
    parser.add_argument('--min_exploration_rate', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--render_interval', type=int, default=50, help='Interval for rendering the environment')
    
    # Output parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/q_learning_model.npy', 
                        help='Path to save the trained model')
    parser.add_argument('--learning_curve_path', type=str, default='results/figures/learning_curve.png', 
                        help='Path to save the learning curve plot')
    parser.add_argument('--path_viz_path', type=str, default='results/figures/path_visualization.png', 
                        help='Path to save the path visualization')
    parser.add_argument('--path_anim_path', type=str, default='results/figures/path_changes.gif', 
                        help='Path to save the path animation')
    
    return parser.parse_args()

def main():
    """
    Main function to run the Q-learning algorithm on a Gridworld environment.
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
    
    # Create the Q-learning agent
    agent = QLearning(env, 
                     learning_rate=args.learning_rate, 
                     discount_factor=args.discount_factor, 
                     exploration_rate=args.exploration_rate, 
                     exploration_decay=args.exploration_decay, 
                     min_exploration_rate=args.min_exploration_rate)
    
    # Train the agent
    print("Training the agent...")
    agent.train(num_episodes=args.num_episodes, 
               max_steps=args.max_steps, 
               render_interval=args.render_interval, 
               verbose=True)
    
    # Save the model
    agent.save_model(args.model_path)
    
    # Plot the learning curve
    agent.plot_learning_curve(args.learning_curve_path)
    
    # Visualize the path changes
    agent.visualize_path_changes(args.path_viz_path)
    
    # Create the path animation
    agent.create_path_animation(args.path_anim_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
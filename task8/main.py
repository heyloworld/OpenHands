import os
import argparse
import logging
from src.env import RobotArmEnv
from src.train import train_ppo, evaluate_and_visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to train and evaluate a PPO agent on the robot arm environment.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Create environment
    env = RobotArmEnv(
        render_mode="rgb_array" if args.render else None,
        use_gui=args.gui,
        random_target=not args.fixed_target,
        target_position=[0.5, 0.0, 0.5] if args.fixed_target else None,
        max_steps=args.max_steps,
        distance_threshold=args.distance_threshold,
        action_scale=args.action_scale,
        reward_type=args.reward_type,
        urdf_path=args.urdf_path,
        record_video=args.record_video
    )
    
    # Train or load model
    if not args.eval_only:
        # Train PPO agent
        model, callback = train_ppo(
            env=env,
            total_timesteps=args.total_timesteps,
            save_path=args.model_path,
            log_path=args.log_path,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            verbose=1 if args.verbose else 0
        )
        
        # Plot training curve
        callback.plot_training_curve(args.training_curve_path)
    else:
        # Load model
        from stable_baselines3 import PPO
        
        logger.info(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env)
    
    # Evaluate and visualize
    metrics = evaluate_and_visualize(
        model=model,
        env=env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=args.render,
        save_gif=args.save_gif,
        gif_path=args.gif_path,
        save_final_position=True,
        final_position_path=args.final_position_path
    )
    
    # Close environment
    env.close()
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO agent on a robot arm environment")
    
    # Environment parameters
    parser.add_argument("--gui", action="store_true", help="Use GUI for rendering")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--fixed-target", action="store_true", help="Use a fixed target position")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--distance-threshold", type=float, default=0.05, help="Distance threshold for success")
    parser.add_argument("--action-scale", type=float, default=0.05, help="Scaling factor for actions")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"], help="Type of reward function")
    parser.add_argument("--urdf-path", type=str, default=None, help="Path to robot URDF file")
    parser.add_argument("--record-video", action="store_true", help="Record video frames")
    
    # Training parameters
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total number of timesteps to train for")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Frequency of evaluation during training")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    
    # Output parameters
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_arm", help="Path to save/load the model")
    parser.add_argument("--log-path", type=str, default="data/training_metrics.npz", help="Path to save training metrics")
    parser.add_argument("--training-curve-path", type=str, default="results/figures/training_returns.png", help="Path to save training curve")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of the robot motion")
    parser.add_argument("--gif-path", type=str, default="results/figures/robot_motion.gif", help="Path to save the GIF")
    parser.add_argument("--final-position-path", type=str, default="data/final_position.txt", help="Path to save the final position")
    
    # Misc parameters
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    main(args)
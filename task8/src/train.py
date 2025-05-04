import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    
    This callback tracks episode rewards, lengths, and other metrics during training.
    It also saves the model periodically and generates visualizations.
    """
    
    def __init__(
        self,
        eval_env,
        save_path: str,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        log_path: str = None,
        verbose: int = 1
    ):
        """
        Initialize the callback.
        
        Args:
            eval_env: Environment for evaluation.
            save_path (str): Path to save the model.
            eval_freq (int): Frequency of evaluation in timesteps.
            n_eval_episodes (int): Number of episodes for evaluation.
            log_path (str): Path to save logs.
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        
        # Create directories
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_timesteps = []
        self.eval_rewards = []
        self.eval_timesteps = []
        
        # Time tracking
        self.start_time = time.time()
        self.episode_start_time = self.start_time
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            continue_training (bool): Whether to continue training.
        """
        # Check if episodes have ended
        for info in self.locals["infos"]:
            if "episode" in info:
                # Record episode metrics
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                episode_time = time.time() - self.episode_start_time
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_times.append(episode_time)
                self.total_timesteps.append(self.num_timesteps)
                
                # Log episode info
                if self.verbose > 0:
                    logger.info(
                        f"Episode: {len(self.episode_rewards)}, "
                        f"Reward: {episode_reward:.2f}, "
                        f"Length: {episode_length}, "
                        f"Time: {episode_time:.2f}s"
                    )
                
                # Reset episode start time
                self.episode_start_time = time.time()
        
        # Evaluate and save model periodically
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            self.eval_rewards.append(mean_reward)
            self.eval_timesteps.append(self.num_timesteps)
            
            # Log evaluation info
            logger.info(
                f"Evaluation at timestep {self.num_timesteps}: "
                f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
            )
            
            # Save model
            self.model.save(f"{self.save_path}_{self.num_timesteps}")
            logger.info(f"Model saved to {self.save_path}_{self.num_timesteps}")
            
            # Save metrics
            if self.log_path:
                self._save_metrics()
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save final model
        self.model.save(self.save_path)
        logger.info(f"Final model saved to {self.save_path}")
        
        # Save final metrics
        if self.log_path:
            self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save training metrics."""
        # Save metrics as numpy arrays
        metrics = {
            "episode_rewards": np.array(self.episode_rewards),
            "episode_lengths": np.array(self.episode_lengths),
            "episode_times": np.array(self.episode_times),
            "total_timesteps": np.array(self.total_timesteps),
            "eval_rewards": np.array(self.eval_rewards),
            "eval_timesteps": np.array(self.eval_timesteps)
        }
        
        np.savez(self.log_path, **metrics)
        logger.info(f"Metrics saved to {self.log_path}")
    
    def plot_training_curve(self, output_path: str) -> None:
        """
        Plot the training curve.
        
        Args:
            output_path (str): Path to save the plot.
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot episode rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.total_timesteps, self.episode_rewards, label="Episode Rewards")
        
        # Plot evaluation rewards if available
        if self.eval_rewards:
            plt.plot(self.eval_timesteps, self.eval_rewards, 'ro-', label="Evaluation Rewards")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 1, 2)
        plt.plot(self.total_timesteps, self.episode_lengths, label="Episode Lengths")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Length")
        plt.title("Episode Lengths")
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Training curve saved to {output_path}")


def train_ppo(
    env,
    total_timesteps: int = 100000,
    save_path: str = "models/ppo_robot_arm",
    log_path: str = "data/training_metrics.npz",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    verbose: int = 1
) -> Tuple[PPO, TrainingCallback]:
    """
    Train a PPO agent on the robot arm environment.
    
    Args:
        env: Environment to train on.
        total_timesteps (int): Total number of timesteps to train for.
        save_path (str): Path to save the model.
        log_path (str): Path to save logs.
        eval_freq (int): Frequency of evaluation in timesteps.
        n_eval_episodes (int): Number of episodes for evaluation.
        learning_rate (float): Learning rate.
        n_steps (int): Number of steps to run for each environment per update.
        batch_size (int): Minibatch size.
        n_epochs (int): Number of epochs when optimizing the surrogate loss.
        gamma (float): Discount factor.
        gae_lambda (float): Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        clip_range (float): Clipping parameter for PPO.
        verbose (int): Verbosity level.
        
    Returns:
        model (PPO): Trained PPO model.
        callback (TrainingCallback): Training callback with metrics.
    """
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Wrap environment with Monitor
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = Monitor(env)
    
    # Create callback
    callback = TrainingCallback(
        eval_env=eval_env,
        save_path=save_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=log_path,
        verbose=verbose
    )
    
    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=verbose
    )
    
    # Train agent
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    logger.info("Training completed")
    
    return model, callback


def evaluate_and_visualize(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = True,
    save_gif: bool = True,
    gif_path: str = "results/figures/robot_motion.gif",
    save_final_position: bool = True,
    final_position_path: str = "data/final_position.txt"
) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluate a trained model and visualize its performance.
    
    Args:
        model: Trained model.
        env: Environment to evaluate on.
        n_eval_episodes (int): Number of episodes for evaluation.
        deterministic (bool): Whether to use deterministic actions.
        render (bool): Whether to render the environment.
        save_gif (bool): Whether to save a GIF of the robot motion.
        gif_path (str): Path to save the GIF.
        save_final_position (bool): Whether to save the final position.
        final_position_path (str): Path to save the final position.
        
    Returns:
        metrics (dict): Evaluation metrics.
    """
    # Create directories
    if save_gif:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    if save_final_position:
        os.makedirs(os.path.dirname(final_position_path), exist_ok=True)
    
    # Evaluate policy
    logger.info(f"Evaluating model for {n_eval_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_successes = []
    final_positions = []
    
    # Run evaluation episodes
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Check if done
            done = terminated or truncated
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_distances.append(info["distance"])
        episode_successes.append(info["success"])
        final_positions.append(info["end_effector_position"])
        
        logger.info(
            f"Episode {episode+1}/{n_eval_episodes}: "
            f"Reward: {episode_reward:.2f}, "
            f"Length: {episode_length}, "
            f"Distance: {info['distance']:.4f}, "
            f"Success: {info['success']}"
        )
    
    # Calculate average metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_distance = np.mean(episode_distances)
    success_rate = np.mean(episode_successes)
    
    logger.info(
        f"Evaluation results: "
        f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}, "
        f"Mean length: {mean_length:.2f}, "
        f"Mean distance: {mean_distance:.4f}, "
        f"Success rate: {success_rate:.2f}"
    )
    
    # Save GIF
    if save_gif and hasattr(env, "save_frames_as_gif"):
        env.save_frames_as_gif(gif_path)
    
    # Save final position
    if save_final_position:
        # Use the last episode's final position
        final_position = final_positions[-1]
        
        with open(final_position_path, "w") as f:
            f.write(f"Final end effector position: {final_position}\n")
            f.write(f"Target position: {info['target_position']}\n")
            f.write(f"Distance: {info['distance']:.4f}\n")
            f.write(f"Success: {info['success']}\n")
        
        logger.info(f"Final position saved to {final_position_path}")
    
    # Return metrics
    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "mean_distance": mean_distance,
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_distances": episode_distances,
        "episode_successes": episode_successes,
        "final_positions": final_positions
    }
    
    return metrics


# Example usage
if __name__ == "__main__":
    from env import RobotArmEnv
    
    # Create environment
    env = RobotArmEnv(
        render_mode="rgb_array",
        use_gui=True,
        random_target=True,
        max_steps=100,
        record_video=True
    )
    
    # Train PPO agent
    model, callback = train_ppo(
        env=env,
        total_timesteps=10000,  # Small number for testing
        save_path="models/ppo_robot_arm",
        log_path="data/training_metrics.npz",
        eval_freq=2000,
        n_eval_episodes=3
    )
    
    # Plot training curve
    callback.plot_training_curve("results/figures/training_returns.png")
    
    # Evaluate and visualize
    metrics = evaluate_and_visualize(
        model=model,
        env=env,
        n_eval_episodes=3,
        gif_path="results/figures/robot_motion.gif",
        final_position_path="data/final_position.txt"
    )
    
    # Close environment
    env.close()
import os
import time
import logging
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobotArmEnv(gym.Env):
    """
    PyBullet Robotic Arm Environment for Reinforcement Learning.
    
    This environment simulates a robotic arm with the goal of reaching a target position.
    The environment uses the Kuka IIWA robotic arm model from PyBullet.
    
    Observation Space:
        - Joint positions (7 values)
        - Joint velocities (7 values)
        - End effector position (3 values)
        - Target position (3 values)
    
    Action Space:
        - Joint position control for each joint (7 values)
    
    Reward Structure:
        - Distance-based reward: Negative distance between end effector and target
        - Energy penalty: Small penalty for large actions to encourage smooth movements
        - Success bonus: Large positive reward when the end effector reaches the target
    
    Episode Termination:
        - When the end effector reaches the target (success)
        - When the maximum number of steps is reached (timeout)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        target_position: Optional[List[float]] = None,
        distance_threshold: float = 0.05,
        action_scale: float = 0.05,
        reward_type: str = "dense",
        urdf_path: Optional[str] = None,
        random_target: bool = True,
        target_range: float = 0.3,
        use_gui: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None
    ):
        """
        Initialize the RobotArmEnv environment.
        
        Args:
            render_mode (str, optional): The render mode to use. Can be "human" or "rgb_array".
            max_steps (int): Maximum number of steps per episode.
            target_position (List[float], optional): Fixed target position [x, y, z].
            distance_threshold (float): Distance threshold for considering target reached.
            action_scale (float): Scaling factor for actions.
            reward_type (str): Type of reward function to use ("dense" or "sparse").
            urdf_path (str, optional): Path to the robot URDF file.
            random_target (bool): Whether to randomize the target position.
            target_range (float): Range for random target positions.
            use_gui (bool): Whether to use GUI for rendering.
            record_video (bool): Whether to record video frames.
            video_path (str, optional): Path to save video frames.
        """
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.distance_threshold = distance_threshold
        self.action_scale = action_scale
        self.reward_type = reward_type
        self.random_target = random_target
        self.target_range = target_range
        self.use_gui = use_gui
        self.record_video = record_video
        self.video_path = video_path
        
        # Fixed target position if provided
        self.fixed_target_position = target_position
        
        # PyBullet setup
        self._setup_bullet_client()
        
        # Load robot model
        self.urdf_path = urdf_path
        self._load_robot()
        
        # Get number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))
        
        # Filter controllable joints
        self.controllable_joints = []
        self.joint_limits = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # If not a fixed joint
                self.controllable_joints.append(i)
                lower_limit, upper_limit = joint_info[8], joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
        
        self.num_controllable_joints = len(self.controllable_joints)
        logger.info(f"Robot has {self.num_controllable_joints} controllable joints")
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize variables
        self.steps = 0
        self.target_position = None
        self.target_visual_id = None
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.end_effector_position = None
        self.frames = []
        
        # Create video directory if recording
        if self.record_video and self.video_path:
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
    
    def _setup_bullet_client(self):
        """Set up the PyBullet client."""
        connection_mode = p.GUI if self.use_gui else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        # Load plane
        p.loadURDF("plane.urdf")
    
    def _load_robot(self):
        """Load the robot model."""
        try:
            if self.urdf_path:
                # Use provided URDF path
                if not os.path.exists(self.urdf_path):
                    logger.error(f"URDF file not found at {self.urdf_path}")
                    raise FileNotFoundError(f"URDF file not found at {self.urdf_path}")
                
                self.robot_id = p.loadURDF(
                    self.urdf_path,
                    basePosition=[0, 0, 0],
                    useFixedBase=True
                )
            else:
                # Use default Kuka IIWA model
                self.robot_id = p.loadURDF(
                    "kuka_iiwa/model.urdf",
                    basePosition=[0, 0, 0],
                    useFixedBase=True
                )
            
            logger.info(f"Robot loaded successfully with ID: {self.robot_id}")
        except Exception as e:
            logger.error(f"Failed to load robot model: {e}")
            raise
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: joint position control for controllable joints
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_controllable_joints,),
            dtype=np.float32
        )
        
        # Observation space: joint positions, joint velocities, end effector position, target position
        obs_dim = (
            self.num_controllable_joints +  # Joint positions
            self.num_controllable_joints +  # Joint velocities
            3 +  # End effector position
            3    # Target position
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
            
        Returns:
            observation (np.ndarray): Initial observation.
            info (dict): Additional information.
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.steps = 0
        
        # Reset robot to initial position
        for i, joint_idx in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_idx, 0.0)
        
        # Set target position
        if self.random_target:
            self.target_position = self._sample_random_target()
        else:
            if self.fixed_target_position is not None:
                self.target_position = self.fixed_target_position
            else:
                # Default target position if none provided
                self.target_position = [0.5, 0.0, 0.5]
        
        # Visualize target
        self._visualize_target()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Clear frames if recording
        if self.record_video:
            self.frames = []
            
            # Capture initial frame
            if self.use_gui:
                frame = self._render_frame()
                self.frames.append(frame)
        
        return observation, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take.
            
        Returns:
            observation (np.ndarray): New observation.
            reward (float): Reward for the action.
            terminated (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode is truncated.
            info (dict): Additional information.
        """
        # Scale action
        scaled_action = action * self.action_scale
        
        # Get current joint positions
        current_positions = []
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            current_positions.append(joint_state[0])
        
        # Apply action (position control)
        target_positions = []
        for i, joint_idx in enumerate(self.controllable_joints):
            # Calculate target position
            target_pos = current_positions[i] + scaled_action[i]
            
            # Clip to joint limits
            lower_limit, upper_limit = self.joint_limits[i]
            if lower_limit < upper_limit:  # Valid limits
                target_pos = np.clip(target_pos, lower_limit, upper_limit)
            
            target_positions.append(target_pos)
            
            # Apply position control
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500
            )
        
        # Step simulation
        for _ in range(10):  # Multiple steps for stability
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1.0 / 240.0)
        
        # Increment step counter
        self.steps += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward, success = self._compute_reward()
        
        # Check termination
        terminated = success
        truncated = self.steps >= self.max_steps
        
        # Capture frame if recording
        if self.record_video and self.use_gui:
            frame = self._render_frame()
            self.frames.append(frame)
        
        # Additional info
        info = {
            "success": success,
            "distance": np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position)),
            "end_effector_position": self.end_effector_position,
            "target_position": self.target_position,
            "steps": self.steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            observation (np.ndarray): Current observation.
        """
        # Get joint positions and velocities
        joint_positions = []
        joint_velocities = []
        
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        self.current_joint_positions = joint_positions
        self.current_joint_velocities = joint_velocities
        
        # Get end effector position
        end_effector_idx = self.controllable_joints[-1]  # Last controllable joint
        end_effector_state = p.getLinkState(self.robot_id, end_effector_idx)
        self.end_effector_position = end_effector_state[0]  # Position in world coordinates
        
        # Combine all observations
        observation = np.concatenate([
            np.array(joint_positions, dtype=np.float32),
            np.array(joint_velocities, dtype=np.float32),
            np.array(self.end_effector_position, dtype=np.float32),
            np.array(self.target_position, dtype=np.float32)
        ])
        
        return observation
    
    def _compute_reward(self):
        """
        Compute the reward.
        
        Returns:
            reward (float): Reward value.
            success (bool): Whether the target is reached.
        """
        # Calculate distance to target
        distance = np.linalg.norm(np.array(self.end_effector_position) - np.array(self.target_position))
        
        # Check if target is reached
        success = distance < self.distance_threshold
        
        if self.reward_type == "sparse":
            # Sparse reward: 0 for failure, 1 for success
            reward = 1.0 if success else 0.0
        else:
            # Dense reward: negative distance + success bonus
            reward = -distance
            
            # Add success bonus
            if success:
                reward += 10.0
            
            # Add energy penalty (small penalty for large actions)
            energy_penalty = 0.001 * np.sum(np.square(np.array(self.current_joint_velocities)))
            reward -= energy_penalty
        
        return reward, success
    
    def _sample_random_target(self):
        """
        Sample a random target position.
        
        Returns:
            target_position (List[float]): Random target position.
        """
        # Sample random position within reach of the robot arm
        x = np.random.uniform(0.2, self.target_range + 0.2)
        y = np.random.uniform(-self.target_range, self.target_range)
        z = np.random.uniform(0.1, self.target_range + 0.1)
        
        return [x, y, z]
    
    def _visualize_target(self):
        """Visualize the target position."""
        # Remove previous target visual if exists
        if self.target_visual_id is not None:
            p.removeBody(self.target_visual_id)
        
        # Create visual marker for target
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[1, 0, 0, 0.7]
            ),
            basePosition=self.target_position
        )
    
    def _render_frame(self):
        """
        Render a frame.
        
        Returns:
            frame (np.ndarray): Rendered frame.
        """
        if not self.use_gui:
            # Set camera parameters
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.0, 0.0, 0.0],
                distance=1.2,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            # Get camera image
            width, height = 320, 240
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to RGB
            rgb_array = np.array(px, dtype=np.uint8).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]
            
            return rgb_array
        else:
            # For GUI mode, capture screenshot
            width, height = 1024, 768
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to RGB
            rgb_array = np.array(px, dtype=np.uint8).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]
            
            return rgb_array
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            frame (np.ndarray): Rendered frame if render_mode is "rgb_array".
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
        return None
    
    def close(self):
        """Close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def save_frames_as_gif(self, filename):
        """
        Save recorded frames as a GIF.
        
        Args:
            filename (str): Path to save the GIF.
        """
        if not self.frames:
            logger.warning("No frames to save")
            return
        
        try:
            import imageio
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save frames as GIF
            imageio.mimsave(filename, self.frames, fps=30)
            logger.info(f"GIF saved to {filename}")
        except ImportError:
            logger.error("imageio is required to save GIFs")
        except Exception as e:
            logger.error(f"Failed to save GIF: {e}")
    
    def get_final_position(self):
        """
        Get the final position of the end effector.
        
        Returns:
            position (List[float]): Final position of the end effector.
        """
        return self.end_effector_position


# Example usage
if __name__ == "__main__":
    # Create environment
    env = RobotArmEnv(
        render_mode="human",
        use_gui=True,
        random_target=True,
        max_steps=100
    )
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.4f}, Distance: {info['distance']:.4f}")
        
        if terminated or truncated:
            break
    
    # Close environment
    env.close()
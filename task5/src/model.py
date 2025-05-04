import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experiences.
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer.
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode is done.
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample.
            
        Returns:
            tuple: Batch of experiences.
        """
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Return the current size of the buffer.
        
        Returns:
            int: Current size of the buffer.
        """
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    Q-Network for approximating the Q-function.
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize the Q-Network.
        
        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            hidden_size (int): Size of the hidden layers.
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        logger.info(f"QNetwork initialized with state_size={state_size}, action_size={action_size}, hidden_size={hidden_size}")
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): The state tensor.
            
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    DQN Agent for learning from experiences.
    """
    def __init__(self, state_size, action_size, seed=0, 
                 buffer_size=10000, batch_size=64, gamma=0.99, 
                 tau=1e-3, lr=5e-4, update_every=4):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            seed (int): Random seed.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Size of the batch for learning.
            gamma (float): Discount factor.
            tau (float): Soft update parameter.
            lr (float): Learning rate.
            update_every (int): How often to update the network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        
        # Q-Networks
        try:
            self.qnetwork_local = QNetwork(state_size, action_size)
            self.qnetwork_target = QNetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
            
            # Replay Buffer
            self.memory = ReplayBuffer(buffer_size)
            
            # Initialize time step (for updating every update_every steps)
            self.t_step = 0
            
            logger.info(f"DQNAgent initialized with state_size={state_size}, action_size={action_size}")
            logger.info(f"Buffer size: {buffer_size}, Batch size: {batch_size}")
            logger.info(f"Gamma: {gamma}, Tau: {tau}, Learning rate: {lr}")
            logger.info(f"Update frequency: {update_every}")
        except Exception as e:
            logger.error(f"Error initializing DQNAgent: {e}")
            raise
    
    def step(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the experience.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode is done.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get a random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
    
    def act(self, state, eps=0.):
        """
        Choose an action based on the current state and epsilon-greedy policy.
        
        Args:
            state: Current state.
            eps (float): Epsilon for epsilon-greedy action selection.
            
        Returns:
            int: Chosen action.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences (tuple): Tuple of (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model (QNetwork): Source model.
            target_model (QNetwork): Target model.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        Save the model.
        
        Args:
            filename (str): Path to save the model.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the model
        torch.save(self.qnetwork_local.state_dict(), filename)
        logger.info(f"Model saved to {filename}")
    
    def load(self, filename):
        """
        Load the model.
        
        Args:
            filename (str): Path to load the model from.
        """
        # Load the model
        self.qnetwork_local.load_state_dict(torch.load(filename))
        self.qnetwork_target.load_state_dict(torch.load(filename))
        logger.info(f"Model loaded from {filename}")
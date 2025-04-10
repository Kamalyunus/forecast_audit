"""
Linear Function Approximation Agent for forecast adjustment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os


class LinearAgent:
    """
    Q-Learning agent with linear function approximation for forecast adjustment.
    Learns to apply uplift/downlift to system-generated forecasts to improve accuracy.
    """
    
    def __init__(self, 
                feature_dim: int,
                action_size: int,
                learning_rate: float = 0.01,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995):
        """
        Initialize Linear Function Approximation Agent.
        
        Args:
            feature_dim: Dimension of state features
            action_size: Number of possible adjustment actions
            learning_rate: Learning rate for weight updates
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize weights for each action
        # We'll have a separate weight vector for each action
        self.weights = np.zeros((action_size, feature_dim))
        
        # Simple buffer for experience replay
        self.buffer_size = 10000
        self.buffer = []
        self.buffer_idx = 0
        
        # Feature normalization values
        self.feature_means = np.zeros(feature_dim)
        self.feature_stds = np.ones(feature_dim)
        self.feature_count = 0
        
        # Action statistics
        self.action_counts = np.zeros(action_size)
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to improve learning stability.
        Uses running mean and standard deviation.
        
        Args:
            features: Raw state features
            
        Returns:
            Normalized features
        """
        # Update feature statistics (during training)
        if self.feature_count < 10000:  # Only update for the first N observations
            self.feature_count += 1
            delta = features - self.feature_means
            self.feature_means += delta / self.feature_count
            
            if self.feature_count > 1:
                # Update running variance
                delta2 = features - self.feature_means
                var_sum = np.sum(delta * delta2, axis=0)
                self.feature_stds = np.sqrt(var_sum / self.feature_count + 1e-8)
        
        # Normalize features
        return (features - self.feature_means) / (self.feature_stds + 1e-8)
    
    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate Q-values for all actions given the state features.
        
        Args:
            features: State features
            
        Returns:
            Array of Q-values for each action
        """
        # Normalize features
        norm_features = self.normalize_features(features)
        
        # Calculate Q-values using linear function: Q(s,a) = w_a^T * features
        q_values = np.dot(self.weights, norm_features)
        
        return q_values
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select an adjustment action based on the current state.
        
        Args:
            state: Current state features
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Selected action index
        """
        q_values = self.get_q_values(state)
        
        # Epsilon-greedy policy
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_values)
        
        # Track action statistics
        self.action_counts[action] += 1
        
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> float:
        """
        Update the weights based on a single experience.
        
        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
            done: Whether the episode is done
            
        Returns:
            TD error magnitude
        """
        # Store experience in buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.buffer_idx] = (state, action, reward, next_state, done)
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        
        # Normalize states
        norm_state = self.normalize_features(state)
        norm_next_state = self.normalize_features(next_state)
        
        # Current Q-value: Q(s,a) = w_a^T * state
        current_q = np.dot(self.weights[action], norm_state)
        
        # Next state's best Q-value
        if done:
            next_q = 0
        else:
            next_q_values = np.dot(self.weights, norm_next_state)
            next_q = np.max(next_q_values)
        
        # Calculate target using standard Q-learning update
        target = reward + self.gamma * next_q
        
        # TD error
        td_error = target - current_q
        
        # Update weights for the selected action
        self.weights[action] += self.learning_rate * td_error * norm_state
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return abs(td_error)
    
    def batch_update(self, batch_size: int = 32) -> float:
        """
        Perform a batch update using experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Average TD error magnitude
        """
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample batch of experiences
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        total_error = 0.0
        
        for state, action, reward, next_state, done in batch:
            # Normalize states
            norm_state = self.normalize_features(state)
            norm_next_state = self.normalize_features(next_state)
            
            # Current Q-value
            current_q = np.dot(self.weights[action], norm_state)
            
            # Next state's best Q-value
            if done:
                next_q = 0
            else:
                next_q_values = np.dot(self.weights, norm_next_state)
                next_q = np.max(next_q_values)
            
            # Calculate target
            target = reward + self.gamma * next_q
            
            # TD error
            td_error = target - current_q
            
            # Update weights
            self.weights[action] += self.learning_rate * td_error * norm_state
            
            total_error += abs(td_error)
        
        return total_error / batch_size
    
    def calculate_adjusted_forecast(self, action_idx: int, forecast: float,
                                   adjustment_factors: List[float] = None) -> float:
        """
        Apply adjustment to forecast based on the selected action.
        
        Args:
            action_idx: Action index
            forecast: Original forecast value
            adjustment_factors: List of adjustment factors for each action
            
        Returns:
            Adjusted forecast
        """
        if adjustment_factors is None:
            # Default adjustment factors range from significant reduction to significant increase
            adjustment_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        # Ensure action index is valid
        action_idx = max(0, min(action_idx, len(adjustment_factors) - 1))
        
        # Apply factor to forecast
        factor = adjustment_factors[action_idx]
        adjusted_forecast = forecast * factor
        
        return max(0, adjusted_forecast)  # Ensure non-negative
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save the agent
        """
        data = {
            'weights': self.weights,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_count': self.feature_count,
            'feature_dim': self.feature_dim,
            'action_size': self.action_size,
            'action_counts': self.action_counts,
            'params': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'LinearAgent':
        """
        Load agent from file.
        
        Args:
            filepath: Path to load the agent from
            
        Returns:
            Loaded agent
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new agent
        agent = cls(
            feature_dim=data['feature_dim'],
            action_size=data['action_size'],
            learning_rate=data['params']['learning_rate'],
            gamma=data['params']['gamma'],
            epsilon_start=data['params']['epsilon'],
            epsilon_end=data['params']['epsilon_end'],
            epsilon_decay=data['params']['epsilon_decay']
        )
        
        # Load saved state
        agent.weights = data['weights']
        agent.feature_means = data['feature_means']
        agent.feature_stds = data['feature_stds']
        agent.feature_count = data['feature_count']
        agent.action_counts = data['action_counts']
        
        return agent


class ClusteredLinearAgent:
    """
    Manages multiple linear agents for different SKU clusters for improved scalability.
    """
    
    def __init__(self, 
                num_clusters: int,
                feature_dim: int,
                action_size: int,
                learning_rate: float = 0.01,
                gamma: float = 0.99):
        """
        Initialize a clustered linear agent.
        
        Args:
            num_clusters: Number of SKU clusters
            feature_dim: Dimension of state features
            action_size: Number of possible adjustment actions
            learning_rate: Learning rate for weight updates
            gamma: Discount factor
        """
        # Create one agent per cluster
        self.agents = [
            LinearAgent(
                feature_dim=feature_dim,
                action_size=action_size,
                learning_rate=learning_rate,
                gamma=gamma
            ) for _ in range(num_clusters)
        ]
        
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.cluster_mapping = {}  # Maps SKU ID to cluster ID
    
    def set_cluster_mapping(self, mapping: Dict[str, int]) -> None:
        """
        Set the mapping from SKU IDs to cluster IDs.
        
        Args:
            mapping: Dictionary mapping SKU ID to cluster ID
        """
        self.cluster_mapping = mapping
    
    def get_cluster_id(self, sku_id: str) -> int:
        """
        Get the cluster ID for a given SKU.
        
        Args:
            sku_id: SKU identifier
            
        Returns:
            Cluster ID
        """
        return self.cluster_mapping.get(sku_id, 0)
    
    def act(self, state: np.ndarray, sku_id: str, explore: bool = True) -> int:
        """
        Select an adjustment action for the given SKU based on its state.
        
        Args:
            state: Current state features
            sku_id: SKU identifier
            explore: Whether to use exploration
            
        Returns:
            Action index
        """
        cluster_id = self.get_cluster_id(sku_id)
        return self.agents[cluster_id].act(state, explore)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, sku_id: str) -> float:
        """
        Update the agent for the corresponding cluster.
        
        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
            done: Whether the episode is done
            sku_id: SKU identifier
            
        Returns:
            TD error magnitude
        """
        cluster_id = self.get_cluster_id(sku_id)
        return self.agents[cluster_id].update(state, action, reward, next_state, done)
    
    def batch_update(self, batch_size: int = 32) -> float:
        """
        Perform batch updates for all clusters.
        
        Args:
            batch_size: Batch size for each cluster
            
        Returns:
            Average TD error magnitude across all clusters
        """
        total_error = 0
        for agent in self.agents:
            total_error += agent.batch_update(batch_size)
        return total_error / len(self.agents)
    
    def calculate_adjusted_forecast(self, action_idx: int, forecast: float, 
                                  sku_id: Optional[str] = None) -> float:
        """
        Calculate adjusted forecast based on action and original forecast.
        
        Args:
            action_idx: Action index
            forecast: Original forecast value
            sku_id: Optional SKU ID (not used in calculation but included for API consistency)
            
        Returns:
            Adjusted forecast
        """
        # All clusters use the same adjustment factors
        adjustment_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        action_idx = max(0, min(action_idx, len(adjustment_factors) - 1))
        
        factor = adjustment_factors[action_idx]
        adjusted_forecast = forecast * factor
        return max(0, adjusted_forecast)
    
    def save(self, base_filepath: str) -> None:
        """
        Save all cluster agents to files.
        
        Args:
            base_filepath: Base path for saving agent files
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.agents):
            filepath = f"{base_filepath}_cluster_{i}.pkl"
            agent.save(filepath)
        
        # Save cluster mapping
        with open(f"{base_filepath}_mapping.pkl", 'wb') as f:
            pickle.dump(self.cluster_mapping, f)
        
        # Save metadata
        metadata = {
            'num_clusters': self.num_clusters,
            'feature_dim': self.feature_dim,
            'action_size': self.action_size
        }
        with open(f"{base_filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, base_filepath: str) -> 'ClusteredLinearAgent':
        """
        Load a clustered agent from files.
        
        Args:
            base_filepath: Base path for loading agent files
            
        Returns:
            Loaded clustered agent
        """
        # Load metadata
        with open(f"{base_filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create agent with metadata
        agent = cls(
            num_clusters=metadata['num_clusters'],
            feature_dim=metadata['feature_dim'],
            action_size=metadata['action_size']
        )
        
        # Load individual cluster agents
        for i in range(metadata['num_clusters']):
            filepath = f"{base_filepath}_cluster_{i}.pkl"
            if os.path.exists(filepath):
                agent.agents[i] = LinearAgent.load(filepath)
        
        # Load cluster mapping
        mapping_path = f"{base_filepath}_mapping.pkl"
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                agent.cluster_mapping = pickle.load(f)
        
        return agent
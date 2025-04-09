import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the GRU-based Q-learning model for Binary Classification
class GRU_QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU_QNet, self).__init__()
        
        # GRU layer (input size = input_dim, hidden size = hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer to output Q-values for binary classification (2 possible actions)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Pass through the GRU
        _, h_n = self.gru(x)  # h_n is the last hidden state, shape: [1, batch_size, hidden_dim]
        
        # Take the last hidden state and pass it through the FC layer
        q_values = self.fc(h_n.squeeze(0))  # Squeeze to get shape: [batch_size, output_dim]
        return q_values


# Define the Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


# Define the DQN Agent for Binary Classification
class DQNAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.99, epsilon=0.1, lr=0.001, batch_size=64, buffer_size=10000):
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration factor
        self.batch_size = batch_size
        
        # Initialize Q-network and target Q-network
        self.q_net = GRU_QNet(input_dim, hidden_dim, output_dim).to(device)
        self.target_q_net = GRU_QNet(input_dim, hidden_dim, output_dim).to(device)
        
        # Copy weights from Q-network to target Q-network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # Random action, binary classification (0 or 1)
        else:
            with torch.no_grad():
                state = state.to(device)  # Move state to GPU
                q_values = self.q_net(state)
                return q_values.argmax().item()
    
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert batch to tensors and move to GPU
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in states])
        next_states = torch.stack([torch.tensor(ns, dtype=torch.float32).to(device) for ns in next_states])
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Get current Q-values from the Q-network
        current_q_values = self.q_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss (Mean Squared Error)
        loss = torch.mean((current_q_values - target_q_values) ** 2)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        # Update the target Q-network with the weights from the Q-network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
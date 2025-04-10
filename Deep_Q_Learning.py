import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Generate Small Random Data for Pipeline Testing
num_samples = 100
sequence_length = 17
features = 17

data_X = torch.randn(num_samples, sequence_length, features)
data_y = torch.randn(num_samples, sequence_length, features)

train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

indices = torch.randperm(num_samples)
X_train = data_X[indices[:train_size]]
y_train = data_y[indices[:train_size]]
X_val = data_X[indices[train_size:train_size + val_size]]
y_val = data_y[indices[train_size:train_size + val_size]]
X_test = data_X[indices[train_size + val_size:]]
y_test = data_y[indices[train_size + val_size:]]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Step 2: Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, nhid, num_layers):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=nhid)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=nhid)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(input_dim, output_dim)
    
    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        return self.fc_out(output)

# Step 3: Normalization and Denormalization
def normalize_data(data, ranges):
    data = data.clone()
    for i in range(data.size(2)):
        min_val, max_val = ranges[f'feature_{i+1}']
        range_diff = max_val - min_val if max_val != min_val else 1e-8
        data[:, :, i] = (data[:, :, i] - min_val) / range_diff
    return data

def denormalize_data(data, ranges):
    data = data.clone()
    for i in range(data.size(2)):
        min_val, max_val = ranges[f'feature_{i+1}']
        data[:, :, i] = data[:, :, i] * (max_val - min_val) + min_val
    return data

# Step 4: Range-Aware Loss
class RangeAwareLoss(nn.Module):
    def __init__(self, ranges):
        super(RangeAwareLoss, self).__init__()
        self.ranges = ranges

    def forward(self, output, target):
        mse_loss = F.mse_loss(output, target)
        penalty = 0
        for i in range(output.size(2)):
            min_val, max_val = self.ranges[f'feature_{i+1}']
            clamped = torch.clamp(output[:, :, i], min=min_val, max=max_val)
            penalty += torch.mean((clamped - output[:, :, i]) ** 2)
        return mse_loss + 0.1 * penalty

# Step 5: Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0
    
    def push(self, state, action, attack_label, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, attack_label, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Step 6: GRU-based Q-Network
class GRU_QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU_QNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        elif x.dim() == 3:
            x = x.squeeze(1)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# Step 7: DQN Agent
class DQNAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, buffer_size=10000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.q_net = GRU_QNet(input_dim, hidden_dim, output_dim).to(device)
        self.target_q_net = GRU_QNet(input_dim, hidden_dim, output_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state = state.to(device)
            q_values = self.q_net(state)
            return q_values.argmax().item()
    
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
        attack_labels = torch.tensor([b[2] for b in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor([b[5] for b in batch], dtype=torch.float32).to(device)

        states = states.squeeze(2)
        next_states = next_states.squeeze(2)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def evaluate(self, X_test, y_test):
        self.q_net.eval()
        correct = 0
        total = X_test.size(0)
        
        with torch.no_grad():
            for i in range(total):
                state = X_test[i].unsqueeze(0).to(device)
                action = self.select_action(state)
                attack_label = random.randint(0, 1)
                if action == attack_label:
                    correct += 1
        
        accuracy = correct / total
        self.q_net.train()
        return accuracy
    
    def save(self, filepath='dqn_agent.pth'):
        try:
            torch.save(self.q_net.state_dict(), filepath)
            print(f"DQN agent model saved to '{filepath}'")
            if os.path.exists(filepath):
                print(f"Confirmed: '{filepath}' exists.")
            else:
                print(f"Warning: '{filepath}' was not found after saving.")
        except Exception as e:
            print(f"Error saving DQN agent model: {e}")

# Step 8: Transformer-based Environment
class NetworkSecurityEnv:
    def __init__(self, transformer_model, ranges, X_train, sequence_length=17, max_steps=10):
        self.transformer_model = transformer_model
        self.ranges = ranges
        self.X_train = X_train.to(device)
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        self.attack_label = None
    
    def reset(self):
        idx = random.randint(0, self.X_train.size(0) - 1)
        src = self.X_train[idx].unsqueeze(1)
        tgt = torch.randn(self.sequence_length, 1, 17).to(device)
        
        src_normalized = normalize_data(src, self.ranges).to(device)
        tgt_normalized = normalize_data(tgt, self.ranges).to(device)
        
        with torch.no_grad():
            attack_free_state = self.transformer_model(src_normalized, tgt_normalized)
        
        self.attack_label = random.randint(0, 1)
        self.state = attack_free_state + (self.attack_label * torch.randn(self.sequence_length, 1, 17).to(device) * 0.5)
        
        self.current_step = 0
        return self.state.clone().detach()

    def step(self, action):
        self.current_step += 1
        
        reward = 1.0 if action == self.attack_label else -1.0
        
        idx = random.randint(0, self.X_train.size(0) - 1)
        src = self.X_train[idx].unsqueeze(1)
        tgt = torch.randn(self.sequence_length, 1, 17).to(device)
        
        src_normalized = normalize_data(src, self.ranges).to(device)
        tgt_normalized = normalize_data(tgt, self.ranges).to(device)
        
        with torch.no_grad():
            attack_free_state = self.transformer_model(src_normalized, tgt_normalized)
        
        self.attack_label = random.randint(0, 1)
        self.state = attack_free_state + (self.attack_label * torch.randn(self.sequence_length, 1, 17).to(device) * 0.5)
        
        done = self.current_step >= self.max_steps
        return self.state.clone().detach(), self.attack_label, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, State Mean: {self.state.mean().item():.4f}, Attack Label: {self.attack_label}")

# Step 9: Train Transformer with Early Stopping
def train_transformer(X_train, y_train, X_val, y_val):
    input_dim = 17
    output_dim = 17
    nhead = 1
    nhid = 128
    num_layers = 2
    
    ranges = {f'feature_{i+1}': (0.0, 1.0) for i in range(17)}
    
    transformer_model = TransformerModel(input_dim, output_dim, nhead, nhid, num_layers).to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
    criterion = RangeAwareLoss(ranges)
    
    num_epochs = 100
    batch_size = 32
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    for epoch in range(num_epochs):
        transformer_model.train()
        indices = torch.randperm(X_train.size(0))
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            batch_indices = indices[i:min(i + batch_size, X_train.size(0))]
            src = X_train[batch_indices]
            tgt = y_train[batch_indices]
            
            src_normalized = normalize_data(src, ranges)
            tgt_normalized = normalize_data(tgt, ranges)
            
            optimizer.zero_grad()
            output = transformer_model(src_normalized, tgt_normalized)
            loss = criterion(output, tgt_normalized)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_indices)
        
        train_loss /= X_train.size(0)
        
        transformer_model.eval()
        with torch.no_grad():
            src_val_normalized = normalize_data(X_val, ranges)
            tgt_val_normalized = normalize_data(y_val, ranges)
            val_output = transformer_model(src_val_normalized, tgt_val_normalized)
            val_loss = criterion(val_output, tgt_val_normalized).item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(transformer_model.state_dict(), 'transformer_model.pth')
            print("Model saved (improved validation loss)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return transformer_model, ranges

# Step 10: Train DQN and Save Agent
def train_dqn(transformer_model, ranges, X_train, X_test, y_test):
    env = NetworkSecurityEnv(transformer_model, ranges, X_train)
    agent = DQNAgent(input_dim=17, hidden_dim=64, output_dim=2)
    
    num_episodes = 200
    print("Starting DQN training...")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, attack_label, reward, done, _ = env.step(action)
            
            agent.replay_buffer.push(state.cpu().numpy(), action, attack_label, reward, next_state.cpu().numpy(), done)
            agent.update()
            
            state = next_state
            total_reward += reward
            
            if done:
                agent.update_target_network()
        
        if (episode + 1) % 10 == 0:
            print(f"DQN Episode [{episode+1}/{num_episodes}], Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            accuracy = agent.evaluate(X_test.to(device), y_test.to(device))
            print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save the DQN agent after training
    print("DQN training completed. Saving agent model...")
    agent.save('dqn_agent.pth')

# Step 11: Deploy DQN Agent
def deploy_dqn_agent(filepath='dqn_agent.pth', input_dim=17, hidden_dim=64, output_dim=2):
    model = GRU_QNet(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"DQN agent loaded from '{filepath}'")
    
    sample_state = torch.randn(sequence_length, 1, input_dim).to(device)
    with torch.no_grad():
        q_values = model(sample_state)
        action = q_values.argmax().item()
        print(f"Sample state inference: Q-values = {q_values}, Action = {action}")
    return model

# Main execution
if __name__ == "__main__":
    train_transformer_flag = True
    
    if train_transformer_flag or not os.path.exists('transformer_model.pth'):
        print("Training Transformer Model (Attack-Free Data Diversifier)...")
        transformer_model, ranges = train_transformer(X_train, y_train, X_val, y_val)
    else:
        print("Loading Pre-trained Transformer Model...")
        transformer_model = TransformerModel(input_dim=17, output_dim=17, nhead=1, nhid=128, num_layers=2).to(device)
        transformer_model.load_state_dict(torch.load('transformer_model.pth'))
        transformer_model.eval()
        ranges = {f'feature_{i+1}': (0.0, 1.0) for i in range(17)}
        print("Transformer model loaded from 'transformer_model.pth'")

    print("\nTraining DQN Agent (Attack Classifier)...")
    train_dqn(transformer_model, ranges, X_train, X_test, y_test)
    
    print("\nDeploying DQN Agent for inference...")
    deployed_model = deploy_dqn_agent()
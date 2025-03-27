import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Transformer-based Belief State Environment for Time Series Classification
class BeliefStateTransformerEnv:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.feature_dim = data.shape[2]

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=2)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def reset(self):
        self.current_index = random.randint(0, len(self.data) - 1)
        self.sequence = self.data[self.current_index]  # Shape: [seq_len, feature_dim]
        self.label = self.labels[self.current_index]
        return self.get_belief_state()

    def step(self, action):
        reward = 1 if action == self.label else -1
        done = True
        return None, reward, done

    def get_belief_state(self):
        seq = self.sequence.unsqueeze(1)  # [seq_len, 1, feature_dim]
        belief = self.transformer(seq).mean(dim=0)  # [1, feature_dim]
        return belief.squeeze(0)  # [feature_dim] ← single sample



# GRU-based Deep Q-Network (Teacher Model)
class DQN_GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 due to bidirection

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # take last time step
        return x

# GRU-based Smaller Deep Q-Network (Student Model)
class DQN_GRU_Student(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(DQN_GRU_Student, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # last timestep output
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Distillation Loss Function

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    soft_teacher = nn.functional.log_softmax(teacher_logits / T, dim=1)
    soft_student = nn.functional.log_softmax(student_logits / T, dim=1)
    soft_loss = nn.functional.kl_div(soft_student, soft_teacher.exp(), reduction='batchmean') * (T ** 2)
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Distill Student Network

def distill_student(student, teacher, env, epochs=10, batch_size=32, lr=0.001):
    optimizer = optim.Adam(student.parameters(), lr=lr)
    buffer = ReplayBuffer()

    for _ in range(100):
        state = env.reset()
        done = False
        with torch.no_grad():
            action = torch.argmax(teacher(state)).item()
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)

    for epoch in range(epochs):
        if len(buffer) < batch_size:
            continue
        batch = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)

        with torch.no_grad():
            teacher_logits = teacher(states)
        student_logits = student(states)
        loss = distillation_loss(student_logits, teacher_logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Distillation Loss: {loss.item():.4f}")

# Generate Sample Data

def generate_data(num_samples=1000, seq_length=10, feature_dim=16):
    data = np.random.rand(num_samples, seq_length, feature_dim).astype(np.float32)
    labels = np.random.randint(0, 2, num_samples)
    return torch.tensor(data), labels

# Train Teacher Model

def train_dqn(env, num_episodes=1000, batch_size=32, gamma=0.99, lr=0.001, epsilon=0.1, patience=20):
    input_dim = env.data.shape[2]
    output_dim = 2
    q_network = DQN_GRU(input_dim, output_dim)
    target_network = DQN_GRU(input_dim, output_dim)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()
    best_loss = float('inf')
    patience_counter = 0
    save_path = "dqn_gru_model.pth"

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(state)).item()

            next_state, reward, done = env.step(action)
            total_reward += reward
            buffer.push(state, action, reward, next_state, done)
            state = next_state if next_state is not None else state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.stack(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
                non_final_next_states = torch.stack([s for s in next_states if s is not None])

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = torch.zeros(batch_size)
                next_q_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()
                targets = rewards + gamma * next_q_values
                loss = loss_fn(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(q_network.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Loss: {loss.item():.4f}")

    return q_network, save_path

# Main

data, labels = generate_data()
env = BeliefStateTransformerEnv(data, labels)

# Train Teacher
trained_q_network, model_path = train_dqn(env)

# Load Teacher
teacher = DQN_GRU(input_dim=data.shape[2], output_dim=2)
teacher.load_state_dict(torch.load(model_path))
teacher.eval()

# Train Student
student = DQN_GRU_Student(input_dim=data.shape[2], output_dim=2)
distill_student(student, teacher, env)
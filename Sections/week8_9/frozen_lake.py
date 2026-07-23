import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# Q-Approzimator
class QNetwork(nn.Module):
    def __init__(self, state_dim=16, action_dim=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x) # Outputs Q-value for all 4 actions

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# The DQN-Agent
class DQNAgent:
    def __init__(self, state_dim=16, action_dim=4, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_state_tensor(self, state):
        return F.one_hot(torch.tensor(state), num_classes=self.state_dim).float()

    def select_action(self, state):
        # Epsilon-Greedy Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Exploitation
        state_tensor = self.get_state_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_tensor = F.one_hot(torch.tensor(states), num_classes=self.state_dim).float()
        next_states_tensor = F.one_hot(torch.tensor(next_states), num_classes=self.state_dim).float()
        
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get Current Q(S_t, A_t; theta) 
        current_q = self.q_net(states_tensor).gather(1, actions_tensor)

        # Get TD Target 
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q = rewards_tensor + (self.gamma * max_next_q * (1 - dones_tensor))

        # Loss and Gradient Descent ---
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# Training loop and results
def train():
    # Initialize deterministic FrozenLake
    env = gym.make('FrozenLake-v1', is_slippery=True, map_name="4x4")
    agent = DQNAgent()
    buffer = ReplayBuffer(capacity=5000)
    
    batch_size = 32
    episodes = 3000
    target_update_freq = 10
    
    rewards_history = []

    print("Training started...")
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            shaped_reward = reward if reward == 1.0 else -0.01
            if done and reward == 0:
                shaped_reward = -1.0 # Fell in a hole

            buffer.push(state, action, shaped_reward, next_state, done)
            agent.train_step(buffer, batch_size)
            
            state = next_state
            total_reward += reward # Track actual reward (0 or 1), not shaped reward
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1} | Win Rate (Last 100): {avg_reward * 100:.1f}% | Epsilon: {agent.epsilon:.2f}")

    env.close()

    # PLOTTING
    plt.figure(figsize=(10, 5))
    rolling_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
    plt.plot(rolling_avg, color='blue', linewidth=2)
    plt.title("DQN Learning Curve on FrozenLake-v1 (is_slippery=True)")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (Rolling 50-Episode Average)")
    plt.grid(True)
    plt.savefig("stochastic_curve.png", dpi=300, bbox_inches='tight')
    print("Graph saved as stochastic_curve.png!")
    plt.show(block=True)

print("Script is loading... about to call train()")
train()
print("Script completely finished!")
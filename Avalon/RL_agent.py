import random
import numpy as np
from avalon_game import AvalonEnvironment
import torch
import torch.nn as nn
import torch.optim as optim

class StatePredictor(nn.Module):
    def __init__(self, state_size, action_size):
        super(StatePredictor, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, state_size)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        next_state = self.fc2(x)
        return next_state

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=1)
        return action_probs

class CustomRLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.state_predictor = StatePredictor(state_size, action_size)
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(list(self.state_predictor.parameters()) + list(self.policy_network.parameters()), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor).detach().numpy().squeeze()
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def learn(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.zeros(1, self.action_size)
        action_tensor[0, action] = 1

        # 状态预测
        predicted_next_state = self.state_predictor(state_tensor, action_tensor)
        state_loss = self.criterion(predicted_next_state, next_state_tensor)

        # 策略优化
        action_probs = self.policy_network(state_tensor)
        log_prob = torch.log(action_probs.squeeze(0)[action])
        policy_loss = -log_prob * reward

        # 总损失
        loss = state_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 训练自定义强化学习代理
env = AvalonEnvironment()
agent = CustomRLAgent(state_size=len(env.get_state()), action_size=len(env.available_actions()))

# 训练循环
for episode in range(1000):
    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    print(f"Episode {episode}: Total Reward: {total_reward}")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import logging
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 编码棋子特征
piece_encoding = {
    '地雷': 1,
    '炸弹': 2,
    '司令': 3,
    '军长': 4,
    '师长': 5,
    '旅长': 6,
    '团长': 7,
    '营长': 8,
    '连长': 9,
    '排长': 10,
    '工兵': 11,
    '军旗': 12,
}

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_value = nn.Linear(128, 1)
        self.fc3_advantage = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state.reshape(state.size(0), -1)))
        x = torch.relu(self.fc2(x))
        value = self.fc3_value(x)
        advantage = self.fc3_advantage(x)
        return value + advantage - advantage.mean()

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=13, num_features=5, hidden_size=128):
        super(PolicyValueNet, self).__init__()
        self.board_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.board_bn1 = nn.BatchNorm2d(32)
        self.board_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.board_bn2 = nn.BatchNorm2d(64)
        
        self.prob_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.prob_bn = nn.BatchNorm2d(32)
        
        self.hidden_size = hidden_size
        
        self.policy_head = nn.Linear(hidden_size, board_size * board_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, board_state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        if board_state.size(1) != 3:
            board_state = board_state[:, :3, :, :]  # 只取前3个通道

        board_state = F.relu(self.board_bn1(self.board_conv1(board_state)))
        board_state = F.relu(self.board_bn2(self.board_conv2(board_state)))
        board_state = board_state.reshape(board_state.size(0), -1)  # 确保 board_state 是 2 维张量
        
        if prob_matrix.dim() == 2:
            prob_matrix = prob_matrix.unsqueeze(0).unsqueeze(0)  # 增加两个维度以符合卷积层输入要求
        elif prob_matrix.dim() == 3:
            prob_matrix = prob_matrix.unsqueeze(1)  # 增加一个维度以符合卷积层输入要求

        prob_matrix = F.relu(self.prob_bn(self.prob_conv(prob_matrix)))
        prob_matrix = prob_matrix.reshape(prob_matrix.size(0), -1)  # 确保 prob_matrix 是 2 维张量
        
        turn = turn.reshape(turn.size(0), -1)  # 确保 turn 是 2 维张量
        num_own_pieces = num_own_pieces.reshape(num_own_pieces.size(0), -1)  # 确保 num_own_pieces 是 2 维张量
        num_opponent_pieces = num_opponent_pieces.reshape(num_opponent_pieces.size(0), -1)  # 确保 num_opponent_pieces 是 2 维张量
        features = features.reshape(features.size(0), -1)  # 确保 features 是 2 维张量

        combined = torch.cat([board_state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features], dim=1)
        
        # 动态创建 fc1 层
        self.fc1 = nn.Linear(combined.size(1), self.hidden_size).to(combined.device)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_size).to(combined.device)

        x = self.fc1(combined)
        if self.training and combined.size(0) > 1:
            x = F.relu(self.fc1_bn(x))
        else:
            x = F.relu(x)
        
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        
        return policy, value

    def pad_input(self, tensor, target_channels):
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # 增加一个维度以符合卷积层输入要求

        if tensor.size(1) < target_channels:
            padding = torch.zeros(tensor.size(0), target_channels - tensor.size(1), tensor.size(2), tensor.size(3)).to(self.device)
            tensor = torch.cat((tensor, padding), dim=1)
        elif tensor.size(1) > target_channels:
            tensor = tensor[:, :target_channels, :, :]

        return tensor
class MCTS:
    def __init__(self, env, num_simulations):
        self.env = env
        self.num_simulations = num_simulations

    def search(self, initial_state, player_color, pi, pi_reg):
        for _ in range(self.num_simulations):
            self._simulate(initial_state, player_color, pi, pi_reg)

        return self._best_actions(initial_state, player_color)

    def _simulate(self, state, player_color, pi, pi_reg):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(state)

        for _ in range(100):  # 限制每次模拟的深度
            valid_actions = env_copy.get_valid_actions(player_color)
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            next_state, reward, done, _ = env_copy.step_cpy(action, pi, pi_reg,player_color)  # 解包返回值并正确处理
            if isinstance(env_copy.state, str) and env_copy.state in ['red_wins', 'blue_wins']:
                break

    def _best_actions(self, state, player_color):
        valid_actions = self.env.get_valid_actions(player_color)
        best_actions = sorted(valid_actions, key=lambda a: self._evaluate_action(state, a,player_color), reverse=True)[:20]
        return best_actions

    def _evaluate_action(self, state, action,player_color):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(state)
        next_state, reward, done, _ = env_copy.step_cpy(action, self.env.pi, self.env.pi_reg,player_color)
        return reward
class DQNAgent:
    def __init__(self, state_size, action_size, env, seed, gamma=0.99, lr=0.001, buffer_size=50000, batch_size=64, update_every=4, tau=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.mcts = MCTS(env, num_simulations=50)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = DuelingQNetwork(self.state_size, action_size, seed).to(self.device)
        self.qnetwork_target = DuelingQNetwork(self.state_size, action_size, seed).to(self.device)
        self.policy_value_net = PolicyValueNet(board_size=13, num_features=5, hidden_size=128).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.policy_value_optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0
        self.eta = 0.01
        epsilon = 1e-8
        self.pi = np.full(action_size, (1.0 / action_size) + epsilon)
        self.pi_reg = np.full(action_size, (1.0 / action_size) + epsilon)

    def get_prob_matrix(self, state):
        prob_matrix = np.zeros((self.env.board_rows, self.env.board_cols))
        for piece in self.env.red_pieces:
            if piece.position:
                x, y = piece.position
                prob_matrix[x, y] = self.env.piece_name_to_value(piece.get_name())
        return prob_matrix
            
    def act(self, state, turn, num_own_pieces, num_opponent_pieces, features, player_color):
        print(f"Current player color in act: {player_color}")

        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # 调整维度
        state_tensor = state_tensor.reshape(1, 3, self.env.board_rows, self.env.board_cols)  # 确保是 (1, 3, board_rows, board_cols)
    
        prob_matrix = self.get_prob_matrix(state)
        prob_matrix_tensor = torch.FloatTensor(prob_matrix).unsqueeze(0).unsqueeze(0).to(self.device)
        turn_tensor = torch.FloatTensor([turn]).unsqueeze(0).to(self.device)
        num_own_pieces_tensor = torch.FloatTensor([num_own_pieces]).unsqueeze(0).to(self.device)
        num_opponent_pieces_tensor = torch.FloatTensor([num_opponent_pieces]).unsqueeze(0).to(self.device)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        self.policy_value_net.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor).squeeze()  # 确保 action_values 是 1 维张量
            policy, value_estimation = self.policy_value_net(state_tensor, prob_matrix_tensor, turn_tensor, num_own_pieces_tensor, num_opponent_pieces_tensor, features_tensor)
            self.pi = policy.cpu().numpy().flatten()  # 更新 pi 为最新策略
            self.pi_reg = self.pi.copy() 
            policy = policy.squeeze()  # 确保 policy 是 1 维张量
            if value_estimation.dim() == 0:
                value_estimation = value_estimation.unsqueeze(0)  # 如果 value_estimation 是标量，则扩展为 1 维张量

            # 确保所有张量的大小一致
            max_len = max(action_values.size(0), policy.size(0), value_estimation.size(0))
            if action_values.size(0) < max_len:
                action_values = F.pad(action_values, (0, max_len - action_values.size(0)), "constant", 0)
            if policy.size(0) < max_len:
                policy = F.pad(policy, (0, max_len - policy.size(0)), "constant", 0)
            if value_estimation.size(0) < max_len:
                value_estimation = F.pad(value_estimation, (0, max_len - value_estimation.size(0)), "constant", 0)

        self.qnetwork_local.train()
        self.policy_value_net.train()

        combined_values = action_values + policy + value_estimation

        # 获取所有有效动作
        valid_actions = self.env.get_valid_actions(player_color)
        if not valid_actions:
            raise ValueError("No valid actions available.")

        # 使用蒙特卡洛树搜索筛选出一系列潜在的良好动作
        mcts_actions = self.mcts.search(self.env.get_state(), player_color, self.pi, self.pi_reg)
        mcts_valid_actions = [a for a in mcts_actions if a in valid_actions]

        # 在蒙特卡洛树搜索筛选出的动作中进行强化学习策略的评估
        if random.random() > self.epsilon:
            action = mcts_valid_actions[np.argmax([combined_values[a] for a in mcts_valid_actions])]
        else:
            action = random.choice(mcts_valid_actions)

        self.pi = policy.cpu().numpy()  # 将策略值保存为 numpy 数组
        self.pi_reg = self.pi.copy()    # 将正则化策略值保存为 numpy 数组
        print(f"Selected action: {action}")
        return action, self.pi, self.pi_reg

    def step(self, state, action, reward, next_state, done):
        if action >= len(self.pi):
            self.pi = np.pad(self.pi, (0, action - len(self.pi) + 1), 'constant', constant_values=0)
            self.pi_reg = np.pad(self.pi_reg, (0, action - len(self.pi_reg) + 1), 'constant', constant_values=0)
        
        transformed_reward = self.reward_transformation(reward, self.pi[action], self.pi_reg[action], self.eta)
        self.memory.add(state, action, transformed_reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, indices, weights = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = (Q_expected - Q_targets).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        q_values = self.qnetwork_local(states).detach().cpu().numpy()
        self.pi = self.replicator_dynamics_update(self.pi, q_values.mean(axis=0))

    def replicator_dynamics_update(self, pi, q_values, learning_rate=0.01):
        new_pi = pi * np.exp(learning_rate * q_values)
        new_pi /= new_pi.sum()
        return new_pi

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reward_transformation(self, reward, pi, pi_reg, eta):
        if pi == 0:
            pi = 1e-10
        if pi_reg == 0:
            pi_reg = 1e-10

        return reward - eta * np.log(pi / pi_reg)

    def fine_tune_policy(self, policy):
        threshold = 0.01  # 阈值可以根据需要调整
        policy[policy < threshold] = 0
        policy /= policy.sum()
        return policy

class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        max_priority = max(self.priorities, default=1.0)
        self.memory.append(e)
        self.priorities.append(max_priority)

    def sample(self):
        self.frame += 1
        beta = self.beta_by_frame(self.frame)
        
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        states = torch.tensor(np.array([e.state for e in experiences]), device=device, dtype=torch.float32)
        actions = torch.tensor(np.array([e.action for e in experiences]), device=device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array([e.reward for e in experiences]), device=device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), device=device, dtype=torch.float32)
        dones = torch.tensor(np.array([e.done for e in experiences]), device=device, dtype=torch.float32).unsqueeze(1)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + self.epsilon

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return len(self.memory)

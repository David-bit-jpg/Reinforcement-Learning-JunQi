import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 编码棋子特征
piece_encoding = {
    '军旗': 0,
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
    '工兵': 11
}

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self.scale_noise(self.out_features))

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)  # 包括棋子特征
        self.fc2 = nn.Linear(128, 128)
        
        self.fc3_adv = NoisyLinear(128, 128)
        self.fc4_adv = NoisyLinear(128, action_size)
        
        self.fc3_val = NoisyLinear(128, 128)
        self.fc4_val = NoisyLinear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        adv = torch.relu(self.fc3_adv(x))
        adv = self.fc4_adv(adv)

        val = torch.relu(self.fc3_val(x))
        val = self.fc4_val(val)
        
        return val + adv - adv.mean()

class DQNAgent:
    def __init__(self, state_size, action_size, env, seed, gamma=0.8, lr=0.001, buffer_size=50000, batch_size=100, update_every=1, tau=0.001, temperature=2.0, epsilon=0.1):
        self.state_size = state_size + 1  # +1 用于棋子特征
        self.action_size = action_size
        self.env = env
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.temperature = temperature
        self.epsilon = epsilon  # 添加 epsilon 参数

        self.qnetwork_local = DuelingQNetwork(self.state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(self.state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, self.state_size, seed)
        self.t_step = 0
        self.episode_memory = []  # 记录每一轮的动作和奖励
        self.episode_rewards = []  # 记录每一轮的奖励
        self.rng = np.random.default_rng(seed)  # 使用 numpy 的随机数生成器
        
    def select_piece_by_priority(self):
        available_pieces = [p for p in (self.env.red_pieces if self.env.current_player == 'red' else self.env.blue_pieces) if p.get_position() is None]
        if not available_pieces:
            return None  # 如果没有可用棋子，返回 None
        for priority_piece in ['军旗', '地雷', '炸弹']:
            for piece in available_pieces:
                if piece.get_name() == priority_piece:
                    return piece
        return self.rng.choice(available_pieces)

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        piece = self.select_piece_by_priority()
        if piece is None:
            print("No pieces available, ending the setup phase.")
            return None  # 返回 None 表示结束布局阶段

        piece_feature = piece_encoding[piece.get_name()]
        state_with_piece = np.append(state, piece_feature)

        state_tensor = torch.from_numpy(state_with_piece).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor).cpu().numpy().flatten()
        self.qnetwork_local.train()

        valid_positions = [pos for pos in piece.get_valid_positions() if pos not in (self.env.occupied_positions_red if piece.get_color() == 'red' else self.env.occupied_positions_blue)]
        camps = [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]
        valid_positions = [pos for pos in valid_positions if pos not in camps]

        if not valid_positions:
            self.env.reward_model.total_reward -= 1000
            return None

        # 使用epsilon-greedy策略选择动作
        if np.random.rand() < epsilon:
            # 随机选择一个有效位置
            target_position = self.rng.choice(valid_positions)
        else:
            # 使用神经网络的预测值来选择最佳位置
            valid_action_values = np.array([action_values[self.env.encode_action(piece, pos)] for pos in valid_positions])
            best_position_index = np.argmax(valid_action_values)
            target_position = valid_positions[best_position_index]

        action = self.env.encode_action(piece, target_position)

        # 记录动作和当前状态
        self.episode_memory.append((state_with_piece, action))
        
        return action

    def step(self, state, action, reward, next_state, done):
        piece = self.select_piece_by_priority()
        if piece is None:
            piece_feature = 0  # 使用默认特征值
            done = True  # 设置 done 标志，结束布局阶段
            self.episode_rewards.append(reward)  # 记录最后一步的奖励
            self.reflect_rewards()  # 反思奖励并学习
            return

        piece_feature = piece_encoding[piece.get_name()]
        state_with_piece = np.append(state, piece_feature)
        next_state_with_piece = np.append(next_state, piece_feature)

        self.memory.add(state_with_piece, action, reward, next_state_with_piece, done)
        self.episode_rewards.append(reward)  # 记录每一步的奖励
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def reflect_rewards(self):
        # 反思整个布局过程中的奖励，并更新网络权重
        cumulative_reward = sum(self.episode_rewards)
        for state, action in self.episode_memory:
            self.memory.add(state, action, cumulative_reward, state, True)
        
        # 清空奖励和动作记录
        self.episode_rewards = []
        self.episode_memory = []

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, state_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.state_size = state_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        states = states.view(-1, self.state_size)
        next_states = next_states.view(-1, self.state_size)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

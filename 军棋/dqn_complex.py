import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_size, action_size, seed, atoms=51, Vmin=-10, Vmax=10):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = float(Vmax - Vmin) / (atoms - 1)
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        self.fc3_adv = NoisyLinear(128, 128)
        self.fc4_adv = NoisyLinear(128, action_size * atoms)
        
        self.fc3_val = NoisyLinear(128, 128)
        self.fc4_val = NoisyLinear(128, atoms)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        adv = torch.relu(self.fc3_adv(x))
        adv = self.fc4_adv(adv).view(-1, self.atoms)

        val = torch.relu(self.fc3_val(x))
        val = self.fc4_val(val).view(-1, 1, self.atoms)
        
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_atoms = val + adv - adv_mean
        q_dist = torch.softmax(q_atoms, dim=-1)
        q = torch.sum(q_dist * self.support, dim=-1)
        
        return q, q_dist
    
class RainbowAgent:
    def __init__(self, state_size, action_size, env, seed, gamma=0.99, lr=0.0005, buffer_size=100000, batch_size=64, update_every=4, n_step=3, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.n_step = n_step
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Vmin = -10
        self.Vmax = 10
        self.atoms = 51
        self.delta_z = float(self.Vmax - self.Vmin) / (self.atoms - 1)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, self.atoms, self.Vmin, self.Vmax).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, self.atoms, self.Vmin, self.Vmax).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, state_size, seed, n_step=self.n_step, gamma=self.gamma)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps):
        self.eps = eps
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = state.view(-1, self.state_size)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values, _ = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        piece = self.select_piece_by_priority()
        valid_positions = [pos for pos in piece.get_valid_positions() if pos not in (self.env.occupied_positions_red if piece.get_color() == 'red' else self.env.occupied_positions_blue)]
        camps = [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]

        valid_positions = [pos for pos in valid_positions if pos not in camps]

        if not valid_positions:
            self.env.reward_model.total_reward -= 1000
            return action

        target_position = random.choice(valid_positions)
        action = self.env.encode_action(piece, target_position)

        return action

    def select_piece_by_priority(self):
        available_pieces = [p for p in (self.env.red_pieces if self.env.current_player == 'red' else self.env.blue_pieces) if p.get_position() is None]
        for priority_piece in ['军旗', '地雷', '炸弹']:
            for piece in available_pieces:
                if piece.get_name() == priority_piece:
                    return piece
        return random.choice(available_pieces)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, is_weights, indices = experiences

        # 获取下一个状态的 Q 值分布
        _, Q_targets_next_dist = self.qnetwork_target(next_states)
        Q_targets_next = (Q_targets_next_dist * self.support).sum(2)
        Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)

        # 计算 Q 目标值
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 获取当前状态的 Q 值分布
        _, Q_expected_dist = self.qnetwork_local(states)
        
        # 确保 actions 是一个二维张量
        actions = actions.unsqueeze(-1)
        
        # 获取当前状态下的 Q 值
        Q_expected_dist = Q_expected_dist.gather(1, actions.expand(actions.size(0), actions.size(1), Q_expected_dist.size(-1)))
        Q_expected = (Q_expected_dist * self.support).sum(2)

        # 计算损失
        loss = (is_weights * nn.MSELoss(reduction='none')(Q_expected, Q_targets)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, 0.001)

        # 更新优先级
        priorities = (loss + self.memory.eps).cpu().data.numpy().flatten()
        self.memory.update_priorities(indices, priorities)


    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        
class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, state_size, seed, alpha=0.6, beta_start=0.4, beta_frames=100000, eps=1e-6, n_step=3, gamma=0.99):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.state_size = state_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.n_step = n_step
        self.gamma = gamma
        self.frame = 1
        self.sum_tree = SumTree(buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(self.experience(state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        R, n_step_next_state, n_step_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        state, action = self.n_step_buffer[0][:2]
        experience = self.experience(state, action, R, n_step_next_state, n_step_done)
        self.memory.append(experience)

        priority = self.sum_tree.max() if self.sum_tree.max() > 0 else 1.0
        self.sum_tree.add(priority, experience)

    def _get_n_step_info(self, n_step_buffer, gamma):
        R = sum([n_step_buffer[i][2] * (gamma ** i) for i in range(len(n_step_buffer))])
        return R, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        experiences = []
        indices = []
        priorities = []
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        segment = self.sum_tree.total() / self.batch_size

        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.sum_tree.get(s)
            priorities.append(p)
            experiences.append(data)
            indices.append(idx)

        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self.sum_tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(is_weight).float().to(device)

        states = states.view(-1, self.state_size)
        next_states = next_states.view(-1, self.state_size)
        actions = actions.view(-1, 1)  # 确保 actions 是一个二维张量

        return (states, actions, rewards, next_states, dones, is_weights, indices)


    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.sum_tree.update(idx, priority)

    def __len__(self):
        return len(self.memory)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def max(self):
        return self.tree[self.capacity - 1:].max()

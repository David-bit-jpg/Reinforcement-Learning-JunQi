import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

class Memory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n,), dtype=object)
        ISWeights = np.empty((n, 1))

        priority_segment = self.tree.total_priority / n
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        if min_prob == 0:
            min_prob = 1 / self.tree.capacity

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_priority
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DoubleDQN(nn.Module):
    def __init__(self, board_rows, board_cols, piece_types):
        super(DoubleDQN, self).__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.piece_types = piece_types

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * board_rows * board_cols, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, board_rows * board_cols * len(piece_types))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * self.board_rows * self.board_cols)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.board_rows, self.board_cols, len(self.piece_types))
        return x
class DQNAgent:
    def __init__(self, model, target_model, memory, piece_types, board_rows, board_cols, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, update_target_every=5):
        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.piece_types = piece_types
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.step_count = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.model.fc3.out_features))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state).detach().numpy()[0]
        return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory.tree.data) < self.batch_size:
            return

        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)

        def get_board_state(x):
            if isinstance(x, dict) and 'board_state' in x:
                board_state = x['board_state']
                if board_state.ndim == 3 and board_state.shape[2] == len(self.piece_types):
                    return np.transpose(board_state, (2, 0, 1))
            return np.zeros((3, self.board_rows, self.board_cols))  # 使用3个通道

        states = torch.tensor(np.array([get_board_state(x[0]) for x in minibatch]), dtype=torch.float32)
        actions = torch.tensor([x[1] for x in minibatch], dtype=torch.int64)
        rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32)
        next_states = torch.tensor(np.array([get_board_state(x[3]) for x in minibatch]), dtype=torch.float32)
        dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32)
        ISWeights = torch.tensor(ISWeights, dtype=torch.float32)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_target_q_values = self.target_model(next_states)

        actions = actions.view(-1, 1)  # 调整 actions 的维度，使其与 q_values 匹配
        q_value = q_values.gather(1, actions).squeeze(1)
        next_q_value = next_target_q_values.gather(1, next_q_values.argmax(1).view(-1, 1)).squeeze(1)

        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value.detach()).pow(2) * ISWeights.squeeze()
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.batch_update(tree_idx, prios.data.cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())



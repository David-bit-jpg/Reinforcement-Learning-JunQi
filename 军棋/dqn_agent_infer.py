import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random



class DoubleDQN(nn.Module):
    def __init__(self, board_rows, board_cols, piece_types):
        super(DoubleDQN, self).__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.piece_types = piece_types

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # 初始化一个小的输入，计算卷积层的输出大小
        temp_input = torch.zeros(1, 3, board_rows, board_cols)
        temp_output = self._forward_conv(temp_input)
        conv_output_size = temp_output.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, len(piece_types))

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        
        # # 打印调试信息
        # print(f'After conv1: {x.shape}')
        # print(f'After conv2: {x.shape}')
        # print(f'After conv3: {x.shape}')
        
        # 动态调整全连接层的输入大小
        x = x.view(x.size(0), -1)
        expected_size = self.fc1.in_features
        actual_size = x.size(1)
        if actual_size < expected_size:
            padding = torch.zeros((x.size(0), expected_size - actual_size), device=x.device)
            x = torch.cat((x, padding), dim=1)
        elif actual_size > expected_size:
            x = x[:, :expected_size]
        
        # print(f'After view: {x.shape}')  # 添加调试信息
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class DQNAgent:
    def __init__(self, model, target_model, action_size, memory_capacity=10000, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.model = model
        self.target_model = target_model
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = Memory(memory_capacity)
        self.batch_size = 64
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))
                
    def act(self, state):
        board_state = state['board_state']
        move_history = state['move_history']
        battle_history = state['battle_history']
        agent_color = state['agent_color']

        opponent_color = 'blue' if agent_color == 'red' else 'red'

        # 提取对方棋子的位置
        opponent_positions = [(x, y) for x in range(board_state.shape[1]) for y in range(board_state.shape[2]) if np.any(board_state[0, x, y] != 0)]

        # 根据对方棋子的位置创建矩阵，推断对方棋子的可能类型
        inferred_pieces = {}
        for pos in opponent_positions:
            input_data = self.prepare_input(board_state[0], move_history, battle_history, pos)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            output = self.model(input_tensor).detach().numpy()
            piece_type_idx = np.argmax(output)

            # 确保 piece_type_idx 在有效范围内
            if piece_type_idx >= len(self.model.piece_types):
                piece_type_idx = len(self.model.piece_types) - 1

            inferred_pieces[pos] = self.model.piece_types[piece_type_idx]

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), inferred_pieces
        state_tensor = torch.tensor(board_state, dtype=torch.float32)
        
        # 确保 state_tensor 的形状是 [1, 3, board_rows, board_cols]
        if state_tensor.shape[1] != 3:
            if state_tensor.shape[1] > 3:
                state_tensor = state_tensor[:, :3, :, :]
            else:
                padding = (0, 0, 0, 0, 0, 3 - state_tensor.shape[1])
                state_tensor = F.pad(state_tensor, padding)

        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item(), inferred_pieces

    def prepare_input(self, board_state, move_history, battle_history, pos):
        x, y = pos
        input_data = np.zeros((3, self.model.board_rows, self.model.board_cols))
        
        if board_state.shape[2] > 0:
            max_channel = np.argmax(board_state[x, y])
            max_channel = min(max_channel, board_state.shape[2] - 1)  # 确保 max_channel 在有效范围内
            board_slice = board_state[:, :, max_channel]
            
            # 确保 board_slice 的形状与 input_data[0] 一致
            padded_slice = np.zeros((self.model.board_rows, self.model.board_cols))
            rows = min(board_slice.shape[0], self.model.board_rows)
            cols = min(board_slice.shape[1], self.model.board_cols)
            
            # 确保填充或修剪的形状正确
            for i in range(rows):
                for j in range(cols):
                    if isinstance(board_slice[i, j], (list, np.ndarray)):
                        padded_slice[i, j] = board_slice[i, j][0]  # 取序列中的第一个元素
                    else:
                        padded_slice[i, j] = board_slice[i, j]
            
            input_data[0] = padded_slice
        else:
            # 如果 board_state 的通道数不足，创建一个包含所有通道数据的统一表示
            temp_data = np.zeros((self.model.board_rows, self.model.board_cols))
            for c in range(board_state.shape[2]):
                temp_data += board_state[:, :, c]
            if board_state.shape[2] < 3:
                temp_data = np.pad(temp_data, ((0, 0), (0, 3 - board_state.shape[2])), mode='constant', constant_values=0)
            input_data[0] = temp_data

        for move in move_history:
            start_pos, _, end_pos = move
            if end_pos == pos:
                input_data[1, start_pos[0], start_pos[1]] += 1

        for battle in battle_history:
            attacker_pos, defender_pos, _ = battle
            if defender_pos == pos:
                input_data[2, attacker_pos[0], attacker_pos[1]] += 1

        return input_data

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
        states = np.array([self.get_board_state(x[0]) for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([self.get_board_state(x[3]) for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 确保动作索引在有效范围内
        actions = torch.clamp(actions, 0, self.model.fc3.out_features - 1)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = (q_values - expected_q_values.detach()).pow(2) * torch.tensor(ISWeights, dtype=torch.float32)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        abs_errors = torch.abs(q_values - expected_q_values).detach().numpy()
        self.memory.batch_update(tree_idx, abs_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
                
    def get_board_state(self, state_dict):
        board_state = state_dict['board_state']
        
        # 如果 board_state 是四维的 (batch_size, channels, rows, cols)，我们去掉 batch_size 维度
        if len(board_state.shape) == 4 and board_state.shape[0] == 1:
            board_state = board_state[0]
        
        # 确保 board_state 是一个三维数组
        if len(board_state.shape) != 3:
            raise ValueError(f"Expected board_state to be a 3D array, but got shape: {board_state.shape}")
        
        transposed_state = np.transpose(board_state, (2, 0, 1))
        
        # 强制调整通道数为3
        if transposed_state.shape[0] > 3:
            transposed_state = transposed_state[:3, :, :]
        elif transposed_state.shape[0] < 3:
            padded_state = np.zeros((3, transposed_state.shape[1], transposed_state.shape[2]))
            padded_state[:transposed_state.shape[0], :, :] = transposed_state
            transposed_state = padded_state
            
        return transposed_state


import numpy as np
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def add(self, priority, data):
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, s):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if s <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    s -= self.tree[left_child_index]
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
        self.capacity = capacity

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
            min_prob = 1 / self.capacity

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
            
    def __len__(self):
        return self.tree.n_entries

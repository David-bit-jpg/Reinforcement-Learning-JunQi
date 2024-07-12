import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from junqi_env_infer import JunQiEnvInfer
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DoubleDQN(nn.Module):
    def __init__(self, board_rows, board_cols, piece_types):
        super(DoubleDQN, self).__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.piece_types = piece_types

        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1)
        self.res_block1 = ResidualBlock(128, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)

        temp_input = torch.zeros(1, 4, board_rows, board_cols)
        temp_output = self._forward_conv(temp_input)
        conv_output_size = temp_output.reshape(1, -1).size(1)

        self.fc1 = nn.Linear(conv_output_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, len(piece_types))

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)

        x = x.reshape(x.size(0), -1)
        expected_size = self.fc1.in_features
        actual_size = x.size(1)
        if actual_size < expected_size:
            padding = torch.zeros((x.size(0), expected_size - actual_size), device=x.device)
            x = torch.cat((x, padding), dim=1)
        elif actual_size > expected_size:
            x = x[:, :expected_size]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class DQNAgent:
    def __init__(self, model, target_model, action_size, initial_board, memory_capacity=10000, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
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
        self.pre_fill_memory(initial_board)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def pre_fill_memory(self, initial_board):
        env = JunQiEnvInfer(initial_board=initial_board)
        state = env.reset()
        for _ in range(self.memory.capacity // 10):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            self.remember(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()
            
    def act(self, state):
        board_state = state['board_state']
        move_history = state['move_history']
        battle_history = state['battle_history']
        agent_color = state['agent_color']
        opponent_commander_dead = state['opponent_commander_dead']

        opponent_color = 'blue' if agent_color == 'red' else 'red'

        # 提取对方棋子的位置
        opponent_positions = []
        for x in range(self.model.board_rows):
            for y in range(self.model.board_cols):
                if np.any(board_state[x, y, :len(self.model.piece_types)] != 0):
                    opponent_positions.append((x, y))

        # 根据对方棋子的位置创建矩阵，推断对方棋子的可能类型
        inferred_pieces = {}
        for pos in opponent_positions:
            input_data = self.prepare_input(board_state, move_history, battle_history, opponent_commander_dead, pos)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            output = self.model(input_tensor).detach().numpy()
            piece_type_idx = np.argmax(output)

            # 确保 piece_type_idx 在有效范围内
            if piece_type_idx >= len(self.model.piece_types):
                piece_type_idx = len(self.model.piece_types) - 1

            inferred_pieces[pos] = self.model.piece_types[piece_type_idx]

        if opponent_commander_dead:
            # 明确军旗的位置
            if opponent_color == 'red':
                flag_positions = [(12, 1), (12, 3)]
            else:
                flag_positions = [(0, 1), (0, 3)]
            for flag_pos in flag_positions:
                inferred_pieces[flag_pos] = '军旗'

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), inferred_pieces

        state_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0)

        # 动态调整输入形状以匹配模型预期形状
        if state_tensor.shape[1] != 4:
            # 计算新的形状
            num_channels = state_tensor.shape[1]
            if num_channels > 4:
                state_tensor = state_tensor[:, :4, :, :]  # 如果通道数大于4，则截取前4个通道
            else:
                # 如果通道数少于4，则填充0使其达到4个通道
                padding = torch.zeros((state_tensor.shape[0], 4 - num_channels, state_tensor.shape[2], state_tensor.shape[3]), device=state_tensor.device)
                state_tensor = torch.cat((state_tensor, padding), dim=1)

        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item(), inferred_pieces
    
    def prepare_input(self, board_state, move_history, battle_history, opponent_commander_dead, pos):
        x, y = pos
        input_data = np.zeros((4, self.model.board_rows, self.model.board_cols))

        input_data[0] = board_state[:, :, :len(self.model.piece_types)].sum(axis=2)

        # 记录自己的移动历史
        for move in move_history:
            start_pos, _, end_pos = move
            if end_pos == pos:
                input_data[1, start_pos[0], start_pos[1]] += 1

        # 更新战斗历史以排除不可能的棋子类型
        for battle in battle_history:
            attacker_piece, attacker_pos, defender_piece, defender_pos, result = battle
            if defender_pos == pos:
                if result == 'win':
                    input_data[2, attacker_pos[0], attacker_pos[1]] += 1
                elif result == 'lose':
                    input_data[2, attacker_pos[0], attacker_pos[1]] -= 1

        # 检查是否有可能的炸弹或地雷
        if self.is_possible_bomb(battle_history, pos):
            input_data[3, x, y] = 1  # 标记为可能的炸弹
        elif self.is_possible_mine(move_history, pos):
            input_data[3, x, y] = 2  # 标记为可能的地雷

        if opponent_commander_dead:
            input_data[3, :, :] = 1

        return input_data

    def is_possible_bomb(self, battle_history, pos):
        # 如果一个棋子周围有很多战斗历史记录，那么这个棋子可能是炸弹
        battle_count = sum(1 for battle in battle_history if battle[3] == pos or battle[1] == pos)
        return battle_count > 3  # 可根据需要调整阈值

    def is_possible_mine(self, move_history, pos):
        # 如果一个棋子位于后两排且没有移动历史记录，那么这个棋子可能是地雷
        if pos[0] >= self.model.board_rows - 2:
            move_count = sum(1 for move in move_history if move[2] == pos)
            return move_count == 0
        return False

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
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

        return loss.item()  # 确保返回损失值


    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def get_board_state(self, state_dict):
        board_state = state_dict['board_state']
        
        if len(board_state.shape) == 4 and board_state.shape[0] == 1:
            board_state = board_state[0]
        
        if len(board_state.shape) != 3:
            raise ValueError(f"Expected board_state to be a 3D array, but got shape: {board_state.shape}")
        
        transposed_state = np.transpose(board_state, (2, 0, 1))
        
        if transposed_state.shape[0] > 4:
            transposed_state = transposed_state[:4, :, :]
        elif transposed_state.shape[0] < 4:
            padded_state = np.zeros((4, transposed_state.shape[1], transposed_state.shape[2]))
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

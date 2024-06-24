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

class InferenceNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InferenceNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
class DarkChessInference:
    def __init__(self, env, input_size, hidden_size, output_size, lr=0.001):
        self.env = env
        self.network = InferenceNetwork(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def infer(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            infer_output = self.network(state_tensor).cpu().numpy().flatten()
        self.update_infer_dict(infer_output)
        return infer_output

    def update_infer_dict(self, infer_output):
        """
        更新推测字典，根据推测网络的输出更新推测信息。
        infer_output: 神经网络的输出，表示对每个棋盘位置的推测得分。
        """
        # 初始化推测字典
        red_infer = {piece: set() for piece in self.env.blue_pieces}
        blue_infer = {piece: set() for piece in self.env.red_pieces}

        # 将推测输出转换为棋盘位置和可能的棋子类型
        board_size = self.env.board_rows * self.env.board_cols
        for i, piece in enumerate(self.env.blue_pieces):
            piece_scores = infer_output[i * board_size: (i + 1) * board_size]
            top_positions = np.argsort(piece_scores)[-3:]  # 获取推测得分最高的三个位置
            for pos in top_positions:
                row, col = divmod(pos, self.env.board_cols)
                red_infer[piece].add((row, col))

        for i, piece in enumerate(self.env.red_pieces):
            piece_scores = infer_output[(len(self.env.blue_pieces) + i) * board_size: (len(self.env.blue_pieces) + i + 1) * board_size]
            top_positions = np.argsort(piece_scores)[-3:]  # 获取推测得分最高的三个位置
            for pos in top_positions:
                row, col = divmod(pos, self.env.board_cols)
                blue_infer[piece].add((row, col))

        # 更新环境中的推测字典
        self.env.red_infer = red_infer
        self.env.blue_infer = blue_infer

    def update_model(self, state, target):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).to(device)
        self.optimizer.zero_grad()
        output = self.network(state_tensor)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()

    def mark_possible_identities(self):
        self.mark_red_possible_identities()
        self.mark_blue_possible_identities()

    def mark_red_possible_identities(self):
        for piece, killer in self.env.blue_died.items():
            if killer.get_position() is not None:
                self.mark_identity(killer, piece, 'red')

    def mark_blue_possible_identities(self):
        for piece, killer in self.env.red_died.items():
            if killer.get_position() is not None:
                self.mark_identity(killer, piece, 'blue')

    def mark_identity(self, killer, victim, color):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
        killer_rank = rank_order[killer.get_name()]
        victim_rank = rank_order[victim.get_name()]

        if color == 'red':
            infer_dict = self.env.red_infer
        else:
            infer_dict = self.env.blue_infer

        if killer_rank > victim_rank:
            possible_ids = [name for name, rank in rank_order.items() if rank > victim_rank]
            infer_dict[killer].update(possible_ids)
        elif killer_rank == 11 or victim_rank == 11:
            infer_dict[killer].update(['炸弹'])

        if killer.get_name() == '工兵' and self.is_railway_turn(killer):
            infer_dict[killer].add('工兵')

    def is_railway_turn(self, piece):
        current_position = piece.get_position()
        if current_position is None:
            return False
        railway_positions = self.env.railways.get(current_position, [])
        for pos in railway_positions:
            if pos in self.env.occupied_positions_red or pos in self.env.occupied_positions_blue:
                continue
            if self.is_valid_turn(piece, pos):
                return True
        return False

    def is_valid_turn(self, piece, target_position):
        current_position = piece.get_position()
        return (current_position[0] != target_position[0] and current_position[1] != target_position[1])

    def position_based_inference(self):
        self.position_based_inference_for_red()
        self.position_based_inference_for_blue()

    def position_based_inference_for_red(self):
        for piece in self.env.red_pieces:
            if piece.get_position() is not None:
                neighbors = self.get_neighbors(piece.get_position())
                for neighbor in neighbors:
                    neighbor_piece = self.env.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_color() == 'blue':
                        self.update_position_based_inference(neighbor_piece, piece, 'red')

    def position_based_inference_for_blue(self):
        for piece in self.env.blue_pieces:
            if piece.get_position() is not None:
                neighbors = self.get_neighbors(piece.get_position())
                for neighbor in neighbors:
                    neighbor_piece = self.env.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_color() == 'red':
                        self.update_position_based_inference(neighbor_piece, piece, 'blue')

    def get_neighbors(self, position):
        row, col = position
        neighbors = [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
            (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)
        ]
        return [(r, c) for r, c in neighbors if 0 <= r < self.env.board_rows and 0 <= c < self.env.board_cols]

    def update_position_based_inference(self, neighbor_piece, piece, color):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
        piece_rank = rank_order[piece.get_name()]
        neighbor_piece_rank = rank_order[neighbor_piece.get_name()]

        if piece_rank > 7:  # piece is a large piece
            if color == 'red':
                self.env.red_infer[neighbor_piece].update(['炸弹'])
            else:
                self.env.blue_infer[neighbor_piece].update(['炸弹'])
        elif piece_rank <= 3:  # piece is a small piece
            possible_ids = [name for name, rank in rank_order.items() if rank > piece_rank]
            if color == 'red':
                self.env.red_infer[neighbor_piece].update(possible_ids)
            else:
                self.env.blue_infer[neighbor_piece].update(possible_ids)

    def remove_impossible_identities(self):
        self.remove_impossible_identities_for_red()
        self.remove_impossible_identities_for_blue()

    def remove_impossible_identities_for_red(self):
        for piece in self.env.red_pieces:
            if piece in self.env.red_died:
                victim = self.env.red_died[piece]
                if '师长' in self.env.blue_infer[victim]:
                    self.env.blue_infer[victim].remove('师长')

    def remove_impossible_identities_for_blue(self):
        for piece in self.env.blue_pieces:
            if piece in self.env.blue_died:
                victim = self.env.blue_died[piece]
                if '师长' in self.env.red_infer[victim]:
                    self.env.red_infer[victim].remove('师长')

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
        
        self.fc1 = nn.Linear(state_size, 128)
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
    def __init__(self, state_size, action_size, env, seed, gamma=0.99, lr=0.001, buffer_size=50000, batch_size=64, update_every=4, tau=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size + 1  # +1 用于棋子特征
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

        self.qnetwork_local = DuelingQNetwork(self.state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(self.state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0

        # Initialize DarkChessInference with neural network
        input_size = state_size + 1  # +1 for piece feature
        hidden_size = 128
        output_size = env.board_rows * env.board_cols * (len(env.red_pieces) + len(env.blue_pieces))
        self.inference_module = DarkChessInference(env, input_size, hidden_size, output_size, lr)

    def act(self, state, player_color, own_pieces, opponent_pieces, own_killed_dict, opponent_killed_dict):
        # 步骤1：调用推测模块，更新推测信息
        infer_output = self.inference(state, player_color, own_pieces, opponent_pieces)
        
        # 步骤2：选择棋子
        piece = self.select_piece(infer_output, player_color, own_pieces)
        if piece is None:
            return None

        piece_feature = piece_encoding[piece.get_name()]
        state_with_piece = np.append(infer_output, piece_feature)

        state_tensor = torch.from_numpy(state_with_piece).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor).cpu().numpy().flatten()
        self.qnetwork_local.train()

        valid_moves = self.env.get_valid_moves(piece)
        if len(valid_moves) == 0:
            return None

        valid_action_values = [action_values[self.env.encode_action(piece, pos)] for pos in valid_moves]

        # 步骤3：选择动作
        if np.random.rand() > self.epsilon:
            target_position = valid_moves[np.argmax(valid_action_values)]
        else:
            target_position = random.choice(valid_moves)

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        action = self.env.encode_action(piece, target_position)
        return action

    def inference(self, state, player_color, own_pieces, opponent_pieces):
        # 将当前状态输入推测网络，并更新推测字典
        infer_output = self.inference_module.infer(state)
        return infer_output

    def select_piece(self, infer_output, player_color, own_pieces):
        available_pieces = [p for p in own_pieces if p.get_position() is not None]
        if len(available_pieces) == 0:
            return None

        piece_scores = []
        board_size = self.env.board_rows * self.env.board_cols
        for piece in available_pieces:
            if piece.get_position() is not None:
                pos = piece.get_position()[0] * self.env.board_cols + piece.get_position()[1]
                piece_scores.append((piece, infer_output[pos]))

        piece_scores.sort(key=lambda x: x[1], reverse=True)
        return piece_scores[0][0] if len(piece_scores) > 0 else None

    def step(self, state, action, reward, next_state, done):
        # 步骤4：积累经验
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

        # 步骤5：根据奖励更新推测模型
        infer_output = self.inference(state, self.env.current_player, self.env.red_pieces, self.env.blue_pieces)
        
        target = np.zeros_like(infer_output)
        board_size = self.env.board_rows * self.env.board_cols

        for piece in self.env.red_pieces + self.env.blue_pieces:
            if piece.get_position() is not None:
                pos = piece.get_position()[0] * self.env.board_cols + piece.get_position()[1]
                if piece in self.env.red_killed.values() or piece in self.env.blue_killed.values():
                    target[pos] = 1.0  # 奖励杀死的棋子
                elif piece in self.env.red_died.keys() or piece in self.env.blue_died.keys():
                    target[pos] = 0.0  # 惩罚死亡的棋子

        self.inference_module.update_model(state.flatten(), target)


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
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
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

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

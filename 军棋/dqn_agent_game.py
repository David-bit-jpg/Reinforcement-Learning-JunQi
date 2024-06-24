import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from pieces import Piece
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
        # 先使用固定规则进行初步推测
        initial_infer = self.initial_inference(state)
        
        # 使用神经网络对初步推测结果进行调整和改进
        adjusted_infer = self.adjust_inference(state, initial_infer)
        
        return adjusted_infer

    def initial_inference(self, state):
        """
        使用固定规则进行初步推测。
        """
        initial_infer = {piece: set() for piece in self.env.blue_pieces}
        
        self.mark_possible_identities(initial_infer)
        self.position_based_inference(initial_infer)
        self.remove_impossible_identities(initial_infer)

        return initial_infer

    def adjust_inference(self, state, initial_infer):
        """
        使用神经网络对初步推测结果进行调整和改进。
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        initial_infer_tensor = self.infer_dict_to_tensor(initial_infer).float().unsqueeze(0).to(device)

        with torch.no_grad():
            infer_output = self.network(torch.cat((state_tensor, initial_infer_tensor), dim=1)).cpu().numpy().flatten()
        
        return self.update_infer_dict(infer_output)

    def infer_dict_to_tensor(self, infer_dict):
        """
        将推测字典转换为张量。
        """
        infer_tensor = np.zeros((self.env.board_rows, self.env.board_cols, 1))
        for piece, positions in infer_dict.items():
            for pos in positions:
                x, y = pos
                infer_tensor[x, y, 0] = 1
        return infer_tensor

    def update_infer_dict(self, infer_output):
        """
        根据神经网络的输出更新推测字典。
        """
        infer_dict = {}
        board_size = self.env.board_rows * self.env.board_cols
        for i, piece in enumerate(self.env.blue_pieces):
            piece_scores = infer_output[i * board_size: (i + 1) * board_size]
            top_position = np.argmax(piece_scores)  # 获取推测得分最高的位置
            row, col = divmod(top_position, self.env.board_cols)
            infer_dict[piece] = (row, col)

        for i, piece in enumerate(self.env.red_pieces):
            piece_scores = infer_output[(len(self.env.blue_pieces) + i) * board_size: (len(self.env.blue_pieces) + i + 1) * board_size]
            top_position = np.argmax(piece_scores)  # 获取推测得分最高的位置
            row, col = divmod(top_position, self.env.board_cols)
            infer_dict[piece] = (row, col)

        return infer_dict

    def mark_possible_identities(self, infer_dict):
        """
        根据环境中的击杀记录，标记可能的棋子身份。
        """
        for piece, killer in self.env.blue_died.items():
            if killer.get_position() is not None:
                self.mark_identity(infer_dict, killer, piece, 'red')

        for piece, killer in self.env.red_died.items():
            if killer.get_position() is not None:
                self.mark_identity(infer_dict, killer, piece, 'blue')

    def mark_identity(self, infer_dict, killer, victim, color):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
        killer_rank = rank_order[killer.get_name()]
        victim_rank = rank_order[victim.get_name()]

        if color == 'red':
            infer_set = infer_dict[killer]
        else:
            infer_set = infer_dict[killer]

        if killer_rank > victim_rank:
            possible_ids = [name for name, rank in rank_order.items() if rank > victim_rank]
            infer_set.update(possible_ids)
        elif killer_rank == 11 or victim_rank == 11:
            infer_set.update(['炸弹'])

        if killer.get_name() == '工兵' and self.is_railway_turn(killer):
            infer_set.add('工兵')

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

    def position_based_inference(self, infer_dict):
        """
        基于棋子的位置关系进行推测调整。
        """
        for piece in self.env.red_pieces:
            if piece.get_position() is not None:
                neighbors = self.get_neighbors(piece.get_position())
                for neighbor in neighbors:
                    neighbor_piece = self.env.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_color() == 'blue':
                        self.update_position_based_inference(infer_dict, neighbor_piece, piece, 'red')

        for piece in self.env.blue_pieces:
            if piece.get_position() is not None:
                neighbors = self.get_neighbors(piece.get_position())
                for neighbor in neighbors:
                    neighbor_piece = self.env.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_color() == 'red':
                        self.update_position_based_inference(infer_dict, neighbor_piece, piece, 'blue')

    def get_neighbors(self, position):
        row, col = position
        neighbors = [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
            (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)
        ]
        return [(r, c) for r, c in neighbors if 0 <= r < self.env.board_rows and 0 <= c < self.env.board_cols]

    def update_position_based_inference(self, infer_dict, neighbor_piece, piece, color):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
        piece_rank = rank_order[piece.get_name()]
        neighbor_piece_rank = rank_order[neighbor_piece.get_name()]

        if piece_rank > 7:  # piece is a large piece
            if color == 'red':
                infer_dict[neighbor_piece].update(['炸弹'])
            else:
                infer_dict[neighbor_piece].update(['炸弹'])
        elif piece_rank <= 3:  # piece is a small piece
            possible_ids = [name for name, rank in rank_order.items() if rank > piece_rank]
            if color == 'red':
                infer_dict[neighbor_piece].update(possible_ids)
            else:
                infer_dict[neighbor_piece].update(possible_ids)

    def remove_impossible_identities(self, infer_dict):
        """
        移除不可能的身份推测。
        """
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }

        # 对红方进行处理
        for piece in self.env.red_pieces:
            if piece in self.env.red_died:
                killer = self.env.red_died[piece]
                victim_rank = rank_order[piece.get_name()]
                for possible_piece, positions in infer_dict.items():
                    if possible_piece.get_color() == 'blue' and killer.get_name() == possible_piece.get_name():
                        possible_ids = [name for name, rank in rank_order.items() if rank <= victim_rank]
                        for pos in positions:
                            if pos in infer_dict[possible_piece]:
                                infer_dict[possible_piece].remove(pos)

        # 对蓝方进行处理
        for piece in self.env.blue_pieces:
            if piece in self.env.blue_died:
                killer = self.env.blue_died[piece]
                victim_rank = rank_order[piece.get_name()]
                for possible_piece, positions in infer_dict.items():
                    if possible_piece.get_color() == 'red' and killer.get_name() == possible_piece.get_name():
                        possible_ids = [name for name, rank in rank_order.items() if rank <= victim_rank]
                        for pos in positions:
                            if pos in infer_dict[possible_piece]:
                                infer_dict[possible_piece].remove(pos)


    def update_model(self, state, target):
        """
        根据奖励更新神经网络模型的权重。
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).to(device)
        output = self.network(state_tensor)
        loss = self.criterion(output, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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

        # 定义棋子的固定价值
        self.piece_values = {
            '排长': 1,
            '连长': 2,
            '营长': 3,
            '团长': 4,
            '旅长': 5,
            '师长': 6,
            '军长': 7,
            '司令': 8,
            '军旗': 9,   # 军旗和地雷价值设为最低
            '地雷': 9,  
            '炸弹': 0    # 炸弹价值设为0
        }

        # 定义棋位的增加价值
        self.position_values = {
            'railway': 10,
            'road': 5,
            'camp': 15,
            'enemy_camp': 20,
            'base': 10
        }

    def get_piece_fixed_value(self, piece):
        return self.piece_values.get(piece.get_name(), 0)

    def get_position_value(self, position):
        if self.env.is_railway(position):
            return self.position_values['railway']
        elif self.env.is_road(position):
            return self.position_values['road']
        elif self.env.is_camp(position):
            return self.position_values['camp']
        elif self.env.is_enemy_camp(position):
            return self.position_values['enemy_camp']
        elif self.env.is_base(position):
            return self.position_values['base']
        else:
            return 0

    def get_neighbors(self, position):
        row, col = position
        neighbors = [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
            (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)
        ]
        return [(r, c) for r, c in neighbors if 0 <= r < self.env.board_rows and 0 <= c < self.env.board_cols]

    def calculate_neighbor_influence(self, piece):
        neighbors = self.get_neighbors(piece.get_position())
        own_color = piece.get_color()
        own_piece_value = self.get_piece_fixed_value(piece)
        
        positive_influence = 0
        negative_influence = 0
        
        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece:
                neighbor_piece_value = self.get_piece_fixed_value(neighbor_piece)
                if neighbor_piece.get_color() == own_color:
                    if neighbor_piece_value > own_piece_value:
                        positive_influence += neighbor_piece_value
                else:
                    if neighbor_piece_value > own_piece_value:
                        negative_influence -= neighbor_piece_value
        
        return positive_influence + negative_influence

    def calculate_protection_value(self, piece):
        if piece.get_color() == 'red':
            flag_position = self.env.get_red_flag_position()
        else:
            flag_position = self.env.get_blue_flag_position()

        piece_position = piece.get_position()
        distance_to_flag = abs(piece_position[0] - flag_position[0]) + abs(piece_position[1] - flag_position[1])

        if distance_to_flag == 0:
            return float('inf')

        protection_value = 15 / distance_to_flag
        return protection_value

    def calculate_attack_value(self, piece):
        if piece.get_color() == 'red':
            flag_position = self.env.get_blue_flag_position()
        else:
            flag_position = self.env.get_red_flag_position()

        piece_position = piece.get_position()
        distance_to_flag = abs(piece_position[0] - flag_position[0]) + abs(piece_position[1] - flag_position[1])

        if distance_to_flag == 0:
            return 1000

        attack_value = 15 / distance_to_flag
        return attack_value

    def select_piece(self, state, player_color, own_pieces):
        available_pieces = [p for p in own_pieces if p.get_position() is not None]
        if len(available_pieces) == 0:
            return None

        piece_scores = []
        for piece in available_pieces:
            if piece.get_position() is not None:
                x, y = piece.get_position()
                fixed_value = self.get_piece_fixed_value(piece)
                position_value = self.get_position_value((x, y))
                neighbor_influence = self.calculate_neighbor_influence(piece)
                protection_value = self.calculate_protection_value(piece)
                attack_value = self.calculate_attack_value(piece)
                total_value = fixed_value + position_value + neighbor_influence + protection_value + attack_value
                piece_scores.append((piece, total_value))

        piece_scores.sort(key=lambda x: x[1], reverse=True)
        return piece_scores[0][0] if len(piece_scores) > 0 else None

    def build_game_tree(self, state, depth, is_maximizing_player):
        if depth == 0 or self.env.is_terminal_state(state):
            return self.evaluate_state(state)

        valid_actions = self.env.get_valid_actions(state)
        if is_maximizing_player:
            max_eval = float('-inf')
            for action in valid_actions:
                next_state = self.env.simulate_action(state, action)
                eval = self.build_game_tree(next_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                next_state = self.env.simulate_action(state, action)
                eval = self.build_game_tree(next_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def evaluate_state(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            value = self.qnetwork_local(state_tensor).cpu().numpy().flatten()[0]
        self.qnetwork_local.train()
        return value

    def act(self, state, player_color, own_pieces, opponent_pieces, own_killed_dict, opponent_killed_dict):
        infer_dict = self.env.inference_module.infer(state)
        opponent_infer_pieces = [Piece(name, pos) for name, pos in infer_dict.items()]

        piece = self.select_piece(state, player_color, own_pieces)
        if piece is None:
            return None

        piece_feature = piece_encoding[piece.get_name()]
        state_with_piece = np.append(state.flatten(), piece_feature)

        best_action = None
        best_value = float('-inf')
        valid_moves = self.env.get_valid_moves(piece)

        for move in valid_moves:
            next_state = self.env.simulate_action(state_with_piece, (piece, move))
            move_value = self.build_game_tree(next_state, depth=3, is_maximizing_player=False)
            if move_value > best_value:
                best_value = move_value
                best_action = self.env.encode_action(piece, move)

        return best_action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        infer_dict = self.env.inference_module.infer(state)
        for piece, pos in infer_dict.items():
            target_position = (pos[0], pos[1])
            inference_reward = self.env.reward_model._inference_accuracy_reward(piece, target_position)
            self.env.inference_module.update_model(state, inference_reward)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
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

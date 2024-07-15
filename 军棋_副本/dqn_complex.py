import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
class PolicyValueNet(nn.Module):
    def __init__(self, board_size=21, num_features=7, hidden_size=128):
        super(PolicyValueNet, self).__init__()
        self.board_conv = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.board_bn = nn.BatchNorm2d(32)
        
        self.prob_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.prob_bn = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(board_size * board_size * 32 + num_features, hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        
        self.policy_head = nn.Linear(hidden_size, board_size * board_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, board_state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        board_state = F.relu(self.board_bn(self.board_conv(board_state)))
        board_state = board_state.view(board_state.size(0), -1)
        
        prob_matrix = F.relu(self.prob_bn(self.prob_conv(prob_matrix)))
        prob_matrix = prob_matrix.view(prob_matrix.size(0), -1)
        
        combined = torch.cat([board_state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features], dim=1)
        
        x = F.relu(self.fc1_bn(self.fc1(combined)))
        
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        
        return policy, value

class InferenceNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(InferenceNetwork, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)  # 初始化时先设为1，稍后调整
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, state_tensor, infer_tensor):
        x = torch.cat((state_tensor, infer_tensor), dim=1)
        input_size = x.size(1)  # 动态获取输入大小

        # 动态调整fc1层的输入大小
        if self.fc1.in_features != input_size:
            self.fc1 = nn.Linear(input_size, self.fc1.out_features).to(x.device)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DarkChessInference:
    def __init__(self, env, hidden_size, output_size, lr=0.001):
        self.env = env
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 初始化网络时不需要输入大小
        self.network = InferenceNetwork(hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def infer(self, state):
        initial_infer = self.initial_inference(state)
        adjusted_infer = self.adjust_inference(state, initial_infer)
        return adjusted_infer

    def initial_inference(self, state):
        initial_infer = {piece: set(piece_encoding.keys()) for piece in self.env.blue_pieces}
        self.mark_possible_identities(initial_infer)
        self.position_based_inference(initial_infer)
        self.remove_impossible_identities(initial_infer)
        return initial_infer

    def adjust_inference(self, state, initial_infer):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).view(1, -1).to(device)
        initial_infer_tensor = torch.from_numpy(self.infer_dict_to_tensor(initial_infer)).float().unsqueeze(0).view(1, -1).to(device)

        adjusted_infer_tensor = self.network(state_tensor, initial_infer_tensor)

        batch_size = state_tensor.size(0)
        board_rows = self.env.board_rows
        board_cols = self.env.board_cols
        piece_encoding_len = len(piece_encoding)

        inferred_shape = (batch_size, board_rows, board_cols, piece_encoding_len)
        print(f"Inferred shape: {inferred_shape}")
        adjusted_infer_tensor = adjusted_infer_tensor.view(inferred_shape)
        
        adjusted_infer = self.tensor_to_infer_dict(adjusted_infer_tensor.squeeze(0).detach().cpu().numpy())
        return adjusted_infer

    def infer_dict_to_tensor(self, infer_dict):
        infer_tensor = np.zeros((self.env.board_rows, self.env.board_cols, len(piece_encoding)))
        for piece, positions in infer_dict.items():
            if isinstance(piece, Piece):
                piece_name = piece.get_name()
                for pos in positions:
                    if isinstance(pos, tuple) and len(pos) == 2:
                        x, y = pos
                        infer_tensor[x, y, piece_encoding[piece_name]] = 1
            elif piece == 'large':
                for pos in positions:
                    if isinstance(pos, tuple) and len(pos) == 2:
                        x, y = pos
                        piece_name = random.choice(['旅长', '师长', '军长', '司令', '炸弹'])
                        infer_tensor[x, y, piece_encoding[piece_name]] = 1
            elif piece == 'small':
                for pos in positions:
                    if isinstance(pos, tuple) and len(pos) == 2:
                        x, y = pos
                        piece_name = random.choice(['军旗', '地雷', '工兵', '排长', '连长', '营长', '团长'])
                        infer_tensor[x, y, piece_encoding[piece_name]] = 1
        return infer_tensor

    def tensor_to_infer_dict(self, tensor):
        infer_dict = {}
        for piece in self.env.blue_pieces:
            piece_name = piece.get_name()
            infer_dict[piece] = set()
            for x in range(tensor.shape[0]):
                for y in range(tensor.shape[1]):
                    if tensor[x, y, piece_encoding[piece_name]] > 0.5:
                        infer_dict[piece].add((x, y))
        return infer_dict

    def mark_possible_identities(self, infer_dict):
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

        if piece_rank > 7:
            infer_dict[neighbor_piece].update(['炸弹'])
        elif piece_rank <= 3:
            possible_ids = [name for name, rank in rank_order.items() if rank > piece_rank]
            infer_dict[neighbor_piece].update(possible_ids)

    def remove_impossible_identities(self, infer_dict):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
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
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).to(device)
        output = self.network(state_tensor)
        loss = self.criterion(output, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class MCTS:
    def __init__(self, policy_value_net, c_puct=1.4, n_playout=1600):
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct
        self.n_playout = n_playout
        self._root = None

    class Node:
        def __init__(self, parent, prior_p):
            self._parent = parent
            self._children = {}
            self._n_visits = 0
            self._Q = 0
            self._u = 0
            self._P = prior_p

        def expand(self, action_priors):
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = MCTS.Node(self, prob)

        def select(self, c_puct):
            return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

        def update(self, leaf_value):
            self._n_visits += 1
            self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits
            if self._parent:
                self._parent.update(-leaf_value)

        def get_value(self, c_puct):
            self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            return self._Q + self._u

        def is_leaf(self):
            return self._children == {}

        def is_root(self):
            return self._parent is None

    def _playout(self, state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features = self.env.simulate_action(state, action)
        action_probs, leaf_value = self.policy_value_net(state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features)
        node.expand(action_probs)
        node.update(-leaf_value)

    def get_move_probs(self, state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        for _ in range(self.n_playout):
            self._playout(state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features)
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = visits / np.sum(visits)
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = None

    def __call__(self, state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        self._root = self.Node(None, 1.0)
        acts, probs = self.get_move_probs(state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features)
        return acts, probs
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_value = nn.Linear(128, 1)
        self.fc3_advantage = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3_value(x)
        advantage = self.fc3_advantage(x)
        return value + advantage - advantage.mean()
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = DuelingQNetwork(self.state_size, action_size, seed).to(self.device)
        self.qnetwork_target = DuelingQNetwork(self.state_size, action_size, seed).to(self.device)
        self.policy_value_net = PolicyValueNet(board_size=21, num_features=7, hidden_size=128).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.policy_value_optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0

        self.mcts = MCTS(self.policy_value_net)

    def get_prob_matrix(self, state, inferred_state):
        prob_matrix = np.zeros((self.env.board_rows, self.env.board_cols))
        for piece in self.env.red_pieces:
            if piece.position:
                x, y = piece.position
                prob_matrix[x, y] = self.env.piece_name_to_value(piece.get_name())
        for piece, positions in inferred_state.items():
            for pos in positions:
                x, y = pos
                prob_matrix[x, y] = self.env.piece_name_to_value(piece.get_name())
        return prob_matrix

    def act(self, state, turn, num_own_pieces, num_opponent_pieces, features):
        inferred_state = self.env.inference_module.infer(state)
        prob_matrix = self.get_prob_matrix(state, inferred_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prob_matrix_tensor = torch.FloatTensor(prob_matrix).unsqueeze(0).to(self.device)
        turn_tensor = torch.FloatTensor([turn]).unsqueeze(0).to(self.device)
        num_own_pieces_tensor = torch.FloatTensor([num_own_pieces]).unsqueeze(0).to(self.device)
        num_opponent_pieces_tensor = torch.FloatTensor([num_opponent_pieces]).unsqueeze(0).to(self.device)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        self.policy_value_net.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
            policy, value_estimation = self.policy_value_net(state_tensor, prob_matrix_tensor, turn_tensor, num_own_pieces_tensor, num_opponent_pieces_tensor, features_tensor)
        self.qnetwork_local.train()
        self.policy_value_net.train()

        # Combine action values, policy probabilities, and value estimation
        action_values = action_values.cpu().numpy().flatten()
        policy = policy.cpu().numpy().flatten()
        value_estimation = value_estimation.item()  # Extract the scalar value
        combined_values = action_values + policy + value_estimation

        # Use MCTS to select the best action based on combined values
        acts, probs = self.mcts(state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features)

        # Use combined values to adjust probabilities
        act_probs = np.zeros_like(probs)
        for i, act in enumerate(acts):
            act_probs[i] = combined_values[act]
        act_probs = act_probs / act_probs.sum()

        if random.random() > self.epsilon:
            return np.random.choice(acts, p=act_probs)
        else:
            return random.choice(acts)

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

    def update_policy_value_net(self, state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features, target):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prob_matrix_tensor = torch.FloatTensor(prob_matrix).unsqueeze(0).to(self.device)
        turn_tensor = torch.FloatTensor([turn]).unsqueeze(0).to(self.device)
        num_own_pieces_tensor = torch.FloatFloat([num_own_pieces]).unsqueeze(0).to(self.device)
        num_opponent_pieces_tensor = torch.FloatFloat([num_opponent_pieces]).unsqueeze(0).to(self.device)
        features_tensor = torch.FloatFloat(features).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatFloat([target]).unsqueeze(0).to(self.device)

        self.policy_value_net.train()
        self.policy_value_optimizer.zero_grad()
        policy, value = self.policy_value_net(state_tensor, prob_matrix_tensor, turn_tensor, num_own_pieces_tensor, num_opponent_pieces_tensor, features_tensor)
        loss = F.mse_loss(value, target_tensor)
        loss.backward()
        self.policy_value_optimizer.step()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
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

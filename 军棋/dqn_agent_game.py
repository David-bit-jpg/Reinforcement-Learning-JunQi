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
        self.prob_bn = nn.BatchNorm2d(32)  # 确保通道数为 32
        
        self.hidden_size = hidden_size
        
        self.policy_head = nn.Linear(hidden_size, 12)  # 12个棋子类别
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, board_state, prob_matrix, turn, num_own_pieces, num_opponent_pieces, features):
        if board_state.size(1) != 3:
            board_state = board_state[:, :3, :, :]  # 只取前3个通道

        board_state = F.relu(self.board_bn1(self.board_conv1(board_state)))
        board_state = F.relu(self.board_bn2(self.board_conv2(board_state)))
        board_state = board_state.reshape(board_state.size(0), -1)  # 使用 .reshape() 而不是 .view()

        if prob_matrix.dim() == 2:
            prob_matrix = prob_matrix.unsqueeze(0).unsqueeze(0)  # 增加两个维度以符合卷积层输入要求
        elif prob_matrix.dim() == 3:
            prob_matrix = prob_matrix.unsqueeze(1)  # 增加一个维度以符合卷积层输入要求

        prob_matrix = F.relu(self.prob_bn(self.prob_conv(prob_matrix)))  # 修正这一行
        prob_matrix = prob_matrix.reshape(prob_matrix.size(0), -1)  # 使用 .reshape() 而不是 .view()

        turn = turn.reshape(turn.size(0), -1)  # 使用 .reshape() 而不是 .view()
        num_own_pieces = num_own_pieces.reshape(num_own_pieces.size(0), -1)  # 使用 .reshape() 而不是 .view()
        num_opponent_pieces = num_opponent_pieces.reshape(num_opponent_pieces.size(0), -1)  # 使用 .reshape() 而不是 .view()
        features = features.reshape(features.size(0), -1)  # 使用 .reshape() 而不是 .view()

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

import copy
import random
import torch
class POMCP:
    def __init__(self, env, weights, num_simulations, agent, gamma=0.99):
        self.env = env
        self.num_simulations = num_simulations
        self.weights = weights
        self.agent = agent  # 新增
        self.gamma = gamma  # 初始折扣因子

    def update_weights(self, weights):
        self.weights = weights

    def search(self, initial_state, player_color, pi, pi_reg):
        for _ in range(self.num_simulations):
            self._simulate(initial_state, player_color, pi, pi_reg, depth=0)
        return self._best_actions(initial_state, player_color)

    def _simulate(self, state, player_color, pi, pi_reg, depth):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(state)

        max_simulation_depth = 30  # 初始模拟深度
        if len(env_copy.red_pieces) + len(env_copy.blue_pieces) < 20:
            max_simulation_depth = 50  # 棋子较少时增加模拟深度
        elif len(env_copy.red_pieces) + len(env_copy.blue_pieces) < 10:
            max_simulation_depth = 100  # 棋子更少时进一步增加模拟深度

        total_reward = 0  # 累计奖励
        total_value = 0  # 累计价值估计
        current_gamma = self.gamma ** depth  # 随着深度增加折扣因子递减

        for _ in range(max_simulation_depth):  # 使用动态调整后的模拟深度
            valid_actions = env_copy.get_valid_actions(player_color)
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            next_state, reward, done, _ = env_copy.step_with_inference(action, pi, pi_reg, player_color, self.weights)
            total_reward += current_gamma * reward  # 折扣后的累计奖励

            # 使用策略价值网络估计当前状态的价值
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.agent.device)
            state_tensor = state_tensor.reshape(1, 3, self.env.board_rows, self.env.board_cols)
            prob_matrix = self.agent.get_prob_matrix(next_state, player_color)  # 使用agent调用
            prob_matrix_tensor = torch.FloatTensor(prob_matrix).unsqueeze(0).unsqueeze(0).to(self.agent.device)
            turn_tensor = torch.FloatTensor([env_copy.turn]).unsqueeze(0).to(self.agent.device)
            num_own_pieces_tensor = torch.FloatTensor([len(env_copy.red_pieces)]).unsqueeze(0).to(self.agent.device)
            num_opponent_pieces_tensor = torch.FloatTensor([len(env_copy.blue_pieces)]).unsqueeze(0).to(self.agent.device)
            features_tensor = torch.FloatTensor([]).unsqueeze(0).to(self.agent.device)

            with torch.no_grad():
                _, value_estimation = self.agent.policy_value_net(state_tensor, prob_matrix_tensor, turn_tensor, num_own_pieces_tensor, num_opponent_pieces_tensor, features_tensor)
            total_value += current_gamma * value_estimation.item()  # 折扣后的累计价值估计

            if done or (isinstance(env_copy.state, str) and env_copy.state in ['red_wins', 'blue_wins']):
                break

            # 更新当前的折扣因子
            current_gamma *= self.gamma

        # 计算最终的综合评估值（折扣后的累计奖励 + 折扣后的累计价值估计）
        return total_reward + total_value

    def _best_actions(self, state, player_color):
        valid_actions = self.env.get_valid_actions(player_color)
        action_values = [self._evaluate_action(state, a, player_color, depth=0) for a in valid_actions]
        best_action_indices = sorted(range(len(action_values)), key=lambda i: action_values[i], reverse=True)[:20]
        best_actions = [valid_actions[i] for i in best_action_indices]
        return best_actions

    def _evaluate_action(self, state, action, player_color, depth):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(state)
        next_state, reward, done, _ = env_copy.step_with_inference(action, self.env.pi, self.env.pi_reg, player_color, self.weights)

        # 使用策略价值网络估计当前状态的价值
        state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.agent.device)
        state_tensor = state_tensor.reshape(1, 3, self.env.board_rows, self.env.board_cols)
        prob_matrix = self.agent.get_prob_matrix(next_state, player_color)  # 使用agent调用
        prob_matrix_tensor = torch.FloatTensor(prob_matrix).unsqueeze(0).unsqueeze(0).to(self.agent.device)
        turn_tensor = torch.FloatTensor([env_copy.turn]).unsqueeze(0).to(self.agent.device)
        num_own_pieces_tensor = torch.FloatTensor([len(env_copy.red_pieces)]).unsqueeze(0).to(self.agent.device)
        num_opponent_pieces_tensor = torch.FloatTensor([len(env_copy.blue_pieces)]).unsqueeze(0).to(self.agent.device)
        features_tensor = torch.FloatTensor([]).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            _, value_estimation = self.agent.policy_value_net(state_tensor, prob_matrix_tensor, turn_tensor, num_own_pieces_tensor, num_opponent_pieces_tensor, features_tensor)

        # 计算综合评估值（折扣后的奖励 + 折扣后的价值估计）
        current_gamma = self.gamma ** depth
        return reward + current_gamma * value_estimation.item()
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
        self.policy_value_net = PolicyValueNet(board_size=13, num_features=5, hidden_size=128).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.policy_value_optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0
        self.eta = 0.01
        epsilon = 1e-8
        self.pi = np.full(action_size, (1.0 / action_size) + epsilon)
        self.pi_reg = np.full(action_size, (1.0 / action_size) + epsilon)
        # 初始权重
        self.weights = {
            "base_reward": 1.0,
            "protection_reward": 0.8,
            "attack_important_reward": 1.2,
            "defense_strategy_reward": 1.0,
            "engineer_mine_reward": 1.0,
            "movement_distance_reward": 0.5,
            "bluffing_reward": 0.7,
            "collaboration_reward": 0.6,
            "defensive_attack_reward": 0.9,
            "attack_flag_reward": 1.0,
            "bomb_risk_reward": 0.8,
            "offensive_defensive_reward": 1.0,
            "block_opponent_strategy_reward": 1.0,
            "aggressive_attack_reward": 1.0
        }
        self.pomcp = POMCP(env, self.weights, num_simulations=50, agent= self)
        
    def get_weights(self):
        return self.weights
            
    def adjust_weights(self, game_state, player_color):
        # 确定当前玩家和对手的棋子颜色
        own_color = player_color
        opponent_color = 'blue' if player_color == 'red' else 'red'

        # 获取当前游戏状态的相关信息
        num_own_pieces = len([p for p in self.env.get_pieces(own_color) if p.get_position()])
        num_opponent_pieces = len([p for p in self.env.get_pieces(opponent_color) if p.get_position()])
        own_flag_position = self.env.get_flag_position(own_color)
        opponent_flag_position = self.env.get_flag_position(opponent_color)

        # 高级棋子数量
        num_own_high_value_pieces = len([p for p in self.env.get_pieces(own_color) if p.get_position() and p.get_name() in ['司令', '军长']])
        num_opponent_high_value_pieces = len([p for p in self.env.get_pieces(opponent_color) if p.get_position() and p.get_name() in ['司令', '军长']])

        # 战略位置权重
        own_railway_control = sum([1 for p in self.env.get_pieces(own_color) if p.get_position() and self.env.is_railway(p.get_position())])
        opponent_railway_control = sum([1 for p in self.env.get_pieces(opponent_color) if p.get_position() and self.env.is_railway(p.get_position())])
        own_camp_control = sum([1 for p in self.env.get_pieces(own_color) if p.get_position() and self.env.is_in_camp(p.get_position())])
        opponent_camp_control = sum([1 for p in self.env.get_pieces(opponent_color) if p.get_position() and self.env.is_in_camp(p.get_position())])

        # 调整进攻与防守的权重
        if num_own_pieces > num_opponent_pieces:
            self.weights["attack_important_reward"] += 0.2
            self.weights["protection_reward"] -= 0.2
        else:
            self.weights["attack_important_reward"] -= 0.2
            self.weights["protection_reward"] += 0.2

        # 如果己方军旗位置已知，则增加保护军旗的权重
        if own_flag_position:
            self.weights["protection_reward"] += 0.3
        else:
            self.weights["protection_reward"] -= 0.3

        # 如果对方军旗位置已知，则增加攻击敌方军旗的权重
        if opponent_flag_position:
            self.weights["attack_flag_reward"] += 0.3
        else:
            self.weights["attack_flag_reward"] -= 0.3

        # 根据高级棋子的剩余数量调整权重
        if num_own_high_value_pieces > num_opponent_high_value_pieces:
            self.weights["defensive_attack_reward"] += 0.2
        else:
            self.weights["defensive_attack_reward"] -= 0.2

        # 检查己方高级棋子是否处于危险位置，如果是则增加保护权重
        for piece in self.env.get_pieces(own_color):
            if piece.get_position() and piece.get_name() in ['司令', '军长']:
                if not self.env.is_in_camp(piece.get_position()):
                    self.weights["protection_reward"] += 0.3
                else:
                    self.weights["protection_reward"] -= 0.1

        # 增加对关键位置的防御权重，例如铁路和行营
        for piece in self.env.get_pieces(own_color):
            if piece.get_position() and self.env.is_railway(piece.get_position()):
                self.weights["defensive_attack_reward"] += 0.2
            else:
                self.weights["defensive_attack_reward"] -= 0.1
            if piece.get_position() and self.env.is_in_camp(piece.get_position()):
                self.weights["protection_reward"] += 0.2
            else:
                self.weights["protection_reward"] -= 0.1

        # 使用铁路和行营控制的数量来调整权重
        self.weights["attack_important_reward"] += 0.1 * own_railway_control
        self.weights["protection_reward"] -= 0.1 * opponent_railway_control
        self.weights["attack_important_reward"] += 0.1 * own_camp_control
        self.weights["protection_reward"] -= 0.1 * opponent_camp_control

        # 增加战术欺骗的奖励
        for piece in self.env.get_pieces(own_color):
            if piece.get_position() and piece.get_name() in ['工兵', '排长']:
                self.weights["bluffing_reward"] += 0.2
            else:
                self.weights["bluffing_reward"] -= 0.1
            if piece.get_position() and piece.get_name() in ['连长', '营长', '团长']:
                self.weights["bluffing_reward"] += 0.1
            else:
                self.weights["bluffing_reward"] -= 0.05

        # 检查对方司令是否已经死亡
        opponent_commander_dead = not any(p.get_name() == '司令' and p.get_color() == opponent_color for p in self.env.get_pieces(opponent_color) if p.get_position())
        if opponent_commander_dead:
            self.weights["attack_flag_reward"] += 0.4  # 对方司令死亡后，增加对军旗的攻击权重
        else:
            self.weights["attack_flag_reward"] -= 0.2

        # 识破对方进攻意图并进行应对
        for piece in self.env.get_pieces(opponent_color):
            if piece.get_position() and piece.get_name() == '工兵':
                target_positions = self.env.get_valid_moves(piece)
                for target in target_positions:
                    if self.env.get_piece_at_position(target) and self.env.get_piece_at_position(target).get_name() in ['地雷', '炸弹']:
                        self.weights["block_opponent_strategy_reward"] += 0.3
                    else:
                        self.weights["block_opponent_strategy_reward"] -= 0.1

        # 增加对关键位置的进攻权重
        for piece in self.env.get_pieces(own_color):
            if piece.get_position() and self.env.is_railway(piece.get_position()):
                self.weights["aggressive_attack_reward"] += 0.2
            else:
                self.weights["aggressive_attack_reward"] -= 0.1
            if piece.get_position() and self.env.is_in_camp(piece.get_position()):
                self.weights["aggressive_attack_reward"] += 0.2
            else:
                self.weights["aggressive_attack_reward"] -= 0.1

        # 确保所有必要的键都存在
        required_keys = [
            "base_reward", "protection_reward", "attack_important_reward", "defense_strategy_reward",
            "engineer_mine_reward", "movement_distance_reward", "bluffing_reward", "collaboration_reward",
            "defensive_attack_reward", "attack_flag_reward", "bomb_risk_reward", "offensive_defensive_reward",
            "block_opponent_strategy_reward", "aggressive_attack_reward"
        ]
        for key in required_keys:
            if key not in self.weights:
                self.weights[key] = 1.0  # 或者其他合理的初始值

    def get_prob_matrix(self, state, player_color):
        prob_matrix = np.zeros((self.env.board_rows, self.env.board_cols))
        
        # 确定当前玩家和对手的棋子颜色
        own_color = player_color
        opponent_color = 'blue' if player_color == 'red' else 'red'
        
        for piece in self.env.get_pieces(own_color):
            if piece.position:
                x, y = piece.position
                prob_matrix[x, y] = self.env.piece_name_to_value(piece.get_name())
                
        for piece in self.env.get_pieces(opponent_color):
            if piece.position:
                x, y = piece.position
                prob_matrix[x, y] = -self.env.piece_name_to_value(piece.get_name())  # 对手的棋子用负值表示
        
        return prob_matrix

            
    def act(self, state, turn, num_own_pieces, num_opponent_pieces, features, player_color):
        self.adjust_weights(state, player_color=player_color)
        self.pomcp.update_weights(self.weights)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # 调整维度
        state_tensor = state_tensor.reshape(1, 3, self.env.board_rows, self.env.board_cols)  # 确保是 (1, 3, board_rows, board_cols)

        prob_matrix = self.get_prob_matrix(state, player_color=player_color)
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
        mcts_actions = self.pomcp.search(self.env.get_state(), player_color, self.pi, self.pi_reg)
        mcts_valid_actions = [a for a in mcts_actions if a in valid_actions]

        # 在蒙特卡洛树搜索筛选出的动作中进行强化学习策略的评估
        if random.random() > self.epsilon:
            action = mcts_valid_actions[np.argmax([combined_values[a] for a in mcts_valid_actions])]
        else:
            action = random.choice(mcts_valid_actions)

        self.pi = policy.cpu().numpy()  # 将策略值保存为 numpy 数组
        self.pi_reg = self.pi.copy()    # 将正则化策略值保存为 numpy 数组
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

        # 确保动作索引不超过当前Q网络的动作维度
        max_action_idx = self.qnetwork_local(states).size(1) - 1
        actions = torch.clamp(actions, 0, max_action_idx)

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = (Q_expected - Q_targets).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 将 loss 转换为一维数组
        self.memory.update_priorities(indices, loss.detach().cpu().numpy().reshape(-1))

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        q_values = self.qnetwork_local(states).detach().cpu().numpy()
        if len(self.pi) != q_values.shape[1]:
            min_len = min(len(self.pi), q_values.shape[1])
            self.pi = self.pi[:min_len]
            q_values = q_values[:, :min_len]

        self.pi = self.replicator_dynamics_update(self.pi, q_values.mean(axis=0))

        # 策略更新
        policy_loss = self.policy_loss(states, actions.squeeze(), rewards)
        self.policy_value_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_value_optimizer.step()



    def replicator_dynamics_update(self, pi, q_values, learning_rate=0.01):
        if len(pi) != len(q_values):
            min_len = min(len(pi), len(q_values))
            pi = pi[:min_len]
            q_values = q_values[:min_len]
        
        new_pi = pi * np.exp(learning_rate * q_values)
        new_pi /= new_pi.sum()
        return new_pi
    
    def policy_loss(self, states, actions, rewards):
        log_probs = torch.log(self.policy_value_net(states)[0])
        selected_log_probs = rewards * log_probs[np.arange(len(actions)), actions]
        loss = -selected_log_probs.mean()
        return loss


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
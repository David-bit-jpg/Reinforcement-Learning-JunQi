import numpy as np

class RewardModel:
    def __init__(self, red_pieces, blue_pieces, env, eta=0.01):
        self.eta = eta
        self.red_pieces = red_pieces
        self.blue_pieces = blue_pieces
        self.env = env
        self.reset()

    def reset(self):
        self.total_reward = 0

    def get_reward(self, state, action, outcome, pi_reg, pi,player_color):
        piece, target_position = action
        # 在报错处之前添加调试信息
        # print(f"pi: {pi}, type: {type(pi)}")
        # print(f"pi_reg: {pi_reg}, type: {type(pi_reg)}")
        
        # 确保 pi 和 pi_reg 是数组
        pi = np.array(pi)
        pi_reg = np.array(pi_reg)

        # 检查 action 是否在 pi 和 pi_reg 的范围内
        piece_index = self.env.get_piece_index(piece)
        action_index = self.env.encode_action(player_color, piece_index, target_position)
        action_prob = pi[action_index] if pi.all() != None and action_index < len(pi) else 0.01
        reg_prob = pi_reg[action_index] if pi_reg.all() != None and action_index < len(pi_reg) else 0.01

        # 防止除以零
        if action_prob == 0:
            action_prob = 1e-10
        if reg_prob == 0:
            reg_prob = 1e-10

        reward = self._calculate_base_reward(state, action, outcome)

        # 加入正则化奖励项
        reward -= self.eta * np.log(action_prob / reg_prob)

        # 加入虚张声势奖励
        bluffing_reward = self._calculate_bluffing_reward(action)
        reward += bluffing_reward

        # 添加协同作战奖励
        collaboration_reward = self._calculate_collaboration_reward(piece, target_position)
        reward += collaboration_reward

        # 添加进攻敌方军棋的奖励
        attack_flag_reward = self._calculate_attack_flag_reward(piece, target_position)
        reward += attack_flag_reward

        self.total_reward += reward
        return reward


    def _calculate_base_reward(self, state, action, outcome):
        piece, target_position = action
        reward = 0
        
        # 棋子的固定价值
        piece_value = self.get_piece_value(piece)
        reward += piece_value

        # 棋子所在位置的增加价值
        position_value = self.get_position_value(target_position)
        reward += position_value

        # 相邻棋子间的影响
        neighboring_value = self.get_neighboring_value(piece, target_position)
        reward += neighboring_value

        # 对己方军旗的保护和对敌方军旗的进攻
        flag_protection_value = self.get_flag_protection_value(piece, target_position)
        reward += flag_protection_value

        if outcome == 'win_game':
            reward += 1000  # 吃掉对方军旗的奖励

        return reward

    def get_piece_value(self, piece):
        value_map = {
            '排长': 1, '连长': 2, '营长': 3, '团长': 4,
            '旅长': 5, '师长': 6, '军长': 7, '司令': 8,
            '工兵': 0.5, '地雷': 0, '炸弹': 0, '军旗': 0
        }
        return value_map.get(piece.get_name(), 0)

    def get_position_value(self, position):
        railway_positions = self.env.railways.keys()
        camp_positions = self.env.get_camps()
        base_positions = [(0, 1), (0, 3), (12, 1), (12, 3)]
        highway_positions = self.env.define_highways()  # 公路线上的位置

        if position in railway_positions:
            return 5  # 铁路线上的高价值
        elif position in camp_positions:
            return 4  # 行营中的高价值
        elif position in base_positions:
            return 3  # 大本营中的价值
        elif position in highway_positions:
            return 2  # 公路线上的价值
        else:
            return 1  # 其他位置的基础价值

    def get_neighboring_value(self, piece, position):
        neighbors = self.get_neighbors(position)
        value = 0

        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece:
                if neighbor_piece.get_color() == piece.get_color():
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value += self.get_piece_value(neighbor_piece) * 0.1  # 保护作用
                else:
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value -= self.get_piece_value(neighbor_piece) * 0.1  # 威胁作用

        return value

    def get_flag_protection_value(self, piece, position):
        if piece.get_name() == '军旗':
            return 0

        flag_position = self.env.get_flag_position(piece.get_color())
        if not flag_position:
            return 0

        distance = self.get_manhattan_distance(position, flag_position)
        return 15 / max(distance, 1)

    def get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, position):
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= x + dx < self.env.board_rows and 0 <= y + dy < self.env.board_cols:
                neighbors.append((x + dx, y + dy))
        return neighbors

    def get_information_value(self, piece, target_position):
        """
        计算信息隐蔽的价值
        """
        current_position = piece.get_position()
        if current_position is None:
            return 0

        # 如果移动后会暴露重要棋子的位置，降低奖励
        if piece.get_name() in ['司令', '军长', '师长']:
            if self.is_position_exposed(target_position):
                return -5  # 根据实际需要调整惩罚值
        return 0

    def is_position_exposed(self, piece, position):
        """
        判断某个位置是否容易暴露
        """
        # 判断炸弹是否在师长或军长后面
        if piece.get_name() == '炸弹':
            nearby_positions = self.get_neighboring_positions(position)
            for nearby_pos in nearby_positions:
                nearby_piece = self.env.get_piece_at_position(nearby_pos)
                if nearby_piece and nearby_piece.get_color() == piece.get_color() and \
                nearby_piece.get_name() in ['师长', '军长']:
                    return True

        # 判断是否在横行霸道的位置
        if self.is_ba_dao_position(position):
            return True

        # 判断小子是否在大子前面
        if self.is_smaller_piece_protecting_larger(piece, position):
            return True

        return False

    def get_neighboring_positions(self, position):
        """
        获取邻近的位置
        """
        x, y = position
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.env.board_rows - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.env.board_cols - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def is_ba_dao_position(self, position):
        """
        判断是否在横行霸道的位置
        """
        # 根据实际定义横行霸道的位置
        # 假设横行霸道的位置是指棋盘中间的几行
        x, y = position
        return 5 <= x <= 7

    def is_smaller_piece_protecting_larger(self, piece, position):
        """
        判断小子是否在大子前面
        """
        nearby_positions = self.get_neighboring_positions(position)
        for nearby_pos in nearby_positions:
            nearby_piece = self.env.get_piece_at_position(nearby_pos)
            if nearby_piece and nearby_piece.get_color() == piece.get_color() and \
            self.get_piece_value(nearby_piece) > self.get_piece_value(piece):
                return True
        return False

    def _calculate_collaboration_reward(self, piece, target_position):
        """
        计算协同作战的奖励
        """
        reward = 0
        neighbors = self.get_neighbors(target_position)
        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece and neighbor_piece.get_color() == piece.get_color():
                reward += self.get_piece_value(neighbor_piece) * 0.1  # 协同作战奖励
        return reward

    def _calculate_bluffing_reward(self, action):
        piece, target_position = action
        reward = 0

        if piece.get_name() == '工兵':
            # 假装成炸弹
            reward += 5
        elif piece.get_name() in ['连长', '营长', '团长']:
            # 假装成中等价值棋子
            reward += 3

        return reward
    
    def _calculate_attack_flag_reward(self, piece, target_position):
        """
        计算进攻敌方军棋的奖励
        """
        reward = 0
        flag_position = self.env.get_flag_position('blue' if piece.get_color() == 'red' else 'red')
        if flag_position:
            distance = self.get_manhattan_distance(target_position, flag_position)
            reward += 10 / max(distance, 1)  # 鼓励靠近敌方军棋
        return reward

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

    def get_reward(self, state, action, outcome, pi_reg, pi, player_color):
        piece, target_position = action
        pi = np.array(pi)
        pi_reg = np.array(pi_reg)
        piece_index = self.env.get_piece_index(piece)
        action_index = self.env.encode_action(player_color, piece_index, target_position)
        action_prob = pi[action_index] if pi.all() != None and action_index < len(pi) else 0.01
        reg_prob = pi_reg[action_index] if pi_reg.all() != None and action_index < len(pi_reg) else 0.01

        if action_prob == 0:
            action_prob = 1e-10
        if reg_prob == 0:
            reg_prob = 1e-10

        reward = self._calculate_base_reward(state, action, outcome, player_color)
        reward -= self.eta * np.log(action_prob / reg_prob)

        bluffing_reward = self._calculate_bluffing_reward(action)
        reward += bluffing_reward

        collaboration_reward = self._calculate_collaboration_reward(piece, target_position)
        reward += collaboration_reward

        attack_flag_reward = self._calculate_attack_flag_reward(piece, target_position)
        reward += attack_flag_reward

        protection_reward = self._calculate_protection_reward(piece, target_position)
        reward += protection_reward

        attack_important_reward = self._calculate_attack_important_reward(piece, target_position)
        reward += attack_important_reward

        engineer_mine_reward = self._calculate_engineer_mine_reward(piece, target_position)
        reward += engineer_mine_reward

        movement_distance_reward = self._calculate_movement_distance_reward(piece, target_position)
        reward += movement_distance_reward

        defense_strategy_reward = self._calculate_defense_strategy_reward(piece, target_position, player_color)
        reward += defense_strategy_reward

        defensive_attack_reward = self._calculate_defensive_attack_reward(piece, target_position)
        reward += defensive_attack_reward

        death_penalty = self._calculate_death_penalty(piece, outcome)
        reward -= death_penalty

        camp_penalty = self._calculate_camp_penalty(piece, target_position)
        reward -= camp_penalty

        engineer_misfire_penalty = self._calculate_engineer_misfire_penalty(piece, target_position)
        reward -= engineer_misfire_penalty
        
        self.total_reward += reward
        return reward

    def _calculate_base_reward(self, state, action, outcome, player_color):
        piece, target_position = action
        reward = 0
        piece_value = self.get_piece_value(piece)
        reward += piece_value
        position_value = self.get_position_value(target_position)
        reward += position_value
        neighboring_value = self.get_neighboring_value(piece, target_position, player_color)
        reward += neighboring_value
        flag_protection_value = self.get_flag_protection_value(piece, target_position)
        reward += flag_protection_value

        if outcome == 'win_game':
            reward += 1000

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
        highway_positions = self.env.define_highways()

        if position in railway_positions:
            return 5
        elif position in camp_positions:
            return 4
        elif position in base_positions:
            return 3
        elif position in highway_positions:
            return 2
        else:
            return 1

    def get_neighboring_value(self, piece, position, player_color):
        neighbors = self.get_neighbors(position, player_color)
        value = 0

        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece:
                if neighbor_piece.get_color() == piece.get_color():
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value += self.get_piece_value(neighbor_piece) * 0.1
                else:
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value -= self.get_piece_value(neighbor_piece) * 0.1

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

    def get_neighbors(self, position, player_color):
        neighbors = set()
        for piece in self.env.red_pieces + self.env.blue_pieces:
            if piece.get_position() and piece.get_color() == player_color:
                valid_moves = self.env.get_valid_moves(piece)
                if position in valid_moves:
                    neighbors.add(piece.get_position())
        return list(neighbors)

    def _calculate_protection_reward(self, piece, target_position):
        reward = 0
        if piece.get_name() in ['司令', '军长', '师长']:
            if self.env.is_in_base(target_position):
                reward += 10
        return reward

    def _calculate_attack_important_reward(self, piece, target_position):
        reward = 0
        opponent_piece = self.env.get_piece_at_position(target_position)
        if opponent_piece and opponent_piece.get_color() != piece.get_color():
            if opponent_piece.get_name() in ['司令', '军长', '师长']:
                reward += 20
        return reward

    def _calculate_engineer_mine_reward(self, piece, target_position):
        reward = 0
        opponent_piece = self.env.get_piece_at_position(target_position)
        if piece.get_name() == '工兵' and opponent_piece and opponent_piece.get_name() == '炸弹':
            reward += 50
        return reward

    def _calculate_movement_distance_reward(self, piece, target_position):
        reward = 0
        current_position = piece.get_position()
        if current_position:
            distance = self.get_manhattan_distance(current_position, target_position)
            reward += distance * 0.5
        return reward

    def _calculate_defense_strategy_reward(self, piece, target_position, player_color):
        reward = 0
        flag_position = self.env.get_flag_position(piece.get_color())
        if flag_position:
            distance = self.get_manhattan_distance(target_position, flag_position)
            reward += 10 / max(distance, 1)

        if self.env.is_in_camp(target_position) and not self.env.get_piece_at_position(target_position):
            reward += self.get_camp_reward(target_position, piece.get_color())

        opponent_color = 'red' if piece.get_color() == 'blue' else 'blue'
        for camp in self.env.get_camps():
            occupant = self.env.get_piece_at_position(camp)
            if occupant and occupant.get_color() == opponent_color:
                reward -= self.get_camp_penalty(camp, piece.get_color())

        reward += self._calculate_guarding_reward(piece, target_position)

        return reward

    def get_camp_reward(self, position, color):
        reward = 0
        if color == 'red':
            reward += (12 - position[0]) * 2
        else:
            reward += position[0] * 2
        return reward

    def get_camp_penalty(self, position, color):
        penalty = 0
        if color == 'red':
            penalty += position[0] * 3
        else:
            penalty += (12 - position[0]) * 3
        return penalty

    def _calculate_guarding_reward(self, piece, target_position):
        reward = 0
        piece_value = self.get_piece_value(piece)
        for neighbor_pos in self.get_neighbors(target_position, piece.get_color()):
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece and neighbor_piece.get_color() == piece.get_color():
                if self.get_piece_value(neighbor_piece) < piece_value:
                    valid_moves = self.env.get_valid_moves(neighbor_piece)
                    if target_position in valid_moves:
                        reward += piece_value * 0.2
                if piece_value >= 6 and neighbor_piece.get_name() == '炸弹':
                    reward += 10
        return reward

    def _calculate_defensive_attack_reward(self, piece, target_position):
        reward = 0
        if piece.get_name() in ['旅长', '团长'] and not self.env.is_in_base(target_position):
            opponent_piece = self.env.get_piece_at_position(target_position)
            if opponent_piece and opponent_piece.get_color() != piece.get_color():
                reward += 15
        return reward

    def _calculate_death_penalty(self, piece, outcome):
        penalty = 0
        if outcome == 'lose_battle':
            penalty += self.get_piece_value(piece) * 10
        return penalty

    def _calculate_camp_penalty(self, piece, target_position):
        penalty = 0
        if self.env.is_in_camp(target_position) and not self.env.get_piece_at_position(target_position):
            penalty += self.get_camp_penalty(target_position, piece.get_color())
        return penalty

    def _calculate_bluffing_reward(self, action):
        piece, target_position = action
        reward = 0

        if piece.get_name() == '工兵':
            reward += 5
        elif piece.get_name() in ['连长', '营长', '团长']:
            reward += 3

        return reward

    def _calculate_collaboration_reward(self, piece, target_position):
        reward = 0
        neighbors = self.get_neighbors(target_position, piece.get_color())
        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece and neighbor_piece.get_color() == piece.get_color():
                reward += self.get_piece_value(neighbor_piece) * 0.1
        return reward

    def _calculate_engineer_misfire_penalty(self, piece, target_position):
        penalty = 0
        if piece.get_name() == '工兵':
            opponent_piece = self.env.get_piece_at_position(target_position)
            if opponent_piece and opponent_piece.get_color() != piece.get_color():
                if opponent_piece.get_name() not in ['地雷', '炸弹']:
                    penalty += 5
        return penalty
    def _calculate_attack_flag_reward(self, piece, target_position):
        """
        计算进攻敌方军旗的奖励
        """
        reward = 0
        flag_position = self.env.get_flag_position('blue' if piece.get_color() == 'red' else 'red')

        if flag_position:
            distance = self.get_manhattan_distance(target_position, flag_position)
            reward += 10 / max(distance, 1)  # 鼓励靠近敌方军旗
        else:
            inferred_flag_position = self.env.inference_model.get_inferred_flag_position('blue' if piece.get_color() == 'red' else 'red')
            if inferred_flag_position:
                distance = self.get_manhattan_distance(target_position, inferred_flag_position)
                reward += 5 / max(distance, 1)  # 鼓励靠近推测的敌方军旗

        return reward

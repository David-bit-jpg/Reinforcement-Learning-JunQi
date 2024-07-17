import numpy as np

class RewardModel:
    def __init__(self, red_pieces, blue_pieces, env, eta=0.01):
        self.eta = eta
        self.red_pieces = red_pieces
        self.blue_pieces = blue_pieces
        self.env = env
        self.reset()
        self.engineer_death_count = 0
        self.bomb_minor_piece_death_count = 0

    def reset(self):
        self.total_reward = 0
        self.engineer_death_count = 0
        self.bomb_minor_piece_death_count = 0

    def get_reward(self, state, action, outcome, pi_reg, pi, player_color, weights):
        piece, target_position = action
        pi = np.array(pi)
        pi_reg = np.array(pi_reg)
        piece_index = self.env.get_piece_index(piece)
        action_index = self.env.encode_action(player_color, piece_index, target_position)
        action_prob = pi[action_index] if pi.all() != None and action_index < len(pi) else 0.01
        reg_prob = pi_reg[action_index] if pi_reg.all() != None and action_index < len(pi_reg) else 0.01

        action_prob = max(action_prob, 1e-10)
        reg_prob = max(reg_prob, 1e-10)

        reward = 0
        if self._is_special_move(piece, target_position, outcome):
            reward += self._calculate_special_move_rewards(piece, target_position, outcome, player_color, weights)
            reward -= self.eta * np.log(action_prob / reg_prob)

        reward -= self._calculate_additional_penalties(piece, target_position, outcome)

        self.total_reward += reward
        return reward


    def _is_special_move(self, piece, target_position, outcome):
        # 判断是否为特殊动作
        if piece.get_name() in ['工兵', '司令', '军长', '师长']:
            return True
        if outcome == 'win_game':
            return True
        if target_position in self.env.get_camps():
            return True
        return False

    def _calculate_special_move_rewards(self, piece, target_position, outcome, player_color, weights):
        rewards = {
            "attack_flag_reward": self._calculate_attack_flag_reward(piece, target_position),
            "protection_reward": self._calculate_protection_reward(piece, target_position),
            "attack_important_reward": self._calculate_attack_important_reward(piece, target_position),
            "engineer_mine_reward": self._calculate_engineer_mine_reward(piece, target_position),
            "movement_distance_reward": self._calculate_movement_distance_reward(piece, target_position),
            "defense_strategy_reward": self._calculate_defense_strategy_reward(piece, target_position, weights),
            "defensive_attack_reward": self._calculate_defensive_attack_reward(piece, target_position),
            "bomb_risk_reward": self._calculate_bomb_risk_reward(piece, target_position),
            "offensive_defensive_reward": self._calculate_offensive_defensive_reward(piece, target_position, player_color, weights),
            "block_opponent_strategy_reward": self._calculate_block_opponent_strategy_reward(piece, target_position, player_color),
            "aggressive_attack_reward": self._calculate_aggressive_attack_reward(piece, target_position, player_color)  # 新增主动进攻奖励
        }
        total_reward = sum(rewards[key] * weights.get(key, 1.0) for key in rewards)
        return total_reward

    def _calculate_additional_penalties(self, piece, target_position, outcome):
        penalties = {
            "death_penalty": self._calculate_death_penalty(piece, outcome),
            "camp_penalty": self._calculate_camp_penalty(piece, target_position),
            "engineer_misfire_penalty": self._calculate_engineer_misfire_penalty(piece, target_position),
            "invalid_move_penalty": self._calculate_invalid_move_penalty(piece, target_position),
            "dangerous_position_penalty": self._calculate_dangerous_position_penalty(piece, target_position)
        }
        total_penalty = sum(penalties[key] for key in penalties)
        return total_penalty

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
            reward += 100

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
            return 2
        elif position in camp_positions:
            return 1.5
        elif position in base_positions:
            return 1.2
        elif position in highway_positions:
            return 1
        else:
            return 0.5

    def get_neighboring_value(self, piece, position, player_color):
        neighbors = self.get_neighbors(position, player_color)
        value = 0

        for neighbor_pos in neighbors:
            neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
            if neighbor_piece:
                if neighbor_piece.get_color() == piece.get_color():
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value += self.get_piece_value(neighbor_piece) * 0.05
                else:
                    if self.get_piece_value(neighbor_piece) > self.get_piece_value(piece):
                        value -= self.get_piece_value(neighbor_piece) * 0.05

        return value

    def get_flag_protection_value(self, piece, position):
        if piece.get_name() == '军旗':
            return 0

        flag_position = self.env.get_flag_position(piece.get_color())
        if not flag_position:
            return 0

        distance = self.get_manhattan_distance(position, flag_position)
        return 7 / max(distance, 1)

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
                reward += 5
        return reward

    def _calculate_attack_important_reward(self, piece, target_position):
        reward = 0
        opponent_piece = self.env.get_piece_at_position(target_position)
        if opponent_piece and opponent_piece.get_color() != piece.get_color():
            if opponent_piece.get_name() in ['司令', '军长', '师长']:
                reward += 10
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

    def _calculate_defense_strategy_reward(self, piece, target_position, weights):
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
                        reward += piece_value * 0.1
                if piece_value >= 6 and neighbor_piece.get_name() == '炸弹':
                    reward += 5
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
            penalty += self.get_piece_value(piece) * 20
            if piece.get_name() == '工兵':
                self.engineer_death_count += 1
                penalty += 10 * self.engineer_death_count
            elif piece.get_name() in ['司令', '军长', '师长']:
                penalty += 50
        return penalty

    def _calculate_bomb_minor_piece_penalty(self, piece):
        penalty = 0
        if piece.get_name() in ['旅长', '团长', '营长', '连长', '排长', '工兵']:
            self.bomb_minor_piece_death_count += 1
            penalty += 10 * self.bomb_minor_piece_death_count
        return penalty

    def _calculate_camp_penalty(self, piece, target_position):
        penalty = 0
        if piece.get_color() == 'red':
            enemy_color = 'blue'
        else:
            enemy_color = 'red'

        if self.env.is_in_camp(target_position):
            occupant = self.env.get_piece_at_position(target_position)
            if occupant and occupant.get_color() == enemy_color:
                flag_position = self.env.get_flag_position(piece.get_color())
                distance_to_flag = self.get_manhattan_distance(target_position, flag_position)
                penalty += (12 - distance_to_flag) * 3  # 距离越近，惩罚越严重
        return penalty

    def _calculate_bluffing_reward(self, piece, target_position):
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

    def _calculate_dangerous_position_penalty(self, piece, target_position):
        penalty = 0
        if not self.env.is_in_camp(target_position):
            neighbors = self.get_neighbors(target_position, 'red' if piece.get_color() == 'blue' else 'blue')
            for neighbor_pos in neighbors:
                neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
                if neighbor_piece and neighbor_piece.get_color() != piece.get_color():
                    penalty += 5
        return penalty
    
    def _calculate_invalid_move_penalty(self, piece, target_position):
        # 检查是否为有效移动
        valid_moves = self.env.get_valid_moves(piece)
        if target_position not in valid_moves:
            return 10  # 给予惩罚分数
        return 0

    def _calculate_bomb_risk_reward(self, piece, target_position):
        # 判断大棋子是否面临被炸弹炸掉的风险，并考虑权衡
        reward = 0
        opponent_piece = self.env.get_piece_at_position(target_position)
        if piece.get_name() in ['师长', '军长', '司令']:
            if opponent_piece and opponent_piece.get_name() == '炸弹':
                reward -= self.get_piece_value(piece) * 2  # 被炸弹炸掉的惩罚
            else:
                reward += self.get_piece_value(piece) * 0.1  # 进攻的奖励
        return reward

    def _calculate_offensive_defensive_reward(self, piece, target_position, player_color, weights):
        # 评估子力对比，判断需要进行防守还是进攻
        reward = 0
        own_pieces_value = sum(self.get_piece_value(p) for p in (self.red_pieces if player_color == 'red' else self.blue_pieces) if p.get_position())
        opponent_pieces_value = sum(self.get_piece_value(p) for p in (self.blue_pieces if player_color == 'red' else self.red_pieces) if p.get_position())

        if own_pieces_value > opponent_pieces_value:
            reward += 5  # 进攻奖励
        else:
            reward += 3  # 防守奖励

        return reward

    def _calculate_block_opponent_strategy_reward(self, piece, target_position, player_color):
        # 阻止对手的战略，例如阻止工兵清除地雷或炸弹
        reward = 0
        opponent_color = 'blue' if player_color == 'red' else 'red'
        if piece.get_name() != '地雷':
            neighbors = self.get_neighbors(target_position, opponent_color)
            for neighbor_pos in neighbors:
                neighbor_piece = self.env.get_piece_at_position(neighbor_pos)
                if neighbor_piece and neighbor_piece.get_color() == opponent_color and neighbor_piece.get_name() == '工兵':
                    reward += 5  # 阻止对方工兵的奖励
        return reward

    def _calculate_aggressive_attack_reward(self, piece, target_position, player_color):
        # 鼓励主动进攻的奖励
        reward = 0
        opponent_piece = self.env.get_piece_at_position(target_position)
        if opponent_piece and opponent_piece.get_color() != piece.get_color():
            reward += 10  # 主动进攻的奖励
        return reward
    
    def _calculate_defense_strategy_reward(self, piece, target_position, weights):
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

        # 新增对行营保护的奖励计算
        reward += self._calculate_protect_camps_reward(piece, target_position)

        return reward

    def _calculate_protect_camps_reward(self, piece, target_position):
        reward = 0
        own_color = piece.get_color()
        opponent_color = 'red' if own_color == 'blue' else 'blue'
        opponent_pieces = self.env.get_pieces(opponent_color)

        for camp in self.env.get_camps():
            if self.env.get_piece_at_position(camp) is None:
                for opponent_piece in opponent_pieces:
                    if opponent_piece.get_position() and target_position in self.env.get_valid_moves(opponent_piece):
                        if camp in self.env.get_valid_moves(opponent_piece):
                            # 如果对方棋子能在下一步占领行营，给予更高的奖励以优先占据行营
                            reward += 5
                            break
        return reward

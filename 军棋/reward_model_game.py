import numpy as np
import numpy as np

class RewardModel:
    def __init__(self, red, blue, eta=0.01):
        self.eta = eta
        self.reset()
        self.red_pieces = red
        self.blue_pieces = blue

    def reset(self):
        self.total_reward = 0

    def get_reward(self, state, action, outcome, correct_mark=False):
        reward = 0
        if state == 'setup':
            reward = self._play_phase_reward(action, outcome)
        
        self.total_reward += reward
        return reward
    
    def _play_phase_reward(self, action, outcome):
        piece, target_position = action
        reward = 0
        if outcome == 'valid_move':
            reward += 5
        elif outcome == 'invalid_move':
            reward -= 5
        elif outcome == 'win_battle':
            reward += 20
            reward += self._piece_value(piece, target_position)  # 奖励捕获重要棋子
        elif outcome == 'lose_battle':
            reward -= 10
        elif outcome == 'draw_battle':
            reward += 5

        piece_position = piece.get_position()
        if piece_position is not None:
            reg_term = self.eta * np.log(1 + np.abs(target_position[0] - piece_position[0]) + np.abs(target_position[1] - piece_position[1]))
            reward += reg_term
        else:
            reward -= 10  # 如果没有有效位置，给予负奖励

        # 推测信息奖励
        reward += self._inference_accuracy_reward(piece, target_position)

        return reward

    def _piece_value(self, piece, target_position):
        piece_values = {
            '军旗': 50,
            '司令': 40,
            '军长': 30,
            '师长': 25,
            '旅长': 20,
            '团长': 15,
            '营长': 10,
            '连长': 5,
            '排长': 5,
            '工兵': 5,
            '地雷': 5,
            '炸弹': 0  # 炸弹无价值，因为它会同时消灭自己和敌人
        }
        value = piece_values.get(piece.get_name(), 0)

        target_piece = self.get_piece_at_position(target_position)
        if target_piece:
            target_value = piece_values.get(target_piece.get_name(), 0)

            # 根据不同棋子战斗结果给予细化奖励
            if piece.get_name() == '炸弹':
                if target_piece.get_name() == '司令':
                    value += 100
                elif target_piece.get_name() == '军长':
                    value += 80
                elif target_piece.get_name() == '师长':
                    value += 50
                elif target_piece.get_name() in ['排长', '连长', '营长']:
                    value -= 10

            # 根据消灭对方棋子的差距增加奖励
            level_difference = abs(piece_values[piece.get_name()] - piece_values[target_piece.get_name()])
            value += max(0, 20 - level_difference)  # 差距越小，奖励越高

        return value

    def _inference_accuracy_reward(self, piece, target_position):
        reward = 0

        important_pieces = {'司令', '军长', '师长', '炸弹'}

        # 检查红方推测信息准确性
        if piece.get_color() == 'red':
            for p, possible_positions in self.red_infer.items():
                if piece == p:
                    if target_position in possible_positions:
                        reward += 10
                        if piece.get_name() in important_pieces:
                            reward += 20
                    else:
                        reward -= 5
                    reward += self._directional_inference_reward(piece, target_position, possible_positions)

        # 检查蓝方推测信息准确性
        if piece.get_color() == 'blue':
            for p, possible_positions in self.blue_infer.items():
                if piece == p:
                    if target_position in possible_positions:
                        reward += 10
                        if piece.get_name() in important_pieces:
                            reward += 20
                    else:
                        reward -= 5
                    reward += self._directional_inference_reward(piece, target_position, possible_positions)

        return reward

    def _directional_inference_reward(self, piece, target_position, possible_positions):
        reward = 0
        target_x, target_y = target_position
        for pos in possible_positions:
            pos_x, pos_y = pos
            if abs(target_x - pos_x) <= 1 and abs(target_y - pos_y) <= 1:
                reward += 2  # 给与方向性推测奖励
        return reward

    def get_total_reward(self):
        return self.total_reward

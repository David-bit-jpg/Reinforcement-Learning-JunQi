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
        # 检查 state 是否是字符串
        if isinstance(state, str) and state == 'setup':
            reward = self._setup_phase_reward(action, outcome)
        
        self.total_reward += reward
        return reward
    def _setup_phase_reward(self, action, outcome):
        piece, position = action
        reward = 0
        camps = [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]
        base_positions = [(12, 1), (12, 3), (0, 1), (0, 3)]

        if outcome == 'valid':
            # 检查是否在禁区位置
            if position in camps:
                reward -= 1000
            
            # 检查大棋子是否放在大本营位置
            if piece.get_name() in ['司令', '军长', '师长', '旅长', '团长', '营长', '炸弹','工兵'] and position in base_positions:
                reward -= 100
                
            # 高级将领放在前线
            if piece.get_name() in ['司令', '军长', '师长','旅长'] and position[0] in [5, 4, 3, 7, 8, 9]:
                reward += 5 
                
            if piece.get_name() in ['司令', '军长', '师长','旅长','工兵'] and position[0] in [0,12]:
                reward -= 20
                
            if piece.get_name() in ['司令', '军长', '师长','旅长','工兵'] and position[0] in [0,12] and position[1] in [2]:
                reward -= 5
            
            if piece.get_name() in ['地雷'] and position[0] in [1,11] and position[1] in [0, 4]:
                reward += 5
            
            if piece.get_name() in ['排长','连长'] and position in base_positions:
                reward += 5
                
            if piece.get_name() in ['团长', '连长', '营长'] and position[0] in [2, 3 ,4, 10, 9, 8]:
                reward += 5
            # 计算左右两边棋子的大小
            left_strength = 0
            right_strength = 0
            for p in self.red_pieces + self.blue_pieces:
                pos = p.get_position()
                if pos:
                    if pos[1] < 2:  # 左边
                        left_strength += self.get_piece_strength(pos)
                    elif pos[1] > 2:  # 右边
                        right_strength += self.get_piece_strength(pos)
            
            if piece.get_name() in ['炸弹', '地雷']:
                strength = 0
            else:
                strength = self.get_piece_strength(position)
            
            if left_strength > 0 and right_strength > 0:
                reward += 20 / (1 + abs(left_strength - right_strength))  # 左右两边棋子大小越接近，奖励越多
            # 弱侧用地雷防守加分
            if piece.get_name() == '地雷':
                if left_strength < right_strength and position[1] < 2:
                    reward += 5  # 左侧弱侧用地雷防守加分
                elif right_strength < left_strength and position[1] > 2:
                    reward += 5  # 右侧弱侧用地雷防守加分
            
            if piece.get_name() == '地雷':
                neighbors = [(position[0] + dx, position[1] + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                for neighbor in neighbors:
                    neighbor_piece = self.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_name() in ['工兵', '炸弹']:
                        reward -= 10 
                        break

                # 检查地雷是否保护了军旗
                flag_neighbors = [(position[0] + dx, position[1] + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                for flag_neighbor in flag_neighbors:
                    flag_piece = self.get_piece_at_position(flag_neighbor)
                    if flag_piece and flag_piece.get_name() == '军旗':
                        reward += 20  # 地雷保护军旗加分
                        break
                    
            # 检查旅长以上棋子周围是否有炸弹
            if piece.get_name() in ['军长', '师长', '旅长']:
                neighbors = [(position[0] + dx, position[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),(-2, 0), (2, 0), (0, -2), (0, 2)]]
                for neighbor in neighbors:
                    neighbor_piece = self.get_piece_at_position(neighbor)
                    if neighbor_piece and neighbor_piece.get_name() == '炸弹':
                        reward += 10  # 提高奖励值
                        break
            
            # 检查工兵不要放在第一排
            if piece.get_name() == '工兵' and position[0] in [5, 7]:
                reward -= 10  # 工兵放在第一排减少奖励

            reward += 1  # 其他合理位置
        else:
            reward -= 1000  # 无效布局减少惩罚力度

        return reward

    def get_piece_at_position(self, position):
        for piece in self.red_pieces + self.blue_pieces:
            if piece.get_position() == position:
                return piece
        return None

    def get_piece_strength(self, position):
        piece = self.get_piece_at_position(position)
        if piece:
            strength_dict = {'军旗': 0, '地雷': 1, '工兵': 2, '排长': 3, '连长': 4, '营长': 5, '团长': 6, '旅长': 7, '师长': 8, '军长': 9, '司令': 10, '炸弹': 11}
            return strength_dict.get(piece.get_name(), 0)
        return 0

import numpy as np

class RewardModel:
    def __init__(self, piece_types):
        self.piece_types = piece_types

    def get_reward(self, initial_belief, updated_belief, actual_belief):
        reward = 0
        for x in range(updated_belief.shape[0]):
            for y in range(updated_belief.shape[1]):
                for piece_index in range(len(self.piece_types)):
                    if updated_belief[x, y, piece_index] == actual_belief[x, y, piece_index]:
                        reward += 1
                    else:
                        reward -= 1
        return reward

    def update_belief(self, belief, move_history, battle_history):
        # 根据移动历史和战斗历史更新belief（推理逻辑实现）
        for move in move_history:
            piece, start_pos, end_pos = move
            belief[start_pos[0], start_pos[1], :] = 0
            piece_index = self.piece_types.index(piece.get_name())
            belief[end_pos[0], end_pos[1], piece_index] = 1
        
        # 处理战斗历史
        for battle in battle_history:
            attacker, defender, result = battle
            if result == 'win':
                defender_pos = defender.get_position()
                defender_index = self.piece_types.index(defender.get_name())
                belief[defender_pos[0], defender_pos[1], defender_index] = 0
            elif result == 'lose':
                attacker_pos = attacker.get_position()
                attacker_index = self.piece_types.index(attacker.get_name())
                belief[attacker_pos[0], attacker_pos[1], attacker_index] = 0
            elif result == 'draw':
                attacker_pos = attacker.get_position()
                defender_pos = defender.get_position()
                attacker_index = self.piece_types.index(attacker.get_name())
                defender_index = self.piece_types.index(defender.get_name())
                belief[attacker_pos[0], attacker_pos[1], attacker_index] = 0
                belief[defender_pos[0], defender_pos[1], defender_index] = 0
        
        # 添加特殊推理逻辑
        self.apply_special_rules(belief)

        return belief

    def apply_special_rules(self, belief):
        # 将特殊规则应用到belief更新中

        # 假设军旗只会出现在对方两个大本营之一中
        for x in range(belief.shape[0]):
            for y in range(belief.shape[1]):
                if (x, y) not in [(0, 1), (0, 3), (12, 1), (12, 3)]:
                    belief[x, y, self.piece_types.index('军旗')] = 0

        # 假设地雷只会出现在对方的后两排
        for x in range(belief.shape[0]):
            for y in range(belief.shape[1]):
                if x not in [0, 1, 11, 12]:
                    belief[x, y, self.piece_types.index('地雷')] = 0

        # 假设炸弹不在交通不发达（能到达地方少）的地方
        for x in range(belief.shape[0]):
            for y in range(belief.shape[1]):
                if not self.is_high_traffic_area((x, y)):
                    belief[x, y, self.piece_types.index('炸弹')] *= 0.5

        # 假设大棋子（师长及以上）周围存在炸弹的可能性更大
        for x in range(belief.shape[0]):
            for y in range(belief.shape[1]):
                if any(belief[x, y, self.piece_types.index(piece)] > 0.2 for piece in ['司令', '军长', '师长']):
                    for nx, ny in self.get_neighboring_positions((x, y)):
                        if self.is_valid_position((nx, ny)):
                            belief[nx, ny, self.piece_types.index('炸弹')] *= 1.5

    def is_high_traffic_area(self, position):
        # 定义高交通区域的逻辑，这里假设行营和铁路交叉点为高交通区域
        camps = [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]
        railways = [(1, 1), (1, 3), (11, 1), (11, 3)]
        return position in camps or position in railways

    def get_neighboring_positions(self, position):
        x, y = position
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < 12:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < 4:
            neighbors.append((x, y + 1))
        return neighbors

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x <= 12 and 0 <= y <= 4

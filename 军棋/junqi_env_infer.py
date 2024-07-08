import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from reward_model_infer import RewardModel
font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = fm.FontProperties(fname=font_path)

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

class JunQiEnvInfer(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, initial_board):
        super(JunQiEnvInfer, self).__init__()
        self.board_rows = 13
        self.board_cols = 5
        self.railways = self.define_railways()
        self.roads = self.define_roads()
        self.highways = self.define_highways()
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.piece_types = list(piece_encoding.keys())
        self.reward_model = RewardModel(self.piece_types)
        self.red_pieces, self.blue_pieces = self.create_pieces(initial_board)
        self.action_space = spaces.Discrete(len(self.piece_types) * self.board_rows * self.board_cols)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_rows, self.board_cols, len(self.piece_types)), dtype=np.float32)
        self.move_history = []
        self.battle_history = []

        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        for piece in self.red_pieces:
            position = piece.get_position()
            if position:
                self.occupied_positions_red.add(position)
        for piece in self.blue_pieces:
            position = piece.get_position()
            if position:
                self.occupied_positions_blue.add(position)
        self.state = 'play'
        self.turn = 0
        self.move_history = []
        self.battle_history = []
        self.generate_random_histories()  # 生成随机的move_history和battle_history
        return self.get_state()

    def generate_random_histories(self):
        piece_positions = [(piece, piece.get_position()) for piece in self.blue_pieces]
        random.shuffle(piece_positions)
        
        # 生成随机移动历史
        num_moves = random.randint(5, 15)
        for _ in range(num_moves):
            piece, pos = random.choice(piece_positions)
            valid_moves = self.get_valid_moves(piece)
            if valid_moves:
                new_pos = random.choice(valid_moves)
                self.move_history.append((piece, pos, new_pos))
                piece.set_position(new_pos)

        # 生成随机战斗历史
        num_battles = random.randint(3, 10)
        for _ in range(num_battles):
            attacker, attacker_pos = random.choice(piece_positions)
            defender, defender_pos = random.choice(piece_positions)
            if attacker.get_color() != defender.get_color() and attacker_pos != defender_pos:
                result = self.resolve_battle(attacker, defender)
                self.battle_history.append((attacker, defender, result))

    def resolve_battle(self, attacker, defender):
        piece_value = self.piece_name_to_value(attacker.get_name())
        target_value = self.piece_name_to_value(defender.get_name())

        if attacker.get_name() == '炸弹' or defender.get_name() == '炸弹':
            if attacker.get_name() == '炸弹' and defender.get_name() != '炸弹':
                attacker.set_position(None)
                defender.set_position(None)
                return 'draw'
            elif attacker.get_name() != '炸弹' and defender.get_name() == '炸弹':
                attacker.set_position(None)
                defender.set_position(None)
                return 'draw'
            elif attacker.get_name() == '炸弹' and defender.get_name() == '炸弹':
                attacker.set_position(None)
                defender.set_position(None)
                return 'draw'
        if attacker.get_name() == '工兵' and defender.get_name() == '地雷':
            defender.set_position(None)
            return 'win'
        if piece_value < target_value:
            defender.set_position(None)
            return 'win'
        if piece_value > target_value:
            attacker.set_position(None)
            return 'lose'
        if piece_value == target_value:
            attacker.set_position(None)
            defender.set_position(None)
            return 'draw'

    def get_valid_moves(self, piece):
        current_position = piece.get_position()
        valid_moves = set()

        if not current_position or piece.get_name() in ['军旗', '地雷']:
            return list(valid_moves)
        
        # 考虑铁路上的移动
        if self.is_railway(current_position):
            if piece.get_name() == '工兵':
                valid_moves.update(self.get_railway_moves_for_worker(current_position, piece))
            else:
                valid_moves.update(self.get_railway_moves(current_position, piece))

        # 考虑道路上的移动
        valid_moves.update(self.get_road_moves(current_position, piece))

        # 移除被占据的行营位置
        camps = self.get_camps()
        occupied_camps = set(camps) & (self.occupied_positions_red | self.occupied_positions_blue)
        valid_moves = [move for move in valid_moves if move not in occupied_camps]

        # 移除被友方棋子占据的位置
        valid_moves = [move for move in valid_moves if not self.is_friendly_occupied(move, piece.get_color())]

        return list(valid_moves)

    def get_state(self):
        state = np.zeros((self.board_rows, self.board_cols, len(self.piece_types)))
        for piece in self.blue_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    piece_index = self.piece_types.index(piece.get_name())
                    state[x, y, piece_index] = 1
        return {
            'board_state': state,
            'move_history': self.move_history,
            'battle_history': self.battle_history
        }

    def step(self, action):
        inferred_pieces = self.decode_action(action)  # 将整数动作解码为推理结果
        reward = self.calculate_reward(inferred_pieces)
        done = self.is_terminal_state()
        info = {}
        return self.get_state(), reward, done, info


    def calculate_reward(self, inferred_pieces):
        # 根据推理结果计算奖励
        actual_pieces = self.get_actual_pieces()
        correct_inferences = 0
        total_inferences = len(inferred_pieces)
        for pos, piece in inferred_pieces.items():
            if pos in actual_pieces and actual_pieces[pos] == piece:
                correct_inferences += 1
        return (correct_inferences / total_inferences)
    

    def get_actual_pieces(self):
        actual_pieces = {}
        for piece in self.blue_pieces:
            pos = piece.get_position()
            if pos:
                actual_pieces[pos] = piece.get_name()
        return actual_pieces

    def is_terminal_state(self):
        return False

    def create_pieces(self, initial_board):
        red_pieces = initial_board[0]
        blue_pieces = initial_board[1]
        for piece in red_pieces:
            position = piece.get_position()
            self.occupied_positions_red.add(position)
        for piece in blue_pieces:
            position = piece.get_position()
            self.occupied_positions_blue.add(position)
        return red_pieces, blue_pieces

    def get_valid_moves(self, piece):
        current_position = piece.get_position()
        valid_moves = set()

        if not current_position or piece.get_name() in ['军旗', '地雷']:
            return list(valid_moves)

        if self.is_railway(current_position):
            if piece.get_name() == '工兵':
                valid_moves.update(self.get_railway_moves_for_worker(current_position, piece))
            else:
                valid_moves.update(self.get_railway_moves(current_position, piece))

        valid_moves.update(self.get_road_moves(current_position, piece))
        camps = self.get_camps()
        occupied_camps = set(camps) & (self.occupied_positions_red | self.occupied_positions_blue)
        valid_moves = [move for move in valid_moves if move not in occupied_camps]
        valid_moves = [move for move in valid_moves if not self.is_friendly_occupied(move, piece.get_color())]

        return list(valid_moves)

    def get_railway_moves_for_worker(self, start_position, piece):
        valid_moves = set()
        queue = [start_position]
        visited = set()

        while queue:
            current_position = queue.pop(0)
            if current_position in visited:
                continue
            visited.add(current_position)

            if current_position != start_position:
                if not self.is_friendly_occupied(current_position, piece.get_color()):
                    valid_moves.add(current_position)

            for neighbor in self.railways.get(current_position, []):
                if neighbor not in visited:
                    if self.is_friendly_occupied(neighbor, piece.get_color()):
                        visited.add(neighbor)
                    else:
                        queue.append(neighbor)

        return list(valid_moves)

    def get_railway_moves(self, current_position, piece):
        valid_moves = []
        color = piece.get_color()

        self.expand_in_direction(current_position, color, valid_moves, is_horizontal=True)
        self.expand_in_direction(current_position, color, valid_moves, is_horizontal=False)

        return valid_moves

    def expand_in_direction(self, current_position, color, valid_moves, is_horizontal):
        if is_horizontal:
            y = current_position[1] - 1
            while y >= 0:
                next_pos = (current_position[0], y)
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                y -= 1

            y = current_position[1] + 1
            while y < self.board_cols:
                next_pos = (current_position[0], y)
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                y += 1
        else:
            x = current_position[0] - 1
            while x >= 0:
                next_pos = (x, current_position[1])
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                x -= 1

            x = current_position[0] + 1
            while x < self.board_rows:
                next_pos = (x, current_position[1])
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                x += 1

    def get_road_moves(self, current_position, piece):
        valid_moves = []
        color = piece.get_color()

        for pos in self.roads.get(current_position, []):
            if not self.is_friendly_occupied(pos, color):
                valid_moves.append(pos)
        return valid_moves
    
    def decode_action(self, action):
        inferred_pieces = {}
        total_positions = self.action_space.n // len(self.piece_types)
        piece_index = action // total_positions
        position_index = action % total_positions
        x = position_index // self.board_cols
        y = position_index % self.board_cols
        piece_type = self.piece_types[piece_index]
        inferred_pieces[(x, y)] = piece_type
        return inferred_pieces

    def is_occupied(self, position):
        return position in self.occupied_positions_red or position in self.occupied_positions_blue

    def is_friendly_occupied(self, position, color):
        return position in (self.occupied_positions_red if color == 'red' else self.occupied_positions_blue)

    def get_camps(self):
        return [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]

    def render(self, mode='human'):
        board = np.zeros((self.board_rows, self.board_cols), dtype=int)
        for piece in self.red_pieces:
            pos = piece.get_position()
            if pos:
                x, y = pos
                board[x, y] = piece_encoding[piece.get_name()]
        for piece in self.blue_pieces:
            pos = piece.get_position()
            if pos:
                x, y = pos
                board[x, y] = piece_encoding[piece.get_name()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xticks(np.arange(self.board_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.board_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.imshow(board, cmap='tab20', origin='upper')
        
        for piece in self.red_pieces + self.blue_pieces:
            pos = piece.get_position()
            if pos:
                x, y = pos
                ax.text(y, x, piece.get_name(), ha='center', va='center', fontproperties=font_prop, fontsize=12)

        plt.show()

    def is_railway(self, position):
        return position in self.railways
                
    def define_railways(self):
        return {
            (1, 0): [(1, 1), (2, 0)],
            (1, 1): [(1, 0), (1, 2)],
            (1, 2): [(1, 1), (1, 3)],
            (1, 3): [(1, 2), (1, 4)],
            (1, 4): [(1, 3), (2, 4)],

            (2, 0): [(1, 0), (3, 0)],
            (2, 4): [(1, 4), (3, 4)],

            (3, 0): [(2, 0), (4, 0)],
            (3, 4): [(2, 4), (4, 4)],

            (4, 0): [(3, 0), (5, 0)],
            (4, 4): [(3, 4), (5, 4)],

            (5, 0): [(4, 0), (5, 1), (6, 0)],
            (5, 1): [(5, 0), (5, 2)],
            (5, 2): [(5, 1), (5, 3), (6, 2)],
            (5, 3): [(5, 2), (5, 4)],
            (5, 4): [(4, 4), (5, 3), (6, 4)],

            (6, 0): [(5, 0), (7, 0)],
            (6, 2): [(5, 2), (7, 2)],
            (6, 4): [(5, 4), (7, 4)],

            (7, 0): [(6, 0), (7, 1)],
            (7, 1): [(7, 0), (7, 2)],
            (7, 2): [(7, 1), (7, 3), (6, 2)],
            (7, 3): [(7, 2), (7, 4)],
            (7, 4): [(6, 4), (7, 3), (8, 4)],

            (8, 0): [(7, 0), (9, 0)],
            (8, 4): [(7, 4), (9, 4)],

            (9, 0): [(8, 0), (10, 0)],
            (9, 4): [(8, 4), (10, 4)],

            (10, 0): [(9, 0), (11, 0)],
            (10, 4): [(9, 4), (11, 4)],

            (11, 0): [(10, 0), (11, 1)],
            (11, 1): [(11, 0), (11, 2)],
            (11, 2): [(11, 1), (11, 3)],
            (11, 3): [(11, 2), (11, 4)],
            (11, 4): [(11, 3), (10, 4)]
        }

    def define_roads(self):
        roads = {}
        for i in range(13):
            for j in range(5):
                neighbors = []
                if i != 6:  # 跳过第6行
                    if i > 0 and i - 1 != 6:  # 确保上方的邻居不是第6行
                        neighbors.append((i - 1, j))
                    if i < 12 and i + 1 != 6:  # 确保下方的邻居不是第6行
                        neighbors.append((i + 1, j))
                if j > 0 and i != 6:  # 确保当前行不是第6行
                    neighbors.append((i, j - 1))
                if j < 4 and i != 6:  # 确保当前行不是第6行
                    neighbors.append((i, j + 1))
                roads[(i, j)] = neighbors

        # 添加对角路径
        diagonal_paths = [
            # 左侧对角
            ((2, 1), (1, 0)), ((2, 1), (3, 0)), 
            ((2, 3), (1, 4)), ((2, 3), (3, 4)), 
            ((4, 1), (3, 0)), ((4, 1), (5, 0)), 
            ((4, 3), (3, 4)), ((4, 3), (5, 4)), 
            ((8, 1), (7, 0)), ((8, 1), (9, 0)), 
            ((8, 3), (7, 4)), ((8, 3), (9, 4)), 
            ((10, 1), (9, 0)), ((10, 1), (11, 0)), 
            ((10, 3), (9, 4)), ((10, 3), (11, 4)),

            # 右侧对角
            ((2, 1), (1, 2)), ((2, 1), (3, 2)), 
            ((2, 3), (1, 2)), ((2, 3), (3, 2)), 
            ((4, 1), (3, 2)), ((4, 1), (5, 2)), 
            ((4, 3), (3, 2)), ((4, 3), (5, 2)), 
            ((8, 1), (7, 2)), ((8, 1), (9, 2)), 
            ((8, 3), (7, 2)), ((8, 3), (9, 2)), 
            ((10, 1), (9, 2)), ((10, 1), (11, 2)), 
            ((10, 3), (9, 2)), ((10, 3), (11, 2))
        ]
        for start, end in diagonal_paths:
            roads[start].append(end)
            roads[end].append(start)

        return roads
    
    def define_highways(self):
        highways = []
        for i in range(1, 13):
            highways.append((i, 2))
        return highways

    def get_camps(self):
        return [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]
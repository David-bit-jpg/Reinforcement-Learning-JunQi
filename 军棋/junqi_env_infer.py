import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from reward_model_infer import RewardModel
from pieces import Piece

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
    
    def __init__(self, initial_board=None):
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_rows, self.board_cols, len(self.piece_types) + 1), dtype=np.float32)
        self.move_history = []
        self.battle_history = []
        self.turn = 0
        self.red_commander_dead = False
        self.blue_commander_dead = False

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
        self.red_commander_dead = False
        self.blue_commander_dead = False
        self.generate_random_histories()
        return self.get_state()
    
    def generate_random_histories(self):
        def move_and_battle(piece, piece_list, opponent_positions):
            pos = piece.get_position()
            valid_moves = self.get_valid_moves(piece)
            opponent_positions_set = {pos for _, pos in opponent_positions}

            battle_moves = [move for move in valid_moves if move in opponent_positions_set]
            if battle_moves:
                new_pos = random.choice(battle_moves)
            else:
                new_pos = random.choice(valid_moves) if valid_moves else None

            if new_pos:
                self.move_history.append((pos, piece, new_pos))
                piece.set_position(new_pos)
                return new_pos
            return None

        def select_piece_with_battle(piece_positions, opponent_positions_set):
            battle_pieces = [piece for piece, pos in piece_positions if pos in opponent_positions_set]
            if battle_pieces:
                return random.choice(battle_pieces)
            else:
                return random.choice(piece_positions)[0] if piece_positions else None

        piece_positions_red = [(piece, piece.get_position()) for piece in self.red_pieces if piece.get_position()]
        piece_positions_blue = [(piece, piece.get_position()) for piece in self.blue_pieces if piece.get_position()]

        for _ in range(1000):
            opponent_positions_blue_set = {pos for _, pos in piece_positions_blue}
            opponent_positions_red_set = {pos for _, pos in piece_positions_red}

            if piece_positions_red:
                piece = select_piece_with_battle(piece_positions_red, opponent_positions_blue_set)
                if piece:
                    new_pos = move_and_battle(piece, self.red_pieces, piece_positions_blue)
                    if new_pos:
                        for blue_piece, blue_pos in piece_positions_blue:
                            if new_pos == blue_pos:
                                result = self.resolve_battle(piece, blue_piece)
                                if result == 'win':
                                    self.battle_history.append((piece.get_position(), blue_piece.get_position(), result))
                                elif result == 'draw':
                                    self.battle_history.append((piece.get_position(), blue_piece.get_position(), 'draw'))
                                elif result == 'lose':
                                    self.battle_history.append((blue_piece.get_position(), piece.get_position(), 'win'))
                                piece_positions_red = [(p, p.get_position()) for p in self.red_pieces if p.get_position()]
                                piece_positions_blue = [(p, p.get_position()) for p in self.blue_pieces if p.get_position()]
                                break

            if piece_positions_blue:
                piece = select_piece_with_battle(piece_positions_blue, opponent_positions_red_set)
                if piece:
                    new_pos = move_and_battle(piece, self.blue_pieces, piece_positions_red)
                    if new_pos:
                        for red_piece, red_pos in piece_positions_red:
                            if new_pos == red_pos:
                                result = self.resolve_battle(piece, red_piece)
                                if result == 'win':
                                    self.battle_history.append((piece.get_position(), red_piece.get_position(), result))
                                elif result == 'draw':
                                    self.battle_history.append((piece.get_position(), red_piece.get_position(), 'draw'))
                                elif result == 'lose':
                                    self.battle_history.append((red_piece.get_position(), piece.get_position(), 'win'))
                                piece_positions_red = [(p, p.get_position()) for p in self.red_pieces if p.get_position()]
                                piece_positions_blue = [(p, p.get_position()) for p in self.blue_pieces if p.get_position()]
                                break

    def piece_name_to_value(self, piece_name):
        value_map = {
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
        return value_map.get(piece_name, 0)
    
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
            if defender.get_name() == '司令':
                if defender.get_color() == 'red':
                    self.red_commander_dead = True
                else:
                    self.blue_commander_dead = True
            defender.set_position(None)
            return 'win'
        if piece_value > target_value:
            if attacker.get_name() == '司令':
                if attacker.get_color() == 'red':
                    self.red_commander_dead = True
                else:
                    self.blue_commander_dead = True
            attacker.set_position(None)
            return 'lose'
        if piece_value == target_value:
            if attacker.get_name() == '司令':
                if attacker.get_color() == 'red':
                    self.red_commander_dead = True
                else:
                    self.blue_commander_dead = True
            if defender.get_name() == '司令':
                if defender.get_color() == 'red':
                    self.red_commander_dead = True
                else:
                    self.blue_commander_dead = True
            attacker.set_position(None)
            defender.set_position(None)
            return 'draw'

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

    def get_state(self):
        state = np.zeros((self.board_rows, self.board_cols, len(self.piece_types) + 1))
        for piece in self.red_pieces + self.blue_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    piece_index = self.piece_types.index(piece.get_name())
                    state[x, y, piece_index] = 1
        if self.red_commander_dead:
            state[:, :, -1] = 1  # 红方司令被消灭
        if self.blue_commander_dead:
            state[:, :, -1] = 2  # 蓝方司令被消灭

        return {
            'board_state': state,
            'move_history': self.move_history,
            'battle_history': self.battle_history,
            'agent_color': 'red' if self.turn % 2 == 0 else 'blue',
            'opponent_commander_dead': self.blue_commander_dead if self.turn % 2 == 0 else self.red_commander_dead  # 添加这个字段
        }


    def step(self, action):
        inferred_pieces = self.decode_action(action)  # 将整数动作解码为推理结果
        reward = self.calculate_reward(inferred_pieces)
        done = self.is_terminal_state()
        info = {}
        self.turn += 1  # 更新回合数

        if self.red_commander_dead:
            for piece in self.blue_pieces:
                if piece.get_name() == '军旗':
                    piece.set_revealed(True)
        if self.blue_commander_dead:
            for piece in self.red_pieces:
                if piece.get_name() == '军旗':
                    piece.set_revealed(True)

        return self.get_state(), reward, done, info

    def calculate_reward(self, inferred_pieces):
        actual_pieces = self.get_actual_pieces()
        reward = 0

        piece_weights = {
            '司令': 5,
            '军长': 4,
            '炸弹': 3,
            '地雷': 3,
            '师长': 2,
            '旅长': 2,
            '团长': 1,
            '营长': 1,
            '连长': 1,
            '排长': 1,
            '工兵': 1,
            '军旗': 0
        }

        def get_nearby_positions(pos):
            x, y = pos
            nearby_positions = [
                (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)
            ]
            return [(nx, ny) for nx, ny in nearby_positions if 0 <= nx < self.board_rows and 0 <= ny < self.board_cols]

        for pos, inferred_type in inferred_pieces.items():
            if pos in actual_pieces:
                actual_type = actual_pieces[pos]
                if inferred_type == actual_type:
                    reward += piece_weights.get(actual_type, 1)
                elif inferred_type in piece_weights and actual_type in piece_weights:
                    reward += piece_weights[actual_type] - piece_weights[inferred_type]

            nearby_positions = get_nearby_positions(pos)
            for nearby_pos in nearby_positions:
                if nearby_pos in actual_pieces and inferred_type == actual_pieces[nearby_pos]:
                    reward += piece_weights.get(actual_pieces[nearby_pos], 1) / 2

        return reward

    def get_actual_pieces(self):
        actual_pieces = {}
        for piece in self.blue_pieces:
            pos = piece.get_position()
            if pos:
                actual_pieces[pos] = piece.get_name()
        return actual_pieces

    def is_terminal_state(self):
        max_turns = 10
        if self.turn >= max_turns:
            return True
        return False

    def create_pieces(self, initial_board):
        red_pieces = initial_board[0] if initial_board else self.generate_random_pieces('red')
        blue_pieces = initial_board[1] if initial_board else self.generate_random_pieces('blue')
        for piece in red_pieces:
            position = piece.get_position()
            self.occupied_positions_red.add(position)
        for piece in blue_pieces:
            position = piece.get_position()
            self.occupied_positions_blue.add(position)
        return red_pieces, blue_pieces

    def generate_random_pieces(self, color):
        pieces = []
        positions = set()
        for piece_name in self.piece_types:
            while True:
                position = (random.randint(0, self.board_rows - 1), random.randint(0, self.board_cols - 1))
                if position not in positions:
                    pieces.append(Piece(piece_name, position, color))
                    positions.add(position)
                    break
        return pieces

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
        total_positions = self.board_rows * self.board_cols
        piece_index = action // total_positions
        position_index = action % total_positions
        piece_index = piece_index % len(self.piece_types)
        piece_type = self.piece_types[piece_index]
        x = position_index // self.board_cols
        y = position_index % self.board_cols
        return {(x, y): piece_type}

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
                if i != 6:
                    if i > 0 and i - 1 != 6:
                        neighbors.append((i - 1, j))
                    if i < 12 and i + 1 != 6:
                        neighbors.append((i + 1, j))
                if j > 0 and i != 6:
                    neighbors.append((i, j - 1))
                if j < 4 and i != 6:
                    neighbors.append((i, j + 1))
                roads[(i, j)] = neighbors

        diagonal_paths = [
            ((2, 1), (1, 0)), ((2, 1), (3, 0)), 
            ((2, 3), (1, 4)), ((2, 3), (3, 4)), 
            ((4, 1), (3, 0)), ((4, 1), (5, 0)), 
            ((4, 3), (3, 4)), ((4, 3), (5, 4)), 
            ((8, 1), (7, 0)), ((8, 1), (9, 0)), 
            ((8, 3), (7, 4)), ((8, 3), (9, 4)), 
            ((10, 1), (9, 0)), ((10, 1), (11, 0)), 
            ((10, 3), (9, 4)), ((10, 3), (11, 4)),

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

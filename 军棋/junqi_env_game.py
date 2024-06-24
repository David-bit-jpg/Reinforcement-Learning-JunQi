import logging
import gym
from gym import spaces
import numpy as np
import random
from pieces import Piece
from reward_model_game import RewardModel
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = fm.FontProperties(fname=font_path)


class JunQiEnvGame(gym.Env):
    
    metadata = {'render.modes': ['human']}
    def __init__(self, initial_board):
        super(JunQiEnvGame, self).__init__()
        self.board_rows = 13
        self.board_cols = 5
        self.railways = self.define_railways()
        self.roads = self.define_roads()
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.red_pieces, self.blue_pieces = self.create_pieces(initial_board)
        self.reward_model = RewardModel(self.red_pieces, self.blue_pieces)
        self.action_space = spaces.Discrete(len(self.red_pieces) * self.board_rows * self.board_cols)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_rows, self.board_cols, 3), dtype=np.float32)
        self.reset()

        # Initialize dictionaries for inference
        self.red_killed = {}
        self.blue_killed = {}
        self.red_died = {}
        self.blue_died = {}

    def reset(self):
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.state = 'play'
        self.current_player = 'red'
        self.turn = 0
        self.red_infer = {piece: set() for piece in self.blue_pieces}
        self.blue_infer = {piece: set() for piece in self.red_pieces}

        # Reset dictionaries for inference
        self.red_killed.clear()
        self.blue_killed.clear()
        self.red_died.clear()
        self.blue_died.clear()

        return self.get_state()

    def get_state(self):
        state = np.zeros((self.board_rows, self.board_cols, 3))
        for piece in self.red_pieces:
            if piece.position:
                x, y = piece.position
                state[x, y, 0] = 1

        for piece in self.blue_pieces:
            if piece.position:
                x, y = piece.position
                state[x, y, 1] = 1

        # 将推理信息整合到状态表示中
        infer_state = np.zeros((self.board_rows, self.board_cols, 2))
        for piece, positions in self.red_infer.items():
            for pos in positions:
                x, y = pos
                infer_state[x, y, 0] = 1

        for piece, positions in self.blue_infer.items():
            for pos in positions:
                x, y = pos
                infer_state[x, y, 1] = 1

        return np.concatenate((state, infer_state), axis=2)

    def step(self, action):
        piece, target_position = self.decode_action(action)
        current_player_color = self.current_player
        valid, outcome = self.make_move(current_player_color, piece, target_position)

        if not valid:
            reward = self.reward_model.get_reward('setup', (piece, target_position), 'invalid_move')
            done = False
            info = {"invalid_move": True}
        else:
            reward = self.reward_model.get_reward('setup', (piece, target_position), outcome)
            done = self.state in ['red_wins', 'blue_wins']
            info = {"invalid_move": False}

        next_state = self.get_state()
        
        return next_state, reward, done, info

    def make_move(self, player_color, piece, target_position):
        if player_color != piece.get_color():
            logging.warning(f"Invalid action: it's {player_color}'s turn, but the piece is {piece.get_color()}.")
            return False, 'invalid_move'

        valid_moves = self.get_valid_moves(piece)
        if target_position not in valid_moves:
            logging.warning(f"Invalid move: the target position {target_position} is not valid for the piece {piece}.")
            return False, 'invalid_move'
        
        if not self.move_piece(piece, target_position):
            logging.warning(f"Move failed: unable to move the piece {piece} to {target_position}.")
            return False, 'invalid_move'

        self.switch_player()
        self.check_winner()
        
        if self.state == 'red_wins' or self.state == 'blue_wins':
            return True, 'win_game'

        return True, 'valid_move'

    def battle(self, attacker, defender):
        rank_order = {
            '军旗': 0, '地雷': 1, '工兵': 2, '排长': 3,
            '连长': 4, '营长': 5, '团长': 6, '旅长': 7,
            '师长': 8, '军长': 9, '司令': 10, '炸弹': 11
        }
        attacker_rank = rank_order[attacker.get_name()]
        defender_rank = rank_order[defender.get_name()]

        if attacker_rank == defender_rank:
            attacker.set_position(None)
            defender.set_position(None)
            return 'draw_battle'
        elif attacker_rank == 11 or defender_rank == 11:
            attacker.set_position(None)
            defender.set_position(None)
            return 'draw_battle'
        elif defender.get_name() == '地雷':
            if attacker.get_name() == '工兵':
                defender.set_position(None)
                self.update_positions(attacker, defender.get_position())
                self.update_killed_dictionaries(attacker, defender)
                return 'win_battle'
            else:
                attacker.set_position(None)
                self.update_killed_dictionaries(defender, attacker)
                return 'lose_battle'
        elif attacker_rank > defender_rank:
            defender.set_position(None)
            self.update_positions(attacker, defender.get_position())
            self.update_killed_dictionaries(attacker, defender)
            return 'win_battle'
        else:
            attacker.set_position(None)
            self.update_killed_dictionaries(defender, attacker)
            return 'lose_battle'

    def update_killed_dictionaries(self, killer, victim):
        if killer.get_color() == 'red':
            self.red_killed[victim] = killer
            self.blue_died[victim] = killer
        else:
            self.blue_killed[victim] = killer
            self.red_died[victim] = killer

    def define_railways(self):
        return {
            # 从 (1, 0) 到 (1, 4) 的横向铁路
            (1, 0): [(1, 1)],
            (1, 1): [(1, 0), (1, 2)],
            (1, 2): [(1, 1), (1, 3)],
            (1, 3): [(1, 2), (1, 4)],
            (1, 4): [(1, 3)],

            # 从 (1, 0) 到 (11, 0) 的纵向铁路
            (1, 0): [(2, 0)],
            (2, 0): [(1, 0), (3, 0)],
            (3, 0): [(2, 0), (4, 0)],
            (4, 0): [(3, 0), (5, 0)],
            (5, 0): [(4, 0), (6, 0)],
            (6, 0): [(5, 0), (7, 0)],
            (7, 0): [(6, 0), (8, 0)],
            (8, 0): [(7, 0), (9, 0)],
            (9, 0): [(8, 0), (10, 0)],
            (10, 0): [(9, 0), (11, 0)],
            (11, 0): [(10, 0)],

            (11, 0): [(11, 1)],
            (11, 1): [(11, 0), (11, 2)],
            (11, 2): [(11, 1), (11, 3)],
            (11, 3): [(11, 2), (11, 4)],
            (11, 4): [(11, 3)],

            (1, 4): [(2, 4)],
            (2, 4): [(1, 4), (3, 4)],
            (3, 4): [(2, 4), (4, 4)],
            (4, 4): [(3, 4), (5, 4)],
            (5, 4): [(4, 4), (6, 4)],
            (6, 4): [(5, 4), (7, 4)],
            (7, 4): [(6, 4), (8, 4)],
            (8, 4): [(7, 4), (9, 4)],
            (9, 4): [(8, 4), (10, 4)],
            (10, 4): [(9, 4), (11, 4)],
            (11, 4): [(10, 4)],

            (5, 0): [(5, 1)],
            (5, 1): [(5, 0), (5, 2)],
            (5, 2): [(5, 1), (5, 3), (6, 2)],
            (5, 3): [(5, 2), (5, 4)],
            (5, 4): [(5, 3)],

            (7, 0): [(7, 1)],
            (7, 1): [(7, 0), (7, 2)],
            (7, 2): [(7, 1), (7, 3), (6, 2)],
            (7, 3): [(7, 2), (7, 4)],
            (7, 4): [(7, 3)],

            (6, 2): [(5, 2), (7, 2)]
        }

    def define_roads(self):
        roads = {}
        for i in range(13):
            for j in range(5):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1, j))
                if i < 12:
                    neighbors.append((i + 1, j))
                if j > 0:
                    neighbors.append((i, j - 1))
                if j < 4:
                    neighbors.append((i, j + 1))
                roads[(i, j)] = neighbors

        # 添加对角路径
        diagonal_paths = [
            # 左侧对角
            # 左侧对角线
            ((2, 1), (1, 0)), ((2, 1), (3, 0)), 
            ((2, 3), (1, 4)), ((2, 3), (3, 4)), 
            ((4, 1), (3, 0)), ((4, 1), (5, 0)), 
            ((4, 3), (3, 4)), ((4, 3), (5, 4)), 
            ((8, 1), (7, 0)), ((8, 1), (9, 0)), 
            ((8, 3), (7, 4)), ((8, 3), (9, 4)), 
            ((10, 1), (9, 0)), ((10, 1), (11, 0)), 
            ((10, 3), (9, 4)), ((10, 3), (11, 4)),

            # 右侧对角线
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

    def get_camps(self):
        return [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]

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

    def get_state(self):
        state = np.zeros((self.board_rows, self.board_cols, 3))
        for piece in self.red_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    state[x, y, 0] = 1

        for piece in self.blue_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    state[x, y, 1] = 1

        return state
    
    def encode_action(self, piece_index, target_position):
        position_index = target_position[0] * self.board_cols + target_position[1]
        return piece_index * (self.board_rows * self.board_cols) + position_index

    def decode_action(self, action):
        piece_index = action // (self.board_rows * self.board_cols)
        position_index = action % (self.board_rows * self.board_cols)
        piece = self.get_piece_by_index(piece_index)
        target_position = (position_index // self.board_cols, position_index % self.board_cols)
        return piece, target_position


    def get_piece_by_index(self, index):
        if index < len(self.red_pieces):
            return self.red_pieces[index]
        else:
            return self.blue_pieces[index - len(self.red_pieces)]

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.board_cols and 0 <= y < self.board_rows

    def is_railway(self, position):
        return position in self.railways

    def can_move(self, piece, target_position):
        if not self.is_valid_position(target_position):
            return False

        if piece.get_name() == '军旗' or piece.get_name() == '地雷':
            return False

        current_position = piece.get_position()
        if current_position is None:
            return False

        if self.is_railway(current_position) and piece.get_name() == '工兵':
            return self.can_move_on_railway(piece, target_position)
        else:
            return self.can_move_on_road(piece, target_position)

    def can_move_on_railway(self, piece, target_position):
        current_position = piece.get_position()
        if target_position in self.railways.get(current_position, []):
            if self.is_clear_path(current_position, target_position, piece):
                return True
        return False

    def can_move_on_road(self, piece, target_position):
        current_position = piece.get_position()
        return target_position in self.roads.get(current_position, [])

    def get_valid_moves(self, piece):
        current_position = piece.get_position()
        valid_moves = []

        if self.is_railway(current_position):
            if piece.get_name() == '工兵':
                for pos in self.railways[current_position]:
                    if self.is_clear_path(current_position, pos, piece):
                        valid_moves.append(pos)
            else:
                for pos in self.railways[current_position]:
                    if self.is_clear_path(current_position, pos, piece):
                        if pos[0] == current_position[0] or pos[1] == current_position[1]:
                            valid_moves.append(pos)
        else:
            for pos in self.roads[current_position]:
                if self.is_valid_position(pos) and self.is_clear_path(current_position, pos, piece):
                    valid_moves.append(pos)

        camps = self.get_camps()
        base_positions = [(12, 1), (12, 3), (0, 1), (0, 3)]
        valid_moves.extend([camp for camp in camps if camp != current_position])
        valid_moves.extend([base for base in base_positions if base != current_position])

        valid_moves = [move for move in valid_moves if not self.is_in_base(move) or piece.get_position() not in base_positions]

        occupied_camps = set(camps) & self.occupied_positions_red | self.occupied_positions_blue
        valid_moves = [move for move in valid_moves if move not in occupied_camps]

        return valid_moves

    def is_clear_path(self, start, end, piece):
        x1, y1 = start
        x2, y2 = end
        occupied_positions = self.occupied_positions_red if piece.get_color() == 'red' else self.occupied_positions_blue
        opponent_positions = self.occupied_positions_blue if piece.get_color() == 'red' else self.occupied_positions_red

        if x1 == x2:
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                if (x1, y) in occupied_positions:
                    return False
                if (x1, y) in opponent_positions:
                    return True  # 如果路径上有对方棋子，返回 True，允许交战
        elif y1 == y2:
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                if (x, y1) in occupied_positions:
                    return False
                if (x, y1) in opponent_positions:
                    return True  # 如果路径上有对方棋子，返回 True，允许交战
        return True

    def move_piece(self, piece, target_position):
        """
        移动棋子到目标位置，执行战斗或更新位置。
        """
        valid_moves = self.get_valid_moves(piece)
        if target_position in valid_moves:
            target_piece = self.get_piece_at_position(target_position)
            if target_piece:
                if target_piece.get_color() != piece.get_color():
                    # 如果目标位置是敌方棋子，进行交战
                    if self.is_in_camp(target_position):
                        return False  # 行营中的棋子不能被攻击
                    self.battle(piece, target_piece)
                else:
                    return False  # 不能移动到己方棋子的位置
            else:
                # 如果目标位置是空的，更新棋子位置
                if not self.is_in_base(target_position) or piece.get_position() in self.get_base_positions():
                    self.update_positions(piece, target_position)
                else:
                    return False  # 不能移动到敌方大本营
            return True
        return False

    def get_base_positions(self):
        """
        获取所有大本营位置
        """
        return [(12, 1), (12, 3), (0, 1), (0, 3)]

    def update_positions(self, piece, target_position):
        """
        更新棋子的位置。
        """
        current_position = piece.get_position()
        if current_position:
            if piece.get_color() == 'red':
                self.occupied_positions_red.remove(current_position)
                self.occupied_positions_red.add(target_position)
            else:
                self.occupied_positions_blue.remove(current_position)
                self.occupied_positions_blue.add(target_position)
        piece.set_position(target_position)

    def get_piece_at_position(self, position):
        """
        获取指定位置的棋子。
        """
        for piece in self.red_pieces + self.blue_pieces:
            if piece.get_position() == position:
                return piece
        return None

    def play_phase(self, action):
        """
        执行玩家的动作，并切换当前玩家。
        """
        piece, target_position = action

        # 确保当前玩家只能选择自己的棋子
        if self.current_player != piece.get_color():
            logging.warning(f"Invalid action: it's {self.current_player}'s turn. Action: {action}")
            return

        # 获取棋子的所有合法移动位置
        valid_moves = self.get_valid_moves(piece)

        if target_position in valid_moves:
            # 尝试移动棋子
            if self.move_piece(piece, target_position):
                self.switch_player()
                self.check_winner()
            else:
                # 如果移动不成功，检查当前玩家是否有其他合法移动，如果没有则当前玩家失败
                if not self.try_other_moves():
                    self.state = 'red_wins' if self.current_player == 'blue' else 'blue_wins'
        else:
            logging.warning(f"Invalid move: the target position {target_position} is not valid for the piece {piece}.")
            # 仍然尝试当前玩家是否有其他合法移动，如果没有则当前玩家失败
            if not self.try_other_moves():
                self.state = 'red_wins' if self.current_player == 'blue' else 'blue_wins'

    def try_other_moves(self):
        """
        尝试当前玩家的其他可行移动，若无可行移动则当前玩家失败。
        """
        pieces = self.red_pieces if self.current_player == 'red' else self.blue_pieces

        for piece in pieces:
            if piece.get_position() is not None:
                valid_moves = self.get_valid_moves(piece)
                for move in valid_moves:
                    if self.move_piece(piece, move):
                        return True
        return False

    def is_in_camp(self, position):
        """
        检查位置是否在行营中。
        """
        camps = self.get_camps()
        return position in camps

    def is_in_base(self, position):
        """
        检查位置是否在大本营中。
        """
        red_bases = [(12, 1), (12, 3)]
        blue_bases = [(0, 1), (0, 3)]
        return position in red_bases or blue_bases

    def render(self, mode='human'):
        """
        渲染棋盘状态，打印当前玩家和棋子位置。
        """
        print(f"当前玩家: {self.current_player}")
        for piece in self.red_pieces + self.blue_pieces:
            print(f"{piece.get_name()} ({piece.get_color()}): {piece.get_position()}")

    def get_action_space_size(self):
        """
        获取动作空间的大小。
        """
        return (len(self.red_pieces) + len(self.blue_pieces)) * (self.board_rows * self.board_cols)

    def switch_player(self):
        """
        切换当前玩家。
        """
        self.current_player = 'blue' if self.current_player == 'red' else 'red'

    def check_winner(self):
        """
        检查游戏是否结束，确定胜者。
        """
        red_flag = self.red_pieces[0]
        blue_flag = self.blue_pieces[0]
        if red_flag.get_position() is None:
            self.state = 'blue_wins'
        elif blue_flag.get_position() is None:
            self.state = 'red_wins'
        elif not self.has_valid_moves('red'):
            self.state = 'blue_wins'
        elif not self.has_valid_moves('blue'):
            self.state = 'red_wins'

    def has_valid_moves(self, player_color):
        """
        检查玩家是否有有效的移动。
        """
        pieces = self.red_pieces if player_color == 'red' else self.blue_pieces
        for piece in pieces:
            if piece.get_position() and self.get_valid_moves(piece):
                return True
        return False
    
    def mark_guess(self, piece, position, guess):
        """
        标记对手棋子的猜测。
        """
        if piece.get_color() == 'red':
            self.red_infer[piece] = guess
        else:
            self.blue_infer[piece] = guess

    def visualize_board(self):
        railways = self.railways
        roads = self.roads
        camps = self.get_camps()
        positions = [(x, y) for x in range(13) for y in range(5) if (x, y) not in [(6, 1), (6, 3)]]

        board_rows, board_cols = 13, 5
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_xticks(np.arange(board_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(board_rows + 1) - 0.5, minor=True)
        ax.tick_params(which="minor", size=0)
        ax.set_xlim(-0.5, board_cols - 0.5)
        ax.set_ylim(-0.5, board_rows - 0.5)
        ax.invert_yaxis()
        ax.set_title("JunQi Board")

        # 绘制道路
        for road in roads:
            for neighbor in roads[road]:
                if neighbor in roads:
                    ax.plot([road[1], neighbor[1]], [road[0], neighbor[0]], 'k-', lw=0.5)

        # 绘制铁路
        for railway in railways:
            for neighbor in railways[railway]:
                if neighbor in railways:
                    ax.plot([railway[1], neighbor[1]], [railway[0], neighbor[0]], 'k-', lw=2)
                    # 绘制黑白相间的线条
                    ax.plot([railway[1], neighbor[1]], [railway[0], neighbor[0]], 'w-', lw=1, zorder=1)

        # 绘制营地
        for camp in camps:
            y, x = camp
            ax.add_patch(plt.Circle((x, y), 0.3, color='green', fill=False, linewidth=2))

        # 绘制棋子可放置的位置
        for pos in positions:
            y, x = pos
            ax.plot(x, y, 'bo')

        plt.show()

    def visualize_full_board(self):
        board_rows, board_cols = 13, 5
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_xticks(np.arange(board_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(board_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_xlim(-0.5, board_cols - 0.5)
        ax.set_ylim(-0.5, board_rows - 0.5)
        ax.invert_yaxis()
        ax.set_title("Full Board Deployment", fontproperties=font_prop)
        
        for piece in self.red_pieces:
            pos = piece.get_position()
            if pos is not None:
                y, x = pos
                ax.text(x, y, piece.get_name(), ha='center', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'), fontproperties=font_prop)

        for piece in self.blue_pieces:
            pos = piece.get_position()
            if pos is not None:
                y, x = pos
                ax.text(x, y, piece.get_name(), ha='center', va='center', fontsize=12, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'), fontproperties=font_prop)

        plt.show()

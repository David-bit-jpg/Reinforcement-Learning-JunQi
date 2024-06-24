import logging
import gym
from gym import spaces
import numpy as np
import random
from pieces import Piece
from reward_model_setup import RewardModel  # 引入奖励模型
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = fm.FontProperties(fname=font_path)
class JunQiEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(JunQiEnv, self).__init__()
        self.board_rows = 13
        self.board_cols = 5
        self.railways = self.define_railways()
        self.roads = self.define_roads()
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.red_pieces, self.blue_pieces = self.create_pieces()
        self.reward_model = RewardModel(self.red_pieces, self.blue_pieces)
        self.action_space = spaces.Discrete(len(self.red_pieces) * self.board_rows * self.board_cols)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_rows, self.board_cols, 3), dtype=np.float32)
        self.reset()

    def define_railways(self):
        railways = {
            (2, 1): [(0, 1), (4, 1), (2, 2)],
            (2, 2): [(2, 1), (2, 3)],
            (2, 3): [(2, 2), (0, 3), (4, 3)],
            (8, 1): [(6, 1), (10, 1), (8, 2)],
            (8, 2): [(8, 1), (8, 3)],
            (8, 3): [(8, 2), (6, 3), (10, 3)],
        }
        return railways

    def define_roads(self):
        roads = {}
        for i in range(5):
            for j in range(13):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1, j))
                if i < 4:
                    neighbors.append((i + 1, j))
                if j > 0:
                    neighbors.append((i, j - 1))
                if j < 12:
                    neighbors.append((i, j + 1))
                roads[(i, j)] = neighbors
        return roads

    def reset(self):
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.state = 'setup'
        self.current_player = 'red'
        self.current_setup_index = 0
        self.turn = 0
        self.red_infer = {piece: set() for piece in self.blue_pieces}
        self.blue_infer = {piece: set() for piece in self.red_pieces}
        self.red_pieces, self.blue_pieces = self.create_pieces()
        self.done = False  # 重置 done 标志
        return self.get_state()

    
    def update_valid_positions(self, color):
        if color == 'red':
            pieces = self.red_pieces
            occupied_positions = self.occupied_positions_red
        else:
            pieces = self.blue_pieces
            occupied_positions = self.occupied_positions_blue
        
        for piece in pieces:
            if piece.get_position() is None:
                piece.valid_positions = [pos for pos in piece.valid_positions if pos not in occupied_positions]

    def get_state(self):
        state = np.zeros((self.board_rows, self.board_cols, 3))
        for piece in self.red_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    state[x, y, 0] = 1  # 红方棋子标记

        for piece in self.blue_pieces:
            if piece.position:
                x, y = piece.position
                if 0 <= x < self.board_rows and 0 <= y < self.board_cols:
                    state[x, y, 1] = 1  # 蓝方棋子标记

        return state

    def update_action_space(self):
    # 根据当前状态和选择的棋子动态计算动作空间大小
        if self.current_player == 'red':
            num_red_pieces = len(self.red_pieces)
            self.action_space = spaces.Discrete(num_red_pieces * self.board_rows * self.board_cols)
        else:
            num_blue_pieces = len(self.blue_pieces)
            self.action_space = spaces.Discrete(num_blue_pieces * self.board_rows * self.board_cols)

    def step(self, action):
        self.update_action_space()
        piece, target_position = self.decode_action(action)
        outcome = self.setup_phase((piece, target_position))
        
        if piece is None or target_position is None:
            reward = -100  # 无效动作的惩罚
            logging.warning(f"Invalid action: piece or target_position is None. Action: {action}")
        else:
            reward = self.reward_model.get_reward(self.state, (piece, target_position), outcome)

        done = self.state == 'play'  # 布局阶段结束后设置 done 为 True
        info = {}
        # 如果进入了 play 状态，展示双方的棋盘布局
        # if self.state == 'play':
        #     self.visualize_full_board()
        return self.get_state(), reward, done, info


    def decode_action(self, action):
        piece_index = action // (self.board_rows * self.board_cols)
        position_index = action % (self.board_rows * self.board_cols)
        piece = self.get_piece_by_index(piece_index)
        target_position = (position_index // self.board_cols, position_index % self.board_cols)
        return piece, target_position

    def encode_action(self, piece, target_position):
        piece_index = (self.red_pieces + self.blue_pieces).index(piece)
        position_index = target_position[0] * self.board_cols + target_position[1]
        return piece_index * (self.board_rows * self.board_cols) + position_index

    def get_piece_by_index(self, index):
        if index < len(self.red_pieces):
            return self.red_pieces[index]
        else:
            return self.blue_pieces[index - len(self.red_pieces)]

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.board_cols and 0 <= y < self.board_rows
    
    def update_positions(self, piece, target_position):
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
        for piece in self.red_pieces + self.blue_pieces:
            if piece.get_position() == position:
                return piece
        return None

    def setup_phase(self, action):
        piece, position = action
        outcome = 'invalid'
        valid_positions = piece.get_valid_positions()
        occupied_positions = self.occupied_positions_red if self.current_player == 'red' else self.occupied_positions_blue

        if position in valid_positions and position not in occupied_positions:
            piece.set_position(position)
            occupied_positions.add(position)
            outcome = 'valid'
            self.update_valid_positions(piece.get_color())
        else:
            outcome = 'invalid'

        self.current_setup_index += 1

        # 切换玩家的逻辑
        if self.current_player == 'red':
            if self.current_setup_index >= len(self.red_pieces):
                self.current_player = 'blue'
                self.current_setup_index = 0
        else:
            if self.current_setup_index >= len(self.blue_pieces):
                self.current_player = 'red'
                self.current_setup_index = 0

        # 检查是否所有棋子都已经布置完毕
        if all(piece.get_position() is not None for piece in self.red_pieces + self.blue_pieces):
            # 切换玩家之前输出当前阵型
            self.state = 'play'
            self.done = True  # 设置 done 标志

        return outcome

    def render(self, mode='human'):
        print(f"当前玩家: {self.current_player}")
        for piece in self.red_pieces + self.blue_pieces:
            print(f"{piece.get_name()} ({piece.get_color()}): {piece.get_position()}")

    def get_action_space_size(self):
        return (len(self.red_pieces) + len(self.blue_pieces)) * (self.board_rows * self.board_cols)

    def switch_player(self):
        self.current_player = 'blue' if self.current_player == 'red' else 'red'

    def has_valid_moves(self, player_color):
        pieces = self.red_pieces if player_color == 'red' else self.blue_pieces
        for piece in pieces:
            if piece.get_position() and self.can_move(piece, piece.get_position()):
                return True
        return False
    
    def create_pieces(self):
        red_valid_positions = [(i, j) for i in range(7, 13) for j in range(5)]
        blue_valid_positions = [(i, j) for i in range(0, 6) for j in range(5)]
        camps = [(2, 1), (2, 3), (4, 1), (4, 3), (8, 1), (8, 3), (10, 1), (10, 3), (3, 2), (9, 2)]
        red_valid_positions = [pos for pos in red_valid_positions if pos not in camps]
        blue_valid_positions = [pos for pos in blue_valid_positions if pos not in camps]
        red_pieces = [
            Piece('军旗', 0, 'red', [(12, 1), (12, 3)]),
            Piece('地雷', 10, 'red', [(i, j) for i in range(11, 13) for j in range(5)]),
            Piece('地雷', 10, 'red', [(i, j) for i in range(11, 13) for j in range(5)]),
            Piece('地雷', 10, 'red', [(i, j) for i in range(11, 13) for j in range(5)]),
            Piece('炸弹', 11, 'red', [(i, j) for i in range(8, 13) for j in range(5)]),
            Piece('炸弹', 11, 'red', [(i, j) for i in range(8, 13) for j in range(5)]),
            Piece('司令', 1, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('军长', 2, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('师长', 3, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('师长', 3, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('旅长', 4, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('旅长', 4, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('团长', 5, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('团长', 5, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('营长', 6, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('营长', 6, 'red', [pos for pos in red_valid_positions if pos not in [(12, 1), (12, 3)]]),
            Piece('连长', 7, 'red', red_valid_positions),
            Piece('连长', 7, 'red', red_valid_positions),
            Piece('连长', 7, 'red', red_valid_positions),
            Piece('排长', 8, 'red', red_valid_positions),
            Piece('排长', 8, 'red', red_valid_positions),
            Piece('排长', 8, 'red', red_valid_positions),
            Piece('工兵', 9, 'red', [pos for pos in red_valid_positions if (pos not in [(12, 1), (12, 3)] and pos[0] not in [5, 7])]),
            Piece('工兵', 9, 'red', [pos for pos in red_valid_positions if (pos not in [(12, 1), (12, 3)] and pos[0] not in [5, 7])]),
            Piece('工兵', 9, 'red', [pos for pos in red_valid_positions if (pos not in [(12, 1), (12, 3)] and pos[0] not in [5, 7])]),
        ]

        blue_pieces = [
            Piece('军旗', 0, 'blue', [(0, 1), (0, 3)]),
            Piece('地雷', 10, 'blue', [(i, j) for i in range(0, 2) for j in range(5)]),
            Piece('地雷', 10, 'blue', [(i, j) for i in range(0, 2) for j in range(5)]),
            Piece('地雷', 10, 'blue', [(i, j) for i in range(0, 2) for j in range(5)]),
            Piece('炸弹', 11, 'blue', [(i, j) for i in range(0, 5) for j in range(5)]),
            Piece('炸弹', 11, 'blue', [(i, j) for i in range(0, 5) for j in range(5)]),
            Piece('司令', 1, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('军长', 2, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('师长', 3, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('师长', 3, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('旅长', 4, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('旅长', 4, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('团长', 5, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('团长', 5, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('营长', 6, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('营长', 6, 'blue', [pos for pos in blue_valid_positions if pos not in [(0, 1), (0, 3)]]),
            Piece('连长', 7, 'blue', blue_valid_positions),
            Piece('连长', 7, 'blue', blue_valid_positions),
            Piece('连长', 7, 'blue', blue_valid_positions),
            Piece('排长', 8, 'blue', blue_valid_positions),
            Piece('排长', 8, 'blue', blue_valid_positions),
            Piece('排长', 8, 'blue', blue_valid_positions),
            Piece('工兵', 9, 'blue', [pos for pos in blue_valid_positions if (pos not in [(0, 1), (0, 3)] and pos[0] not in [5, 7])]),
            Piece('工兵', 9, 'blue', [pos for pos in blue_valid_positions if (pos not in [(0, 1), (0, 3)] and pos[0] not in [5, 7])]),
            Piece('工兵', 9, 'blue', [pos for pos in blue_valid_positions if (pos not in [(0, 1), (0, 3)] and pos[0] not in [5, 7])]),
        ]

        return red_pieces, blue_pieces

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

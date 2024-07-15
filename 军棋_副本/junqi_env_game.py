import logging
import gym
from gym import spaces
import numpy as np
import random
from pieces import Piece
from reward_model_game import RewardModel
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import traceback
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import copy
import torch
from dqn_agent_infer import DQNAgent as DQNAgentInfer
from dqn_agent_infer import DoubleDQN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class JunQiEnvGame(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, initial_board, input_size, hidden_size, output_size, lr=0.001):
        super(JunQiEnvGame, self).__init__()
        self.board_rows = 13
        self.board_cols = 5
        self.railways = self.define_railways()
        self.roads = self.define_roads()
        self.highways = self.define_highways()
        self.occupied_positions_red = set()
        self.occupied_positions_blue = set()
        self.red_pieces, self.blue_pieces = self.create_pieces(initial_board)
        self.reward_model = RewardModel(self.red_pieces, self.blue_pieces, self)
        self.action_space = spaces.Discrete(len(self.red_pieces) * self.board_rows * self.board_cols)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_rows, self.board_cols, 3), dtype=np.float32)
        self.pi = None
        self.pi_reg = None
        self.initial_board = initial_board
        self.piece_types = list(piece_encoding.keys())
        self.device = device
        self.model = self.load_inference_model('/code/军棋/models/infer_model.pth')

        self.move_history = []
        self.battle_history = []
        self.red_commander_dead = False
        self.blue_commander_dead = False

        self.reset()

    def load_inference_model(self, model_path):
        # 定义推理模型的架构和初始化
        model = DoubleDQN(board_rows=13, board_cols=5, piece_types=list(piece_encoding.keys()))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 定义目标模型
        target_model = DoubleDQN(board_rows=13, board_cols=5, piece_types=list(piece_encoding.keys()))
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        
        # 创建 DQNAgent 实例
        action_size = len(self.red_pieces) * self.board_rows * self.board_cols
        agent = DQNAgentInfer(model=model, target_model=target_model, action_size=action_size, initial_board=self.initial_board)
        return agent

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
        return self.get_state()

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

    def set_state(self, state):
        self.state = state
            
    def step(self, action, pi, pi_reg, current_player_color, weights):
        player_color, piece, target_position = self.decode_action(action)
        if piece.get_color() != current_player_color:
            logging.warning(f"Invalid action: In turn {self.turn}, it's {current_player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return self.get_state(), -1, True, {}  # 返回一个负奖励并结束该回合
        initial_state = self.get_state()
        self.make_move(current_player_color, piece, target_position)

        # 其他逻辑
        next_state = self.get_state()
        reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'valid_move', pi, pi_reg, player_color, weights)
        if current_player_color == 'red' and self.check_winner(current_player_color) == 'red_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color, weights)
        elif current_player_color == 'blue' and self.check_winner(current_player_color) == 'blue_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color, weights)
        done = True  # 每次移动一个棋子后即结束
        info = {"invalid_move": reward < 0}
        return next_state, reward, done, info

    def step_with_inference(self, action, pi, pi_reg, current_player_color, weights):
        player_color, piece, target_position = self.decode_action(action)
        if piece.get_color() != current_player_color:
            logging.warning(f"Invalid action: In turn {self.turn}, it's {current_player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return self.get_state(), -1, True, {}  # 返回一个负奖励并结束该回合
        initial_state = self.get_state()

        # 使用推理模型来推测对手棋子的状态
        inferred_pieces = self.infer_opponent_pieces(current_player_color)

        # 创建环境副本并设置推测状态
        env_copy = copy.deepcopy(self)
        for (x, y), piece_type in inferred_pieces.items():
            if piece_type != '':  # 确保推测的棋子类型有效
                rank = piece_encoding[piece_type]
                color = 'blue' if current_player_color == 'red' else 'red'
                valid_positions = self.railways.get((x, y), []) + self.roads.get((x, y), [])
                inferred_piece = Piece(piece_type, rank, color, valid_positions)
                inferred_piece.set_position((x, y))
                env_copy.update_positions(inferred_piece, (x, y), inferred_piece.get_color())

        env_copy.make_move_cpy(current_player_color, piece, target_position)

        # 其他逻辑
        next_state = env_copy.get_state()
        reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'valid_move', pi, pi_reg, player_color, weights)
        if current_player_color == 'red' and self.check_winner(current_player_color) == 'red_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color, weights)
        elif current_player_color == 'blue' and self.check_winner(current_player_color) == 'blue_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color, weights)
        done = True  # 每次移动一个棋子后即结束
        info = {"invalid_move": reward < 0}
        return next_state, reward, done, info

    def infer_opponent_pieces(self, current_player_color):
        board_state = self.get_full_state()['board_state']
        move_history = self.move_history
        battle_history = self.battle_history
        opponent_commander_dead = self.red_commander_dead if current_player_color == 'blue' else self.blue_commander_dead

        state = {
            'board_state': board_state,
            'move_history': move_history,
            'battle_history': battle_history,
            'agent_color': current_player_color,
            'opponent_commander_dead': opponent_commander_dead
        }

        # 确保 board_state 是 4 维的
        if board_state.ndim == 3:
            board_state = np.expand_dims(board_state, axis=0)

        # 转换为tensor输入并调整形状
        input_tensor = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2)
        output = self.model.model(input_tensor).detach().numpy()  # 调用模型的forward方法

        inferred_pieces = {}
        opponent_color = 'blue' if current_player_color == 'red' else 'red'

        # 假设输出是每个棋子类别的概率分布
        piece_types_prob = output[0]  # 形状为 (12,)
        for x in range(self.board_rows):
            for y in range(self.board_cols):
                if board_state[0, x, y, 1] == 1:  # 检查这个位置是否有对方棋子
                    piece_type_index = np.argmax(piece_types_prob)
                    piece_type = self.piece_types[piece_type_index]
                    inferred_pieces[(x, y)] = piece_type
        return inferred_pieces

    def get_full_state(self):
        state = np.zeros((self.board_rows, self.board_cols, 4))
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
        if self.red_commander_dead:
            state[:, :, 2] = 1  # 红方司令被消灭
        if self.blue_commander_dead:
            state[:, :, 2] = 2  # 蓝方司令被消灭
        return {
            'board_state': state,
            'move_history': self.move_history,
            'battle_history': self.battle_history,
            'agent_color': 'red' if self.turn % 2 == 0 else 'blue'  # 假设轮流行动，红方先手
        }

    def set_state(self, state):
        self.state = state
            
    def make_move(self, player_color, piece, target_position):
        if player_color != piece.get_color():
            logging.warning(f"Invalid action: In turn {self.turn}, it's {player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return

        valid_moves = self.get_valid_moves(piece)
        if target_position not in valid_moves:
            logging.warning(f"Invalid move: In turn {self.turn}, it's {player_color}'s turn. the target position {target_position} is not valid for the piece {piece}.")
            return

        target_piece = self.get_piece_at_position(target_position)
        if target_piece and target_piece.get_color() == piece.get_color():
            logging.warning(f"Invalid move: target position {target_position} is occupied by a friendly piece.")
            return

        # 记录移动历史
        self.move_history.append((piece, piece.get_position(), target_position))

        # 移动棋子到目标位置
        self.update_positions(piece, target_position, player_color)

        # 检查是否有战斗发生
        target_piece = self.get_piece_at_position(target_position)
        if target_piece:
            if target_piece.get_color() != piece.get_color():
                if self.is_in_camp(target_position):
                    logging.warning(f"Cannot attack a piece in a camp at {target_position}.")
                    return  # 不能攻击在行营中的棋子
                self.battle(piece, target_piece)

    def make_move_cpy(self, player_color, piece, target_position):
        if player_color != piece.get_color():
            # logging.warning(f"Invalid action: In turn {self.turn}, it's {player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return

        valid_moves = self.get_valid_moves(piece)
        if target_position not in valid_moves:
            # logging.warning(f"Invalid move: In turn {self.turn}, it's {player_color}'s turn. the target position {target_position} is not valid for the piece {piece}.")
            return

        target_piece = self.get_piece_at_position(target_position)
        if target_piece and target_piece.get_color() == piece.get_color():
            # logging.warning(f"Invalid move: target position {target_position} is occupied by a friendly piece.")
            return

        # 记录移动历史
        self.move_history.append((piece, piece.get_position(), target_position))

        # 移动棋子到目标位置
        self.update_positions(piece, target_position, player_color)

        # 检查是否有战斗发生
        target_piece = self.get_piece_at_position(target_position)
        if target_piece:
            if target_piece.get_color() != piece.get_color():
                if self.is_in_camp(target_position):
                    # logging.warning(f"Cannot attack a piece in a camp at {target_position}.")
                    return  # 不能攻击在行营中的棋子
                self.battle(piece, target_piece)

    def update_positions(self, piece, target_position, player_color):
        current_position = piece.get_position()
        if current_position == target_position:
            return False

        # 移动棋子到目标位置之前，检查是否有敌方棋子
        target_piece = self.get_piece_at_position(target_position)
        if target_piece and target_piece.get_color() != piece.get_color():
            # 执行战斗
            battle_result = self.battle(piece, target_piece)
            self.battle_history.append((piece.get_color(), piece.get_name(), piece.get_position(), target_piece.get_color(), target_piece.get_name(), target_piece.get_position(), battle_result))  # 记录战斗历史
            if battle_result == 'win_battle':
                target_piece.set_position(None)  # 移除被击败的棋子
            elif battle_result == 'lose_battle':
                piece.set_position(None)  # 移除己方棋子
                return False  # 不进行位置更新
            else:
                piece.set_position(None)
                target_piece.set_position(None)
                return False  # 不进行位置更新

        if current_position:
            if piece.get_color() == 'red':
                if current_position in self.occupied_positions_red:
                    self.occupied_positions_red.remove(current_position)
                self.occupied_positions_red.add(target_position)
            else:
                if current_position in self.occupied_positions_blue:
                    self.occupied_positions_blue.remove(current_position)
                self.occupied_positions_blue.add(target_position)

        piece.set_position(target_position)
        return True
    
    def update_positions_cpy(self, piece, target_position, player_color):
        current_position = piece.get_position()
        if current_position == target_position:
            return False

        # 移动棋子到目标位置之前，检查是否有敌方棋子
        target_piece = self.get_piece_at_position(target_position)
        if target_piece and target_piece.get_color() != piece.get_color():
            # 执行战斗
            battle_result = self.battle(piece, target_piece)
            self.battle_history.append((piece.get_color(), piece.get_name(), piece.get_position(), target_piece.get_color(), target_piece.get_name(), target_piece.get_position(), battle_result))  # 记录战斗历史
            if battle_result == 'win_battle':
                target_piece.set_position(None)  # 移除被击败的棋子
            elif battle_result == 'lose_battle':
                piece.set_position(None)  # 移除己方棋子
                return False  # 不进行位置更新
            else:
                piece.set_position(None)
                target_piece.set_position(None)
                return False  # 不进行位置更新

        if current_position:
            if piece.get_color() == 'red':
                if current_position in self.occupied_positions_red:
                    self.occupied_positions_red.remove(current_position)
                self.occupied_positions_red.add(target_position)
            else:
                if current_position in self.occupied_positions_blue:
                    self.occupied_positions_blue.remove(current_position)
                self.occupied_positions_blue.add(target_position)

        piece.set_position(target_position)
        return True

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
                        visited.add(neighbor)  # 友方棋子阻挡，标记为已访问，防止重复检查
                    else:
                        queue.append(neighbor)

        return list(valid_moves)

    def get_railway_moves(self, current_position, piece):
        valid_moves = []
        color = piece.get_color()

        # 横向检查
        self.expand_in_direction(current_position, color, valid_moves, is_horizontal=True)
        # 纵向检查
        self.expand_in_direction(current_position, color, valid_moves, is_horizontal=False)

        return valid_moves

    def expand_in_direction(self, current_position, color, valid_moves, is_horizontal):
        if is_horizontal:
            # 向左
            y = current_position[1] - 1
            while y >= 0:
                next_pos = (current_position[0], y)
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break  # 友方棋子阻挡
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                y -= 1

            # 向右
            y = current_position[1] + 1
            while y < self.board_cols:
                next_pos = (current_position[0], y)
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break  # 友方棋子阻挡
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                y += 1
        else:
            # 向上
            x = current_position[0] - 1
            while x >= 0:
                next_pos = (x, current_position[1])
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break  # 友方棋子阻挡
                    valid_moves.append(next_pos)
                    if self.is_occupied(next_pos):
                        break
                else:
                    break
                x -= 1

            # 向下
            x = current_position[0] + 1
            while x < self.board_rows:
                next_pos = (x, current_position[1])
                if self.is_railway(next_pos):
                    if self.is_friendly_occupied(next_pos, color):
                        break  # 友方棋子阻挡
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

    def is_occupied(self, position):
        return position in self.occupied_positions_red or position in self.occupied_positions_blue

    def is_friendly_occupied(self, position, color):
        return position in (self.occupied_positions_red if color == 'red' else self.occupied_positions_blue)

    def get_railway_moves_horizontal(self, current_position, piece):
        valid_moves = []
        # 向左检查
        y = current_position[1] - 1
        while y >= 0:
            next_pos = (current_position[0], y)
            if self.is_railway(next_pos) and not self.is_friendly_occupied(next_pos, piece.get_color()):
                valid_moves.append(next_pos)
                if self.is_occupied(next_pos):
                    break
            else:
                break
            y -= 1

        # 向右检查
        y = current_position[1] + 1
        while y < self.board_cols:
            next_pos = (current_position[0], y)
            if self.is_railway(next_pos) and not self.is_friendly_occupied(next_pos, piece.get_color()):
                valid_moves.append(next_pos)
                if self.is_occupied(next_pos):
                    break
            else:
                break
            y += 1

        return valid_moves

    def get_railway_moves_vertical(self, current_position, piece):
        valid_moves = []
        # 向上检查
        x = current_position[0] - 1
        while x >= 0:
            next_pos = (x, current_position[1])
            if self.is_railway(next_pos) and not self.is_friendly_occupied(next_pos, piece.get_color()):
                valid_moves.append(next_pos)
                if self.is_occupied(next_pos):
                    break
            else:
                break
            x -= 1

        # 向下检查
        x = current_position[0] + 1
        while x < self.board_rows:
            next_pos = (x, current_position[1])
            if self.is_railway(next_pos) and not self.is_friendly_occupied(next_pos, piece.get_color()):
                valid_moves.append(next_pos)
                if self.is_occupied(next_pos):
                    break
            else:
                break
            x += 1

        return valid_moves

    def move_piece(self, piece, target_position):
        """
        移动棋子到目标位置，执行战斗或更新位置。
        """
        current_position = piece.get_position()
        if not current_position:
            print(f"Piece {piece.get_name()} has no current position.")
            return False

        valid_moves = self.get_valid_moves(piece)
        print(f"Valid moves for piece {piece.get_name()} at {current_position}: {valid_moves}")

        if target_position in valid_moves:
            print(f"Moving piece {piece.get_name()} to {target_position}")
            # 更新棋子位置
            self.update_positions(piece, target_position)

            # 检查目标位置是否有敌方棋子，并进行战斗
            target_piece = self.get_piece_at_position(target_position)
            if target_piece and target_piece.get_color() != piece.get_color():
                if self.is_in_camp(target_position):
                    print(f"Cannot attack piece in camp at {target_position}")
                    return False  # 行营中的棋子不能被攻击
                self.battle(piece, target_piece)

            return True
        print(f"Target position {target_position} is not a valid move for piece {piece.get_name()}")
        return False

    def get_base_positions(self):
        """
        获取所有大本营位置
        """
        return [(12, 1), (12, 3), (0, 1), (0, 3)]

    def get_piece_at_position(self, position):
        """
        获取指定位置的棋子。
        """
        for piece in self.red_pieces + self.blue_pieces:
            if piece.get_position() == position:
                return piece
        return None

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

    def simulate_action(self, state, action):
        """
        模拟一个走法，返回新的状态
        """
        piece, target_position = action
        new_state = state.copy()
        # 移动棋子
        current_position = piece.get_position()
        if current_position:
            new_state[current_position[0], current_position[1], 0] = 0  # 移除当前棋子位置
        new_state[target_position[0], target_position[1], 0] = 1  # 更新新位置
        # 处理战斗情况
        target_piece = self.get_piece_at_position(target_position)
        if target_piece:
            if piece.get_color() != target_piece.get_color():
                outcome = self.battle(piece, target_piece)
                if outcome == 'win_battle':
                    new_state[target_position[0], target_position[1], 1] = 0  # 移除被打败的棋子
                elif outcome == 'lose_battle':
                    new_state[target_position[0], target_position[1], 0] = 0  # 移除己方棋子
        return new_state

    def encode_action(self, player_color, piece_index, target_position):
        position_index = target_position[0] * self.board_cols + target_position[1]
        total_positions = self.board_rows * self.board_cols
        if player_color == 'red':
            return piece_index * total_positions + position_index
        else:
            return (piece_index + len(self.red_pieces)) * total_positions + position_index

    def decode_action(self, action):
        total_positions = self.board_rows * self.board_cols
        piece_index = action // total_positions
        position_index = action % total_positions
        if piece_index < len(self.red_pieces):
            player_color = 'red'
            piece_index = piece_index
        else:
            player_color = 'blue'
            piece_index = piece_index - len(self.red_pieces)
        piece = self.get_piece_by_index(player_color, piece_index)
        target_position = (position_index // self.board_cols, position_index % self.board_cols)
        return player_color, piece, target_position

    def check_winner(self, player_color):
        """
        检查是否有玩家获胜。
        """
        red_flag_position = self.get_flag_position('red')
        blue_flag_position = self.get_flag_position('blue')

        if red_flag_position is None:
            return 'blue_wins'
        elif blue_flag_position is None:
            return 'red_wins'
        else:
            # 如果当前玩家没有合法的移动，则当前玩家失败
            if not self.get_valid_actions(player_color):
                if player_color == 'red':
                    return 'blue_wins'
                else:
                    return 'red_wins'
        return 'No'

    def get_piece_by_index(self, player_color, index):
        if player_color == 'red':
            piece = self.red_pieces[index]
        else:
            piece = self.blue_pieces[index]
        return piece

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.board_cols and 0 <= y < self.board_rows

    def is_railway(self, position):
        return position in self.railways
    
    def get_pieces(self, color):
        if color == 'red':
            return self.red_pieces
        elif color == 'blue':
            return self.blue_pieces
        else:
            raise ValueError("Invalid color")

    def is_terminal_state(self, state):
        """
        判断当前状态是否为终局状态。
        """
        red_flag = self.get_flag_position('red')
        blue_flag = self.get_flag_position('blue')
        # 如果任何一方的军旗被捕获，游戏结束
        if red_flag is None or blue_flag is None:
            return True
        # 如果任何一方没有可行的移动，游戏结束
        if not self.has_valid_moves('red') or not self.has_valid_moves('blue'):
            return True
        return False

    def get_flag_position(self, color):
        """
        获取军旗的位置。
        """
        pieces = self.red_pieces if color == 'red' else self.blue_pieces
        for piece in pieces:
            if piece.get_name() == '军旗':
                return piece.get_position()
        return None

    def can_move(self, piece, target_position):
        if not self.is_valid_position(target_position):
            return False
        if piece.get_name() == '军旗'or piece.get_name() == '地雷':
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
    
    def get_piece_index(self, piece):
        if piece in self.red_pieces:
            return self.red_pieces.index(piece)
        elif piece in self.blue_pieces:
            return self.blue_pieces.index(piece) + len(self.red_pieces)
        else:
            return -1

    def can_move_on_road(self, piece, target_position):
        current_position = piece.get_position()
        return target_position in self.roads.get(current_position, [])

    def render(self, mode='human'):
        """
        渲染棋盘状态，打印当前玩家和棋子位置。
        """
        print(f"当前玩家: {self.current_player}")
        for piece in self.red_pieces + self.blue_pieces:
            print(f"{piece.get_name()} ({piece.get_color()}): {piece.get_position()}")

    def piece_name_to_value(self, piece_name):
        """
        Converts a piece name to its corresponding value.
        """
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
    
    def battle(self, piece, target_piece):
        """
        执行战斗逻辑。
        """
        piece_value = self.piece_name_to_value(piece.get_name())
        target_value = self.piece_name_to_value(target_piece.get_name())

        if piece.get_name() == '炸弹' or target_piece.get_name() == '炸弹':
            if piece.get_name() == '炸弹' and target_piece.get_name() != '炸弹':
                piece.set_position(None)
                target_piece.set_position(None)
                return 'draw_battle'
            elif piece.get_name() != '炸弹' and target_piece.get_name() == '炸弹':
                piece.set_position(None)
                target_piece.set_position(None)
                return 'draw_battle'
            elif piece.get_name() == '炸弹' and target_piece.get_name() == '炸弹':
                piece.set_position(None)
                target_piece.set_position(None)
                return 'draw_battle'
        if piece.get_name() == '工兵' and target_piece.get_name() == '地雷':
            target_piece.set_position(None)
            return 'win_battle'
        if piece_value < target_value:
            target_piece.set_position(None)
            if target_piece.get_name() == '司令':
                # 一旦司令被消灭，暴露军旗的位置
                try:
                    flag_piece = next(p for p in (self.red_pieces if target_piece.get_color() == 'red' else self.blue_pieces) if p.get_name() == '军旗')
                    flag_piece.set_position(target_piece.get_position())
                except StopIteration:
                    print(f"Warning: No flag piece found for color {target_piece.get_color()}")
            return 'win_battle'
        if piece_value > target_value:
            piece.set_position(None)
            return 'lose_battle'
        if piece_value == target_value:
            piece.set_position(None)
            target_piece.set_position(None)
            return 'draw_battle'

    def get_move_history(self):
        return self.move_history

    def get_battle_history(self):
        return self.battle_history
    
    def visualize_full_board(self, last_move=None, battle_info=None):
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

        if last_move:
            piece, start_pos, end_pos = last_move
            sy, sx = start_pos
            ey, ex = end_pos
            ax.arrow(sx, sy, ex - sx, ey - sy, head_width=0.3, head_length=0.3, fc='green', ec='green')

        if battle_info:
            attacker, defender, result = battle_info
            if result == 'win':
                attacker_pos = attacker.get_position()
                defender_pos = defender.get_position()
                if attacker_pos and defender_pos:
                    ay, ax_pos = attacker_pos
                    dy, dx = defender_pos
                    ax.plot([ax_pos, dx], [ay, dy], 'r-', linewidth=2)
                    ax.text(dx, dy, f"{defender.get_name()} defeated by {attacker.get_name()}", ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='yellow', edgecolor='red', boxstyle='round,pad=0.3'), fontproperties=font_prop)
            elif result == 'lose':
                attacker_pos = attacker.get_position()
                defender_pos = defender.get_position()
                if attacker_pos and defender_pos:
                    ay, ax_pos = attacker_pos
                    dy, dx = defender_pos
                    ax.plot([ax_pos, dx], [ay, dy], 'r-', linewidth=2)
                    ax.text(ax_pos, ay, f"{attacker.get_name()} defeated by {defender.get_name()}", ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='yellow', edgecolor='blue', boxstyle='round,pad=0.3'), fontproperties=font_prop)
            elif result == 'draw':
                attacker_pos = attacker.get_position()
                defender_pos = defender.get_position()
                if attacker_pos and defender_pos:
                    ay, ax_pos = attacker_pos
                    dy, dx = defender_pos
                    ax.plot([ax_pos, dx], [ay, dy], 'r-', linewidth=2)
                    ax.text((ax_pos + dx) / 2, (ay + dy) / 2, "Draw", ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='yellow', edgecolor='purple', boxstyle='round,pad=0.3'), fontproperties=font_prop)

        plt.show()
        
    def visualize_inferred_board(self, player_color):
        inferred_pieces = self.infer_opponent_pieces(player_color)
        board_rows, board_cols = 13, 5
        
        # 创建一个大的图形，包含两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
        
        # 设置子图1：实际棋盘布局
        ax1.set_xticks(np.arange(board_cols + 1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(board_rows + 1) - 0.5, minor=True)
        ax1.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax1.tick_params(which="minor", size=0)
        ax1.set_xlim(-0.5, board_cols - 0.5)
        ax1.set_ylim(-0.5, board_rows - 0.5)
        ax1.invert_yaxis()
        ax1.set_title("Actual Full Board Deployment", fontproperties=font_prop)

        # 显示实际棋盘布局
        for piece in self.red_pieces:
            pos = piece.get_position()
            if pos is not None:
                y, x = pos
                ax1.text(x, y, piece.get_name(), ha='center', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'), fontproperties=font_prop)

        for piece in self.blue_pieces:
            pos = piece.get_position()
            if pos is not None:
                y, x = pos
                ax1.text(x, y, piece.get_name(), ha='center', va='center', fontsize=12, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'), fontproperties=font_prop)

        # 设置子图2：推理后的棋盘布局
        ax2.set_xticks(np.arange(board_cols + 1) - 0.5, minor=True)
        ax2.set_yticks(np.arange(board_rows + 1) - 0.5, minor=True)
        ax2.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax2.tick_params(which="minor", size=0)
        ax2.set_xlim(-0.5, board_cols - 0.5)
        ax2.set_ylim(-0.5, board_rows - 0.5)
        ax2.invert_yaxis()
        ax2.set_title(f"Inferred Board Deployment by {player_color}", fontproperties=font_prop)

        # 显示推理出的对方棋子
        for (pos, piece_type) in inferred_pieces.items():
            x, y = pos
            ax2.text(y, x, piece_type, ha='center', va='center', fontsize=12, color='grey', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3'), fontproperties=font_prop)
        
        # 显示自己的棋子
        for piece in self.red_pieces if player_color == 'red' else self.blue_pieces:
            pos = piece.get_position()
            if pos is not None:
                y, x = pos
                color = 'red' if player_color == 'red' else 'blue'
                ax2.text(x, y, piece.get_name(), ha='center', va='center', fontsize=12, color=color, bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'), fontproperties=font_prop)

        plt.show()
        
    def get_action_space_size(self):
        return (len(self.red_pieces) + len(self.blue_pieces)) * (self.board_rows * self.board_cols)

    def get_state_size(self):
        return self.observation_space.shape[0] * self.observation_space.shape[1] * self.observation_space.shape[2]
    
    def get_valid_actions(self, player_color):
        valid_actions = []
        current_pieces = None
        if player_color == 'red':
            current_pieces = self.red_pieces
        else:
            current_pieces = self.blue_pieces

        base_positions = self.get_base_positions()  # 获取所有大本营位置

        for piece_index, piece in enumerate(current_pieces):
            position = piece.get_position()
            if position and position not in base_positions:  # 检查棋子不在基地里
                valid_moves = self.get_valid_moves(piece)
                for move in valid_moves:
                    action = self.encode_action(player_color, piece_index, move)
                    valid_actions.append(action)
        return valid_actions

    def prepare_input(self, board_state, move_history, battle_history, opponent_commander_dead, pos):
        x, y = pos
        input_data = np.zeros((4, self.model.board_rows, self.model.board_cols))
        
        if board_state.shape[2] > 0:
            max_channel = np.argmax(board_state[x, y])
            max_channel = min(max_channel, board_state.shape[2] - 1)
            board_slice = board_state[:, :, max_channel]
            
            padded_slice = np.zeros((self.model.board_rows, self.model.board_cols))
            rows = min(board_slice.shape[0], self.model.board_rows)
            cols = min(board_slice.shape[1], self.model.board_cols)
            
            for i in range(rows):
                for j in range(cols):
                    if isinstance(board_slice[i, j], (list, np.ndarray)):
                        padded_slice[i, j] = board_slice[i, j][0]
                    else:
                        padded_slice[i, j] = board_slice[i, j]
            
            input_data[0, :, :] = padded_slice
        
        # 处理 move_history
        move_count = min(len(move_history), 10)
        for idx, move in enumerate(move_history[-move_count:]):
            piece, start_pos, end_pos = move
            color_idx = 1 if piece.get_color() == 'red' else 2
            input_data[color_idx, start_pos[0], start_pos[1]] = idx + 1
            input_data[color_idx, end_pos[0], end_pos[1]] = idx + 1
        
        # 处理 battle_history
        for battle in battle_history:
            attacker_color, attacker_name, attacker_pos, defender_color, defender_name, defender_pos, result = battle
            color_idx = 3 if attacker_color == 'red' else 2
            input_data[color_idx, attacker_pos[0], attacker_pos[1]] = piece_encoding[attacker_name]
            input_data[color_idx, defender_pos[0], defender_pos[1]] = piece_encoding[defender_name]
        
        if opponent_commander_dead:
            input_data[3, :, :] = 1  # 标记对方司令已被消灭
        
        return input_data

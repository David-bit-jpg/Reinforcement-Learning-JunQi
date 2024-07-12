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
import numpy as np

class InferenceModel:
    def __init__(self, board_rows, board_cols, piece_types, env):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.piece_types = piece_types
        self.env = env
        self.belief = np.ones((board_rows, board_cols, len(piece_types))) / len(piece_types)
        self.move_history = []
        self.battle_history = []
        self.stationary_pieces = {}
        self.flag_positions = {}
        self.device = device

    def update_belief(self, move_history, battle_history):
        self.move_history.extend(move_history)
        self.battle_history.extend(battle_history)

        self.update_belief_from_move_history()
        self.update_belief_from_battle_history()
        self.update_belief_from_stationary_pieces()
        self.update_belief_from_flag_positions()
        self.update_belief_from_mine_positions()

    def apply_initial_guess(self):
        opponent_color = 'blue' if self.env.red_pieces[0].get_color() == 'red' else 'red'
        opponent_base_line = 0 if opponent_color == 'blue' else 12
        opponent_second_base_line = 1 if opponent_color == 'blue' else 11

        for piece in self.env.get_pieces(opponent_color):
            x, y = piece.get_position()
            if x is not None and y is not None:
                if x == opponent_base_line:
                    if y in [1, 3]:
                        self.belief[x, y, self.piece_types.index('军旗')] = 0.5
                    self.belief[x, y, self.piece_types.index('地雷')] = 0.3
                    self.belief[x, y, self.piece_types.index('司令')] = 0.1
                    self.belief[x, y, self.piece_types.index('军长')] = 0.1
                elif x == opponent_second_base_line:
                    self.belief[x, y, self.piece_types.index('地雷')] = 0.3
                elif self.env.is_railway((x, y)):
                    self.belief[x, y, self.piece_types.index('工兵')] = 0.5
                else:
                    self.belief[x, y, self.piece_types.index('炸弹')] = 0.2
                    self.belief[x, y, self.piece_types.index('师长')] = 0.2
                    self.belief[x, y, self.piece_types.index('旅长')] = 0.2
                    self.belief[x, y, self.piece_types.index('团长')] = 0.2

                self.belief[x, y, :] /= np.sum(self.belief[x, y, :])

    def update_belief_from_move_history(self):
        for move in self.move_history:
            piece, start_pos, end_pos = move
            x, y = end_pos
            piece_type = piece.get_name()

            if piece_type == '工兵':
                self.belief[x, y, :] *= 1.5
            else:
                self.belief[x, y, :] *= 1.1
            self.belief[x, y, :] /= np.sum(self.belief[x, y, :])

            if piece_type in ['司令', '军长']:
                for nx, ny in self.get_neighboring_positions(end_pos):
                    if self.is_valid_position((nx, ny)):
                        self.belief[nx, ny, self.piece_types.index('炸弹')] *= 1.5
                        self.belief[nx, ny, :] /= np.sum(self.belief[nx, ny, :])

    def update_belief_from_battle_history(self):
        for battle in self.battle_history:
            attacker, defender, result = battle
            attacker_pos = attacker.get_position()
            defender_pos = defender.get_position()

            if result == 'win':
                if attacker_pos:
                    x, y = attacker_pos
                    self.belief[x, y, :] *= 1.2
                    self.belief[x, y, :] /= np.sum(self.belief[x, y, :])
                if defender_pos:
                    x, y = defender_pos
                    self.belief[x, y, :] *= 0.8
                    self.belief[x, y, :] /= np.sum(self.belief[x, y, :])
            elif result == 'lose':
                if attacker_pos:
                    x, y = attacker_pos
                    self.belief[x, y, :] *= 0.8
                    self.belief[x, y, :] /= np.sum(self.belief[x, y, :])
                if defender_pos:
                    x, y = defender_pos
                    self.belief[x, y, :] *= 1.2
                    self.belief[x, y, :] /= np.sum(self.belief[x, y, :])

            if defender.get_name() == '司令' and result == 'win':
                flag_position = self.get_flag_position(defender.get_color())
                if flag_position:
                    self.flag_positions[defender.get_color()] = flag_position

    def update_belief_from_stationary_pieces(self):
        for pos, data in self.stationary_pieces.items():
            if data['moves'] == 0 and data['wins'] == 0:
                self.belief[pos[0], pos[1], self.piece_types.index('地雷')] *= 1.5
                self.belief[pos[0], pos[1], :] /= np.sum(self.belief[pos[0], pos[1], :])
            elif data['wins'] > 2:
                high_value_pieces = ['司令', '军长', '师长']
                for piece in high_value_pieces:
                    self.belief[pos[0], pos[1], self.piece_types.index(piece)] *= 1.2
                self.belief[pos[0], pos[1], :] /= np.sum(self.belief[pos[0], pos[1], :])

    def update_belief_from_flag_positions(self):
        for color, position in self.flag_positions.items():
            x, y = position
            self.belief[x, y, :] = 0
            self.belief[x, y, self.piece_types.index('军旗')] = 1
            self.belief[x, y, :] /= np.sum(self.belief[x, y, :])

    def update_belief_from_mine_positions(self):
        opponent_color = 'blue' if self.env.red_pieces[0].get_color() == 'red' else 'red'
        opponent_second_base_line = 1 if opponent_color == 'blue' else 11

        for x in range(self.board_rows):
            for y in range(self.board_cols):
                piece = self.env.get_piece_at_position((x, y))
                if piece and piece.get_color() == opponent_color and piece.get_position() == (x, y):
                    if x in [0, opponent_second_base_line]:
                        self.belief[x, y, self.piece_types.index('地雷')] *= 1.5
                    elif piece in self.stationary_pieces and self.stationary_pieces[piece] > 3:
                        self.belief[x, y, self.piece_types.index('地雷')] *= 1.5
                    self.belief[x, y, :] /= np.sum(self.belief[x, y, :])

    def infer_opponent_pieces(self):
        inferred_pieces = {}
        opponent_color = 'blue' if self.env.red_pieces[0].get_color() == 'red' else 'red'
        for x in range(self.board_rows):
            for y in range(self.board_cols):
                piece = self.env.get_piece_at_position((x, y))
                if piece and piece.get_color() == opponent_color:
                    piece_type_index = np.argmax(self.belief[x, y, :])
                    piece_type = self.piece_types[piece_type_index]
                    inferred_pieces[(x, y)] = piece_type
        return inferred_pieces

    def get_neighboring_positions(self, position):
        x, y = position
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.board_rows - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.board_cols - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.board_rows and 0 <= y < self.board_cols

    def get_flag_position(self, color):
        return self.env.get_flag_position(color)

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
        
        self.piece_types = list(piece_encoding.keys())
        self.inference_model = InferenceModel(self.board_rows, self.board_cols, self.piece_types, self)  # 传递 env 对象

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
        self.inference_model = InferenceModel(self.board_rows, self.board_cols, self.piece_types, self)  # 传递 env 对象
        self.move_history = []
        self.battle_history = []
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

    def get_state_size(self):
        state = self.get_state()
        return state.size 
    
    def set_state(self, state):
        self.state = state

    def step(self, action, pi, pi_reg, current_player_color, weights):
        player_color, piece, target_position = self.decode_action(action)
        if piece.get_color() != current_player_color:
            logging.warning(f"Invalid action: In turn {self.turn}, it's {current_player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return self.get_state(), -1, True, {}  # 返回一个负奖励并结束该回合
        initial_state = self.get_state()
        self.make_move(current_player_color, piece, target_position)

        # 更新信念状态
        observations = self.get_observation()
        move_history = self.get_move_history()
        battle_history = self.get_battle_history()
        self.inference_model.update_belief(move_history, battle_history)

        # 其他逻辑
        next_state = self.get_state()
        reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'valid_move', pi, pi_reg, player_color,weights)
        if current_player_color == 'red' and self.check_winner(current_player_color) == 'red_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color,weights)
        elif current_player_color == 'blue' and self.check_winner(current_player_color) == 'blue_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color,weights)
        done = True  # 每次移动一个棋子后即结束
        info = {"invalid_move": reward < 0}
        return next_state, reward, done, info

    def step_with_inference(self, action, pi, pi_reg, current_player_color,weights):
        player_color, piece, target_position = self.decode_action(action)
        if piece.get_color() != current_player_color:
            logging.warning(f"Invalid action: In turn {self.turn}, it's {current_player_color}'s turn, but the piece is {piece.get_color()} {piece.get_name()}.")
            return self.get_state(), -1, True, {}  # 返回一个负奖励并结束该回合
        initial_state = self.get_state()

        # 使用推理模型来推测对手棋子的状态
        inferred_pieces = self.inference_model.infer_opponent_pieces()

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
        reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'valid_move', pi, pi_reg, player_color,weights)
        if current_player_color == 'red' and self.check_winner(current_player_color) == 'red_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color,weights)
        elif current_player_color == 'blue' and self.check_winner(current_player_color) == 'blue_wins':
            reward = self.reward_model.get_reward(initial_state, (piece, target_position), 'win_game', pi, pi_reg, player_color,weights)
        done = True  # 每次移动一个棋子后即结束
        info = {"invalid_move": reward < 0}
        return next_state, reward, done, info
    
    def get_observation(self):
        observations = {}
        for piece in self.red_pieces + self.blue_pieces:
            pos = piece.get_position()
            if pos:
                observations[pos] = piece.get_name()
        return observations

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

    def get_pieces(self, player_color):
        return self.red_pieces if player_color == 'red' else self.blue_pieces

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
            self.battle_history.append((piece, target_piece, battle_result))  # 记录战斗历史
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
            self.battle_history.append((piece, target_piece, battle_result))  # 记录战斗历史
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

    def check_winner(self,player_color):
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
        inferred_pieces = self.inference_model.infer_opponent_pieces()
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
        
    def get_pieces(self, color):
        if color == 'red':
            return self.red_pieces
        elif color == 'blue':
            return self.blue_pieces
        else:
            raise ValueError("Invalid color")

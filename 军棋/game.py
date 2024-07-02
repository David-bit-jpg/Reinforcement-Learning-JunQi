import random
from junqi_env_game import JunQiEnvGame  # 确保你的环境文件名为 junqi_env_game.py
from pieces import Piece

def get_random_positions(env, num_positions):
    all_positions = [(x, y) for x in range(env.board_rows) for y in range(env.board_cols) if (x, y) not in [(6, 1), (6, 3)]]
    return random.sample(all_positions, num_positions)

# 示例初始棋盘配置，包含红方和蓝方棋子
initial_board = (
    [
        Piece("司令", 10, "red", []), Piece("军长", 9, "red", []), Piece("师长", 8, "red", []),
        Piece("旅长", 7, "red", []), Piece("团长", 6, "red", []), Piece("营长", 5, "red", []),
        Piece("连长", 4, "red", []), Piece("排长", 3, "red", []), Piece("工兵", 1, "red", []),
        Piece("炸弹", 2, "red", []), Piece("地雷", 1, "red", []), Piece("军旗", 0, "red", [])
    ],  # Red pieces
    [
        Piece("司令", 10, "blue", []), Piece("军长", 9, "blue", []), Piece("师长", 8, "blue", []),
        Piece("旅长", 7, "blue", []), Piece("团长", 6, "blue", []), Piece("营长", 5, "blue", []),
        Piece("连长", 4, "blue", []), Piece("排长", 3, "blue", []), Piece("工兵", 1, "blue", []),
        Piece("炸弹", 2, "blue", []), Piece("地雷", 1, "blue", []), Piece("军旗", 0, "blue", [])
    ]  # Blue pieces
)

env = JunQiEnvGame(initial_board, 10, 10, 10)

# 随机分配红方和蓝方棋子位置，确保不重复
red_positions = get_random_positions(env, len(initial_board[0]))
blue_positions = get_random_positions(env, len(initial_board[1]))
for i, piece in enumerate(initial_board[0]):
    piece.set_position(red_positions[i])
for i, piece in enumerate(initial_board[1]):
    piece.set_position(blue_positions[i])

# 更新 occupied positions
env.reset()

# 固定选择红方和蓝方的棋子
red_pieces = initial_board[0]
blue_pieces = initial_board[1]

def simulate_battle(env, red_piece, blue_piece):
    target_position = blue_piece.get_position()

    # 打印战斗前状态
    print(f"战斗前状态：")
    print(f"红方棋子：{red_piece.get_name()} 在 {red_piece.get_position()}")
    print(f"蓝方棋子：{blue_piece.get_name()} 在 {blue_piece.get_position()}")

    # 进行战斗并更新位置
    initial_state = env.get_state()
    env.make_move('red', red_piece, target_position)

    # 检查战斗结果
    target_piece = env.get_piece_at_position(target_position)
    if target_piece and target_piece.get_color() != red_piece.get_color():
        battle_result = env.battle(red_piece, target_piece)
        if battle_result == 'win_battle':
            print(f"红方的 {red_piece.get_name()} 吃掉了 蓝方的 {target_piece.get_name()}")
        elif battle_result == 'lose_battle':
            print(f"红方的 {red_piece.get_name()} 被 蓝方的 {target_piece.get_name()} 吃掉了")
        else:
            print(f"红方的 {red_piece.get_name()} 和 蓝方的 {target_piece.get_name()} 打成平手")
    else:
        print(f"没有战斗发生，红方的 {red_piece.get_name()} 移动到了 {target_position}")

    # 打印战斗后状态
    print(f"战斗后状态：")
    print(f"红方棋子：{red_piece.get_name()} 在 {red_piece.get_position()}")
    if target_piece:
        print(f"蓝方棋子：{target_piece.get_name()} 在 {target_piece.get_position()}")

# 随机选择一个红方和蓝方的棋子进行战斗模拟
red_piece = random.choice(red_pieces)
blue_piece = random.choice(blue_pieces)
simulate_battle(env, red_piece, blue_piece)

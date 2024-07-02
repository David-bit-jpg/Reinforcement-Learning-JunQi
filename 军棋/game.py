import matplotlib.pyplot as plt
import numpy as np
import random
from junqi_env_game import JunQiEnvGame  # 确保你的环境文件名为 junqi_env_game.py
from pieces import Piece
import matplotlib.font_manager as fm

font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = fm.FontProperties(fname=font_path)

def draw_piece_moves(ax, piece, env, color):
    """
    绘制棋子的合法移动位置
    """
    valid_moves = env.get_valid_moves(piece)
    for move in valid_moves:
        ax.plot(move[1], move[0], f'{color[0]}o')

def draw_piece_position(ax, piece, color):
    """
    绘制棋子的位置
    """
    current_pos = piece.get_position()
    ax.text(current_pos[1], current_pos[0], piece.get_name(), color=color, fontsize=12, ha='center', va='center', fontproperties=font_prop)

def visualize_board(env, pieces):
    railways = env.railways
    roads = env.roads
    camps = env.get_camps()
    base_positions = env.get_base_positions()

    fig, ax = plt.subplots(figsize=(8, 12))

    # 绘制棋盘格子
    ax.set_xticks(np.arange(env.board_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(env.board_rows + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", size=0)
    ax.set_xlim(-0.5, env.board_cols - 0.5)
    ax.set_ylim(-0.5, env.board_rows - 0.5)
    ax.invert_yaxis()
    ax.set_title("JunQi Board", fontproperties=font_prop)

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

    # 绘制大本营
    for base in base_positions:
        y, x = base
        ax.add_patch(plt.Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='black'))

    return fig, ax

def get_random_positions(env, num_positions):
    all_positions = [(x, y) for x in range(env.board_rows) for y in range(env.board_cols) if (x, y) not in [(6, 1), (6, 3)]]
    return random.sample(all_positions, num_positions)

# 示例初始棋盘配置，仅包含红方棋子
initial_board = (
    [
        Piece("司令", 10, "red", []), Piece("军长", 9, "red", []), Piece("师长", 8, "red", []),
        Piece("旅长", 7, "red", []), Piece("团长", 6, "red", []), Piece("营长", 5, "red", []),
        Piece("连长", 4, "red", []), Piece("排长", 3, "red", []), Piece("工兵", 1, "red", [])
    ],  # Red pieces
    []  # Blue pieces为空
)

env = JunQiEnvGame(initial_board, 10, 10, 10)

# 随机分配红方棋子位置，确保不重复
red_positions = get_random_positions(env, len(initial_board[0]))
for i, piece in enumerate(initial_board[0]):
    piece.set_position(red_positions[i])

# 更新 occupied positions
env.reset()

# 固定选择红方的棋子
red_pieces = initial_board[0]

# 可视化棋盘和棋子的合法移动位置
fig, ax = visualize_board(env, red_pieces)

# 展示所有红方棋子的位置
for piece in red_pieces:
    draw_piece_position(ax, piece, 'red')

# 展示红方工兵的合法移动位置
worker_piece = next(p for p in red_pieces if p.get_name() == '工兵')
draw_piece_moves(ax, worker_piece, env, 'red')

plt.show()

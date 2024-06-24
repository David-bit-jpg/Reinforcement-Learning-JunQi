import matplotlib.pyplot as plt
import numpy as np
from junqi_env_game import JunQiEnvGame  # 确保你的环境文件名为 junqi_env_game.py

def visualize_board(env):
    railways = env.railways
    roads = env.roads
    camps = env.get_camps()
    base_positions = env.get_base_positions()
    positions = [(x, y) for x in range(env.board_rows) for y in range(env.board_cols) if (x, y) not in [(6, 1), (6, 3)]]

    fig, ax = plt.subplots(figsize=(8, 12))

    # 绘制棋盘格子
    ax.set_xticks(np.arange(env.board_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(env.board_rows + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", size=0)
    ax.set_xlim(-0.5, env.board_cols - 0.5)
    ax.set_ylim(-0.5, env.board_rows - 0.5)
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

    # 绘制大本营
    for base in base_positions:
        y, x = base
        ax.add_patch(plt.Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='black'))

    # 绘制棋子可放置的位置
    for pos in positions:
        y, x = pos
        ax.plot(x, y, 'bo')

    plt.show()

# 创建JunQiEnvGame环境
class DummyPiece:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def get_position(self):
        return self.position

# 示例初始棋盘配置
initial_board = (
    [DummyPiece("军旗", (0, 0)), DummyPiece("地雷", (1, 1))],  # Red pieces
    [DummyPiece("军旗", (2, 2)), DummyPiece("地雷", (3, 3))]   # Blue pieces
)

env = JunQiEnvGame(initial_board)

# 可视化棋盘
visualize_board(env)

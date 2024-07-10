import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent_infer import DQNAgent as DQNAgentInfer, DoubleDQN
from junqi_env_infer import JunQiEnvInfer
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

# 初始化棋盘状态、移动历史、战斗历史和司令状态
board_state = np.random.randint(2, size=(13, 5, 13))
move_history = [((1, 1), (2, 2), 'red'), ((3, 3), (4, 4), 'blue')]
battle_history = [((5, 5), (6, 6), 'draw'), ((7, 7), (8, 8), 'win')]
red_commander_dead = False
blue_commander_dead = True

# 创建推理环境和智能体
initial_board = None  # 用随机初始化
env_infer = JunQiEnvInfer(initial_board)
model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
target_model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
agent_infer = DQNAgentInfer(model, target_model, action_size=env_infer.action_space.n)

# 加载训练好的推理模型权重
agent_infer.model.load_state_dict(torch.load('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/infer_model.pth'))
agent_infer.model.eval()

# 构造当前状态
state = {
    'board_state': board_state,
    'move_history': move_history,
    'battle_history': battle_history,
    'agent_color': 'red',
    'red_commander_dead': red_commander_dead,
    'blue_commander_dead': blue_commander_dead
}

# 使用推理模型进行推理
action, inferred_pieces = agent_infer.act(state)

# 可视化推理结果
def visualize_inferred_pieces(board_state, inferred_pieces):
    fig, ax = plt.subplots(figsize=(10, 6))
    board = np.zeros((board_state.shape[0], board_state.shape[1]), dtype=int)

    for pos, piece_type in inferred_pieces.items():
        x, y = pos
        board[x, y] = piece_encoding[piece_type]

    ax.set_xticks(np.arange(board.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(board.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.imshow(board, cmap='tab20', origin='upper')

    for pos, piece_type in inferred_pieces.items():
        x, y = pos
        ax.text(y, x, piece_type, ha='center', va='center', fontsize=12)

    plt.show()

visualize_inferred_pieces(board_state, inferred_pieces)

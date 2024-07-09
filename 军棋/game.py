import random
from junqi_env_game import JunQiEnvGame  # 确保你的环境文件名为 junqi_env_game.py
from pieces import Piece
from junqi_env_infer import JunQiEnvInfer
import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent_infer import DQNAgent, DoubleDQN, Memory
from junqi_env_infer import JunQiEnvInfer
from junqi_env_setup import JunQiEnvSetUp
from dqn_agent_setup import DQNAgent as DQNAgentSetUp
from tqdm import tqdm

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
env = JunQiEnvSetUp()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
action_size = env.get_action_space_size()
agent = DQNAgentSetUp(state_size, action_size, env, seed=0)

# 加载训练好的模型权重
agent.qnetwork_local.load_state_dict(torch.load('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/setup_model.pth'))

# 设置模型为评估模式
agent.qnetwork_local.eval()

def generate_deployment(env, agent, epsilon=0.1, max_t=1000):
    while True:
        state = env.reset()
        state = state.flatten()
        for t in range(max_t):
            action = agent.act(state, epsilon)  # 调用时传递 epsilon 参数
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            state = next_state
            if done:
                break

        # 检查是否所有棋子都有确定的位置
        all_pieces_placed = all(piece.get_position() is not None for piece in env.red_pieces + env.blue_pieces)
        if all_pieces_placed:
            break

# 生成布局
generate_deployment(env, agent, epsilon=0.2)  # 调整 epsilon 值来控制随机性

red_pieces = env.red_pieces
blue_pieces = env.blue_pieces

# 定义训练超参数
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 1e-3
EPISODES = 1000
MEMORY_SIZE = 10000

# 初始化环境和智能体
initial_board = [red_pieces, blue_pieces]

env_infer = JunQiEnvInfer(initial_board)
model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
target_model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
memory = Memory(capacity=10000)
PREFILL_STEPS = 2000  # 根据需要调整这个值
action_size = env_infer.action_space.n  # 获取动作空间的大小
agent_infer = DQNAgent(model, target_model, memory, action_size, env_infer.board_rows, env_infer.board_cols)

env_infer.generate_random_histories()

print(env_infer.battle_history)
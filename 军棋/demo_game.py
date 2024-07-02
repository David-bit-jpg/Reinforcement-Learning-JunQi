import torch
import torch.optim as optim
import os
import numpy as np
from collections import deque
import random
from dqn_agent_game import DQNAgent
from junqi_env_game import JunQiEnvGame
from dqn_agent_setup import DQNAgent as DQNAgentSetUp
from junqi_env_setup import JunQiEnvSetUp

# 确保模型保存的目录存在
MODEL_SAVE_PATH = "/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/game_model.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

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
initial_board = [red_pieces, blue_pieces]  # 初始化棋盘
env = JunQiEnvGame(initial_board, input_size=13*5*3, hidden_size=128, output_size=65, lr=0.001)
state_size = env.get_state_size()
action_size = len(env.red_pieces) * env.board_rows * env.board_cols
agent = DQNAgent(state_size, action_size, env, seed=0)
def train_agents(agent_red, agent_blue, num_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay):
    scores = []
    eps = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        score = 0
        current_player = 'red'
        current_agent = agent_red

        for t in range(max_t):
            action, pi, pi_reg = current_agent.act(state, env.turn, len(env.red_pieces), len(env.blue_pieces), features=[], player_color=current_player)
            next_state, reward, done, info = env.step(action, pi, pi_reg, current_player)
            current_agent.step(state.flatten(), action, reward, next_state.flatten(), True)  # 每步都认为done是True
            state = next_state.flatten()
            score += reward

            # 检查是否有玩家获胜
            winner = env.check_winner(current_player)
            if winner != 'No':
                print(f"Episode {episode + 1}: {winner}")
                break

            # 切换玩家和代理
            if current_player == 'red':
                current_player = 'blue'
                current_agent = agent_blue
            else:
                current_player = 'red'
                current_agent = agent_red

        scores.append(score)
        eps = max(epsilon_end, epsilon_decay * eps)
        print(f"Episode {episode + 1}/{num_episodes}, Score: {score}, Epsilon: {eps}")

    return agent_red, agent_blue

# 初始化环境和智能体
initial_board = [red_pieces, blue_pieces]  # 初始化棋盘
env = JunQiEnvGame(initial_board, input_size=13*5*3, hidden_size=128, output_size=65, lr=0.001)
state_size = env.get_state_size()
action_size = len(env.red_pieces) * env.board_rows * env.board_cols
agent_red = DQNAgent(state_size, action_size, env, seed=0)
agent_blue = DQNAgent(state_size, action_size, env, seed=1)

num_episodes = 1000
max_t = 100
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
trained_agent_red, trained_agent_blue = train_agents(agent_red, agent_blue, num_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay)

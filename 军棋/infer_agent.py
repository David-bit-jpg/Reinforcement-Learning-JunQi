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

# 设置参数
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 1e-3
EPISODES = 1000
MEMORY_SIZE = 10000

# 加载布局模型
env_setup = JunQiEnvSetUp()
state_size = env_setup.observation_space.shape[0] * env_setup.observation_space.shape[1] * env_setup.observation_space.shape[2]
action_size = env_setup.get_action_space_size()
agent_setup = DQNAgentSetUp(state_size, action_size, env_setup, seed=0)
agent_setup.qnetwork_local.load_state_dict(torch.load('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/setup_model.pth'))
agent_setup.qnetwork_local.eval()

# 生成布局
def generate_deployment(env, agent, epsilon=0.1, max_t=1000):
    while True:
        state = env.reset()
        state = state.flatten()
        for t in range(max_t):
            action = agent.act(state, epsilon)
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            state = next_state
            if done:
                break
        all_pieces_placed = all(piece.get_position() is not None for piece in env.red_pieces + env.blue_pieces)
        if all_pieces_placed:
            break

generate_deployment(env_setup, agent_setup, epsilon=0.2)
red_pieces = env_setup.red_pieces
blue_pieces = env_setup.blue_pieces
initial_board = [red_pieces, blue_pieces]

# 初始化推理环境
env_infer = JunQiEnvInfer(initial_board)
model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
target_model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
memory = Memory(capacity=10000)
agent_infer = DQNAgent(model, target_model, memory, env_infer.piece_types, env_infer.board_rows, env_infer.board_cols)

# 训练函数
def train_agent(env, agent, episodes=1000):
    for e in range(episodes):
        state = env.reset()
        state = np.transpose(state['board_state'], (2, 0, 1))  # 调整状态的维度顺序
        state = state[:3]  # 确保状态只有3个通道
        state = np.expand_dims(state, axis=0)  # 增加一个维度
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.transpose(next_state['board_state'], (2, 0, 1))
            next_state = next_state[:3]  # 确保状态只有3个通道
            next_state = np.expand_dims(next_state, axis=0)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay()
        
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")


# 训练agent
train_agent(env_infer, agent_infer, episodes=EPISODES)

# 保存模型权重
torch.save(agent_infer.model.state_dict(), '/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/infer_model.pth')

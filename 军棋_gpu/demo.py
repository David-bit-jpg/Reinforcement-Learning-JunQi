import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent_setup import DQNAgent
from junqi_env_setup import JunQiEnvSetUp


# 创建环境和智能体
env = JunQiEnvSetUp()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
action_size = env.get_action_space_size()
agent = DQNAgent(state_size, action_size, env, seed=0)

# 加载训练好的模型权重
agent.qnetwork_local.load_state_dict(torch.load('/code/军棋/models/setup_model.pth'))

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

# 可视化布局
env.visualize_full_board()

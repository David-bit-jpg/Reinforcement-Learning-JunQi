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

initial_board = [red_pieces, blue_pieces]

def train_agent(env, agent, episodes=3000):
    rewards = []
    losses = []

    for e in range(episodes):
        state = env.reset()
        env.generate_random_histories()
        total_reward = 0
        done = False
        loss_list = []

        while not done:
            try:
                action, inferred_pieces = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # 打印当前状态和动作以进行调试
                print(f"Episode: {e}, Action: {action}, Inferred Pieces: {inferred_pieces}")
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward
            except ValueError as ve:
                # print(f"ValueError encountered: {ve}")
                # print(f"State tensor shape: {state['board_state'].shape}")
                raise ve

        rewards.append(total_reward)
        if len(loss_list) > 0:
            avg_loss = np.mean(loss_list)
            losses.append(avg_loss)
        else:
            avg_loss = None
            losses.append(None)

        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Reward: {total_reward}, Avg Loss: {avg_loss if avg_loss is not None else 'N/A'}")

    return rewards, losses

env_infer = JunQiEnvInfer(initial_board=initial_board)
state_shape = env_infer.observation_space.shape
action_size = env_infer.action_space.n

model = DoubleDQN(state_shape[0], state_shape[1], list(piece_encoding.keys()))
target_model = DoubleDQN(state_shape[0], state_shape[1], list(piece_encoding.keys()))

agent_infer = DQNAgent(model, target_model, action_size, initial_board)

rewards, losses = train_agent(env_infer, agent_infer, episodes=3000)

# 保存模型权重
agent_infer.save('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/infer_model.pth')

# 绘制奖励和损失图表
def plot_training_progress(rewards, losses):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Loss', color=color)
    ax2.plot(losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

plot_training_progress(rewards, losses)

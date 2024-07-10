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
env_infer = JunQiEnvInfer(initial_board)
model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
target_model = DoubleDQN(env_infer.board_rows, env_infer.board_cols, env_infer.piece_types)
agent_infer = DQNAgent(model, target_model, action_size)

def train_agent(env, agent, episodes=1000):
    rewards = []
    losses = []
    
    # 创建一个 tqdm 进度条
    with tqdm(total=episodes, desc="Training Progress", unit="episode") as pbar:
        for e in range(episodes):
            state_dict = env.reset()
            board_state = state_dict['board_state']
            transposed_state = np.transpose(board_state, (2, 0, 1))

            channels = transposed_state.shape[0]
            height = transposed_state.shape[1]
            width = transposed_state.shape[2]
            padded_state = np.zeros((1, channels, height, width))
            padded_state[0, :channels, :height, :width] = transposed_state

            state = {
                'board_state': padded_state, 
                'move_history': state_dict['move_history'], 
                'battle_history': state_dict['battle_history'], 
                'agent_color': 'red', 
                'opponent_commander_dead': state_dict['opponent_commander_dead']
            }

            done = False
            total_reward = 0
            episode_losses = []

            while not done:
                action, inferred_pieces = agent.act(state)
                next_state_dict, reward, done, info = env.step(action)
                next_board_state = next_state_dict['board_state']
                next_transposed_state = np.transpose(next_board_state, (2, 0, 1))

                next_channels = next_transposed_state.shape[0]
                next_height = next_transposed_state.shape[1]
                next_width = next_transposed_state.shape[2]
                next_padded_state = np.zeros((1, next_channels, next_height, next_width))
                next_padded_state[0, :next_channels, :next_height, :next_width] = next_transposed_state

                next_state = {
                    'board_state': next_padded_state, 
                    'move_history': next_state_dict['move_history'], 
                    'battle_history': next_state_dict['battle_history'], 
                    'agent_color': 'red', 
                    'opponent_commander_dead': next_state_dict['opponent_commander_dead']
                }

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss.item())

            rewards.append(total_reward)
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                losses.append(avg_loss)
            else:
                losses.append(0)

            # 更新进度条
            pbar.set_postfix({"Total Reward": total_reward, "Avg Loss": avg_loss if episode_losses else 0})
            pbar.update(1)

            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return rewards, losses

# 训练 agent
rewards, losses = train_agent(env_infer, agent_infer, episodes=1000)
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

import torch
import torch.optim as optim
import os
import numpy as np
from collections import deque
import random
from dqn_agent_game import DQNAgent
from junqi_env_game import JunQiEnvGame as JunQiEnvGameModel
from junqi_env_game_model import JunQiEnvGame as JunQiEnvGameSetModel
from dqn_agent_setup import DQNAgent as DQNAgentSetUp
from junqi_env_setup import JunQiEnvSetUp
from tqdm import tqdm

# 确保模型保存的目录存在
MODEL_SAVE_PATH = "/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/game_model.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 初始化布局环境和布局代理
env_setup = JunQiEnvSetUp()
state_size_setup = env_setup.observation_space.shape[0] * env_setup.observation_space.shape[1] * env_setup.observation_space.shape[2]
action_size_setup = env_setup.get_action_space_size()
agent_setup = DQNAgentSetUp(state_size_setup, action_size_setup, env_setup, seed=0)

# 加载训练好的模型权重
agent_setup.qnetwork_local.load_state_dict(torch.load('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/setup_model.pth'))

# 设置模型为评估模式
agent_setup.qnetwork_local.eval()

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
generate_deployment(env_setup, agent_setup, epsilon=0.2)  # 调整 epsilon 值来控制随机性

red_pieces = env_setup.red_pieces
blue_pieces = env_setup.blue_pieces

# 初始化环境和智能体
initial_board = [red_pieces, blue_pieces]  # 初始化棋盘
env_model = JunQiEnvGameModel(initial_board, input_size=13 * 5 * 4, hidden_size=128, output_size=65, lr=0.001)
env_setmodel = JunQiEnvGameSetModel(initial_board, input_size=13 * 5 * 4, hidden_size=128, output_size=65, lr=0.001)
state_size = env_model.get_state_size()
action_size = len(env_model.red_pieces) * env_model.board_rows * env_model.board_cols
agent_red = DQNAgent(state_size, action_size, env_model, seed=0)
agent_blue = DQNAgent(state_size, action_size, env_setmodel, seed=1)

def train_agents(agent_red, agent_blue, num_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay):
    scores = []
    eps = epsilon_start
    red_moves = []
    blue_moves = []
    red_rewards = []
    blue_rewards = []

    for episode in range(num_episodes):
        state = env_model.reset()
        state = state.flatten()
        score = 0
        current_player = 'red'
        current_agent = agent_red
        episode_moves = {'red': [], 'blue': []}
        red_episode_reward = 0
        blue_episode_reward = 0
        last_move = None
        battle_info = None

        with tqdm(total=max_t, desc=f"Episode {episode + 1}/{num_episodes}") as pbar:
            for t in range(max_t):
                try:
                    action, pi, pi_reg = current_agent.act(state, env_model.turn, len(env_model.red_pieces), len(env_model.blue_pieces), features=[], player_color=current_player)
                except ValueError as e:
                    print(e)
                    break

                next_state, reward, done, info = env_model.step(action, pi, pi_reg, current_player, current_agent.get_weights())
                current_agent.step(state.flatten(), action, reward, next_state.flatten(), done)
                state = next_state.flatten()
                score += reward

                # 记录动作
                episode_moves[current_player].append(action)

                # 记录奖励
                if current_player == 'red':
                    red_episode_reward += reward
                else:
                    blue_episode_reward += reward

                # 更新进度条描述
                pbar.set_postfix({'Red Reward': red_episode_reward, 'Blue Reward': blue_episode_reward, 'Epsilon': eps})
                pbar.update(1)

                # 每100步可视化推理棋盘
                # if t % 100 == 0:
                # env_model.visualize_inferred_board(player_color=current_player)
                # 检查是否有战斗发生并记录战斗信息
                if 'battle_info' in info:
                    battle_info = info['battle_info']
                    last_move = (current_agent.get_piece_by_index(action // (env_model.board_rows * env_model.board_cols)), (action // env_model.board_cols % env_model.board_rows, action % env_model.board_cols), next_state)
                    # env_model.visualize_full_board(last_move=last_move, battle_info=battle_info)
                    battle_info = None  # 重置战斗信息

                # 检查是否有玩家获胜
                winner = env_model.check_winner(current_player)
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
        red_rewards.append(red_episode_reward)
        blue_rewards.append(blue_episode_reward)
        
        # 打印进度和奖励
        print(f"Episode {episode + 1}/{num_episodes}, Red Reward: {red_episode_reward:.2f}, Blue Reward: {blue_episode_reward:.2f}, Epsilon: {eps:.2f}")
        
        # 将每个episode的动作保存到总的动作列表中
        red_moves.append(episode_moves['red'])
        blue_moves.append(episode_moves['blue'])

    # 保存动作到文件
    with open('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/red_moves.txt', 'w') as f:
        for episode in red_moves:
            f.write(' '.join(map(str, episode)) + '\n')

    with open('/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/blue_moves.txt', 'w') as f:
        for episode in blue_moves:
            f.write(' '.join(map(str, episode)) + '\n')

    return agent_red, agent_blue

# 训练代理
num_episodes = 1000
max_t = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
trained_agent_red, trained_agent_blue = train_agents(agent_red, agent_blue, num_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay)
torch.save(trained_agent_red.qnetwork_local.state_dict(), '/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/game_agent_with_inference_model.pth')
torch.save(trained_agent_blue.qnetwork_local.state_dict(), '/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/models/game_agent_without_inference_moel.pth')

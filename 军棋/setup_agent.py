import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent_setup import DQNAgent
from junqi_env_setup import JunQiEnvSetUp

def dqn(n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        state = state.flatten()
        score = 0
        for t in range(max_t):
            action = agent.act(state, epsilon)  # 调用时传递 epsilon 参数
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        epsilon = max(eps_end, eps_decay * epsilon)  # 动态调整 epsilon 值
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.2f}')
    
    # 保存最后的模型
    torch.save(agent.qnetwork_local.state_dict(), '/Users/davidwang/Documents/GitHub/LLM_GAME/军棋/modles/setup_model.pth')
    return scores

# 创建环境和智能体
env = JunQiEnvSetUp()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
action_size = env.get_action_space_size()
agent = DQNAgent(state_size, action_size, env, seed=0)

# 训练智能体
scores = dqn()

# 可视化训练曲线
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

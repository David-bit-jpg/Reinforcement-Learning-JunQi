from avalon_game import AvalonEnvironment
from RL_agent import CustomRLAgent
# 训练自定义强化学习代理
env = AvalonEnvironment()
agent = CustomRLAgent(state_size=len(env.get_state()), action_size=len(env.available_actions()))

# 训练循环
for episode in range(1000):
    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_exploration()
    print(f"Episode {episode}: Total Reward: {total_reward}")

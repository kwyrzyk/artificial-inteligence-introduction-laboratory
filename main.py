from QLearningAgent import *
import gym
import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use("Agg")


def test_learning_rates(episodes, trials, learning_rates, strategy):
    results = {lr: [] for lr in learning_rates}
    env = GymEnvironment('Taxi-v3')

    environments = []
    for _ in range(trials):
        env = GymEnvironment('Taxi-v3')
        environments.append(env)

    for lr in learning_rates:
        trial_rewards = []
        for env in environments:
            env_copy = copy.deepcopy(env)
            agent_params = AgentParams(env_copy, learning_rate=lr, discount_factor=0.99)
            agent = QLearningAgent(agent_params)
            rewards = agent.train(episodes, strategy)
            trial_rewards.append(rewards)
        
        avg_rewards = np.mean(trial_rewards, axis=0)
        results[lr] = avg_rewards   

    plt.figure(figsize=(10, 6))
    for lr, rewards in results.items():
        plt.plot(range(episodes), rewards, label=f"Learning Rate: {lr}")

    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.title("Comparison of Learning Rates - " + str(strategy))
    plt.legend()
    plt.grid()
    plt.savefig(str(strategy).replace(" ", "_").lower() + '_learning_rate.png')
    # plt.savefig('learning_rate.png')

class GymEnvironment(Environment):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.states_num = self.env.observation_space.n
        self.actions_num = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()
        return int(state)
    
    def step(self, action):
        next_state, reward, done,_ ,_ = self.env.step(action)
        return int(next_state), int(reward), int(done)

if __name__ == "__main__":
    # env = GymEnvironment('Taxi-v3')
    # agent_params = AgentParams(env, 0.1, 0.99)
    # EPISODES = 1000

    # agent = QLearningAgent(agent_params)
    # result = agent.train(EPISODES, EpsilonGreedyStrategy(0.1))

    # test_learning_rates(1000, 3, [0.01, 0.1, 0.2], EpsilonGreedyStrategy(0.1))
    test_learning_rates(1000, 1, [0.01], EpsilonGreedyStrategy(0.1))
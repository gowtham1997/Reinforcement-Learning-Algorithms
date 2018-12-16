import gym
# import numpy as np
from collections import Counter, defaultdict
from tensorboardX import SummaryWriter


GAMMA = 0.9
ENV = 'FrozenLake-v0'
# ENV = 'FrozenLake8x8-v0'
TEST_EPISODES = 20


class Agent():

    def __init__(self, gamma, env):
        self.transitions_table = defaultdict(Counter)
        self.rewards_table = defaultdict(float)
        self.Qvalues_table = defaultdict(float)
        self.gamma = gamma
        self.env = gym.make(env)
        self.state = self.env.reset()

    def get_best_action(self, state):
        best_Q, best_action = float("-inf"), 0
        for action in range(self.env.action_space.n):
            Q = self.Qvalues_table[(state, action)]
            if Q > best_Q:
                best_Q = Q
                best_action = action
        return best_action

    def value_iterate(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                Q_value = 0.0
                transitions = self.transitions_table[(state, action)]
                total = sum(transitions.values())
                for new_state, num_transition in transitions.items():
                    reward = self.rewards_table[(state, action, new_state)]
                    new_best_action = self.get_best_action(new_state)
                    Q_value += (num_transition /
                                total) * (reward +
                                          GAMMA *
                                          self.Qvalues_table[(new_state,
                                                              new_best_action)
                                                             ])
                self.Qvalues_table[(state, action)] = Q_value

    def play_n_random_steps(self, n):
        # for exploration
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards_table[(self.state, action, new_state)] = reward
            self.transitions_table[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def play_episode(self, env):
        # we pass a seperate env here so that the state and env of the class
        # don't change
        total_reward = 0.0
        state = env.reset()
        while True:
            # we select action greedily(exploitation) here as exploration is
            # already done by calling play_n_random_steps function
            action = self.get_best_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards_table[(state, action, new_state)] = reward
            self.transitions_table[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV)
    agent = Agent(GAMMA, ENV)
    writer = SummaryWriter()
    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        # play n random steps are update values
        agent.play_n_random_steps(100)
        agent.value_iterate()
        reward = 0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar('reward', reward, iter_no)

        if reward > best_reward:
            print("Best reward updated from %.3f to %.3f at %.1f "
                  "iterations" % (best_reward,
                                  reward, iter_no))
            best_reward = reward
        if reward > 0.8:
            print('Successfully completed at {} iterations'.format(iter_no))
            break
        if iter_no > 500:
            print('Not suceesful')
            break
    test_env.env.close()

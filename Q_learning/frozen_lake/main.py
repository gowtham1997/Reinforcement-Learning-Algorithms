import gym
from collections import defaultdict
from tensorboardX import SummaryWriter

GAMMA = 0.9
ALPHA = 0.5
TEST_EPISODES = 20
ENV = 'FrozenLake-v0'


class Agent:

    def __init__(self):
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.reward_table = defaultdict(float)
        self.Qvalues_table = defaultdict(float)

    def sample_env(self):
        old_state = self.state
        action = self.env.action_space.sample()
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def get_best_value_and_action(self, state):
        best_Qvalue, best_action = float("-inf"), 0.0
        for action in range(self.env.action_space.n):
            value = self.Qvalues_table[(state, action)]
            if best_Qvalue < value:
                best_Qvalue = value
                best_action = action
        return best_Qvalue, best_action

    def Qvalue_update(self, state, action, reward, new_state):
        best_Qvalue, _ = self.get_best_value_and_action(new_state)
        new_Q = reward + GAMMA * best_Qvalue
        old_Q = self.Qvalues_table[(state, action)]
        self.Qvalues_table[(state, action)] = (
            1 - ALPHA) * old_Q + ALPHA * new_Q

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.get_best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV)
    writer = SummaryWriter(comment="-q-learning")
    agent = Agent()
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        old_state, action, reward, new_state = agent.sample_env()
        agent.Qvalue_update(old_state, action, reward, new_state)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated from %.3f to %.3f at %.1f "
                  "iterations" % (best_reward,
                                  reward, iter_no))
            best_reward = reward
        if reward > 0.8:
            print('Successfully completed at {} iterations'.format(iter_no))
            break
    test_env.env.close()
    agent.env.env.close()

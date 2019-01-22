import gym
import ptan
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # note we return unscaled logits and hence we need to apply
        # log_softmax when calculating loss
        return self.net(x)


def calc_q_vals(rewards):
    # for policy gradient calculation we need q values at every time step
    # q_t = r_t + GAMMA * q_(t+1)
    sum_r = 0.0
    q_vals = []
    for reward in reversed(rewards):
        sum_r *= GAMMA
        sum_r += reward
        q_vals.append(sum_r)
    # reversed returns an iterator so make it a list
    return list(reversed(q_vals))


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    writer = SummaryWriter(comment='-cartpole_reinforce')

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    # float_32_preprocesser vectorises input into torch float32 tensors
    agent = ptan.agent.PolicyAgent(
        net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0
    # to store local rewards in an episode
    curr_rewards = []
    batch_states, batch_actions, batch_qvals = [], [], []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(exp.action)
        curr_rewards.append(exp.reward)

        # if episode is over calculate accumulated q_val from local rewards
        if exp.reward is None:
            batch_episodes += 1
            batch_qvals.extend(calc_q_vals(curr_rewards))
            curr_rewards.clear()

        # bookkeeping and printing progress to tensorboard
        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" %
                      (step_idx, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_v = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)

        log_prob_actions_v = batch_qvals * \
            log_prob_v[range(len(batch_states)), batch_actions_v]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()

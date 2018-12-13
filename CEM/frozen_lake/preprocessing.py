import torch
import numpy as np
from collections import namedtuple
import gym
import torch.nn as nn


class DiscrteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # converting discrete to Box space
        self.observation_space = gym.spaces.Box(0.0, 1.0,
                                                (env.observation_space.n, ),
                                                np.float32)

    def observation(self, obs):
        # one hot conversion of discrete obs
        new_obs = np.copy(self.observation_space.low)
        new_obs[obs] = 1.0
        return new_obs


def iterate_batches(env, net, batch_size):
    np.random.seed(12345)
    obs = env.reset()
    Episode_Step = namedtuple('Episode_Step', field_names=[
                              'observation', 'action'])
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])

    episode_steps = []
    batch = []
    episode_reward = 0.0

    softmax = nn.Softmax(dim=1)

    while True:
        obs_tensor = torch.FloatTensor([obs])
        logits = net(obs_tensor)
        action_prob_tensor = softmax(logits)
        action_probs = action_prob_tensor.data.numpy()[0]

        action = np.random.choice(len(action_probs), p=action_probs)
        new_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(Episode_Step(obs, action))

        if is_done:
            batch.append(Episode(episode_reward, episode_steps))
            new_obs = env.reset()
            episode_reward = 0.0
            episode_steps = []
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = new_obs

        # print(action_probs)


def filter_batches(batch, percentile, gamma):
    # discound reward and value episodes which take less steps to win.
    disc_rewards = list(
        map(lambda s: s.reward * (gamma ** len(s.steps)), batch))
    reward_boundary = np.percentile(disc_rewards, percentile)
    # store elite examples and pass them on in the next iteration.
    elite_batch = []

    train_obs = []
    train_actions = []
    for element, disc_reward in zip(batch, disc_rewards):
        if disc_reward > reward_boundary:
            train_actions.extend(map(lambda s: s.action, element.steps))
            train_obs.extend(map(lambda s: s.observation, element.steps))
            elite_batch.append(element)

    return elite_batch, train_obs, train_actions, reward_boundary

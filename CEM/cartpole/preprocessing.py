import torch
import numpy as np
from collections import namedtuple
import torch.nn as nn


def iterate_batches(env, net, input_shape, batch_size):

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


def filter_batches(batch, percentile):

    rewards = list(map(lambda s: s.reward, batch))
    reward_boundary = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    train_obs = []
    train_actions = []
    for element in batch:
        if element.reward < reward_boundary:
            continue
        train_actions.extend(map(lambda s: s.action, element.steps))
        train_obs.extend(map(lambda s: s.observation, element.steps))

    return train_obs, train_actions, reward_boundary, reward_mean

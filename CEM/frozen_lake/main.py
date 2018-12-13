import gym
import numpy as np
from model import Net
from preprocessing import iterate_batches, filter_batches, DiscrteOneHotWrapper
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
import torch.optim as optim

BATCH_SIZE = 100
HIDDEN_SIZE = 128
PERCENTILE = 30
LEARNING_RATE = 0.001
GAMMA = 0.9

if __name__ == '__main__':
    env = DiscrteOneHotWrapper(gym.make('FrozenLake-v0'))
    env = gym.wrappers.Monitor(env, 'monitor', force=True)
    obs = env.reset()
    INPUT_SHAPE = len(obs)
    NUM_ACTION = env.action_space.n
    # print(NUM_ACTION)
    objective = nn.CrossEntropyLoss()
    net = Net(INPUT_SHAPE, HIDDEN_SIZE, NUM_ACTION)
    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()

    full_batch = []

    for iter_num, batch in \
            enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, train_obs, train_actions, \
            reward_boundary = filter_batches(
                full_batch + batch, PERCENTILE, GAMMA)
        if not full_batch:
            continue
        train_obs = torch.FloatTensor(train_obs)
        train_actions = torch.LongTensor(train_actions)
        # only keep latest 500 episodes
        full_batch = full_batch[-500:]
        # print(train_obs.shape, train_actions.shape)

        opt.zero_grad()
        logits = net(train_obs)
        loss = objective(logits, train_actions)
        loss.backward()
        opt.step()
        print(
            "%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
                iter_num, loss.item(),
                reward_mean, reward_boundary, len(full_batch)))
        writer.add_scalar("loss", loss.item(), iter_num)
        writer.add_scalar("reward_mean", reward_mean, iter_num)
        writer.add_scalar("reward_bound", reward_boundary, iter_num)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
    env.env.close()

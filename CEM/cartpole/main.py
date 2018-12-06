import gym
from model import Net
from preprocessing import iterate_batches, filter_batches
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
import torch.optim as optim

BATCH_SIZE = 16
HIDDEN_SIZE = 128
PERCENTILE = 70
LEARNING_RATE = 0.01

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, 'monitor', force=True)
    obs = env.reset()
    INPUT_SHAPE = len(obs)
    NUM_ACTION = env.action_space.n
    # print(NUM_ACTION)
    objective = nn.CrossEntropyLoss()
    net = Net(INPUT_SHAPE, HIDDEN_SIZE, NUM_ACTION)
    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()
    for iter_num, batch in \
            enumerate(iterate_batches(env, net, INPUT_SHAPE, BATCH_SIZE)):
        train_obs, train_actions, \
            reward_boundary, reward_mean = filter_batches(batch, PERCENTILE)
        train_obs = torch.FloatTensor(train_obs)
        train_actions = torch.LongTensor(train_actions)
        # print(train_obs.shape, train_actions.shape)

        opt.zero_grad()
        logits = net(train_obs)
        loss = objective(logits, train_actions)
        loss.backward()
        opt.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_num, loss.item(), reward_mean, reward_boundary))
        writer.add_scalar("loss", loss.item(), iter_num)
        writer.add_scalar("reward_boundary", reward_boundary, iter_num)
        writer.add_scalar("reward_mean", reward_mean, iter_num)
        if reward_mean > 199:
            print('Solved')
            break
    writer.close()
    env.env.close()

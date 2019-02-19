import gym
import ptan
import numpy as np
import collections
import os

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch import multiprocessing as mp
from lib import common

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = os.cpu_count() - 1
NUM_ENVS = 15

TotalReward = collections.namedtuple('TotalReward', field_names='reward')


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(
        x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, GAMMA, REWARD_STEPS)
    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'running on device {device}')
    print(f'number of processes {PROCESSES_COUNT}')
    writer = SummaryWriter(comment="-a3c-data_" + NAME)
    env = make_env()
    print(env.observation_space.shape, env.action_space.n)
    net = common.AtariA2C(env.observation_space.shape,
                          env.action_space.n).to(device)
    print(net)
    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(
            target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    step_idx = 0
    batch = []

    try:
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(writer,
                                                 batch_size=100) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, step_idx):
                            break

                    step_idx += 1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue

                    states_v, actions_t, vals_ref_v = \
                        common.unpack_batch(
                            batch, net,
                            last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                    batch.clear()

                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)
                    # print(
                    #    # f'logit shape: {logits_v.shape}, ' +
                    #    # f'value_v shape: {value_v.shape}')
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                    # print(f'loss_value_v shape: {loss_value_v.shape}')
                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                    # print(vals_ref_v, value_v)
                    # print(f'adv_v shape: {adv_v.shape}')
                    log_prob_actions_v = adv_v * \
                        log_prob_v[range(BATCH_SIZE), actions_t]
                    # print(log_prob_v[range(BATCH_SIZE), actions_t])
                    # print(
                    #    # f'log_prob_actions_v.shape: ' +
                    #    # f'{log_prob_actions_v.shape}')
                    loss_policy_v = -log_prob_actions_v.mean()
                    # print(f'loss_policy_v.shape: {loss_policy_v.shape}')
                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA * \
                        (prob_v * log_prob_v).sum(dim=1).mean()
                    # print(f'entropy_loss_v.shape: {entropy_loss_v.shape}')
                    # break
                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()

                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        for data_proc in data_proc_list:
            data_proc.terminate()
            data_proc.join()

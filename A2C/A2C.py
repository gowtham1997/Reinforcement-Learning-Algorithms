import gym

import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 10
NUM_ENVS = 1

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out_size(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def get_conv_out_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        out = self.conv(x)
        return int(np.prod(out.size()))

    def forward(self, x):
        fx = x.float() / 255.0
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    # returns state, actions, and q_vals
    states = []
    actions = []
    last_states = []
    not_done_idxs = []
    rewards = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state:
            not_done_idxs.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idxs:
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_values_v = net(last_states_v)[1]
        last_values_np = last_values_v.data.cpu().numpy()
        last_values_np = np.squeeze(last_values_np)
        rewards_np[not_done_idxs] += GAMMA ** REWARD_STEPS * last_values_np

    vals_ref_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_v, vals_ref_v


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device {}'.format(device))
    writer = SummaryWriter(comment='-A2C algorithm')

    def make_env():
        return ptan.common.wrappers.wrap_dqn(gym.make(
            'PongNoFrameskip-v4'))
    # create multiple envs to get lot of uncorrelated data
    envs = [make_env() for _ in range(NUM_ENVS)]

    net = A2C(envs[0].observation_space.shape,
              envs[0].action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(lambda x: net(
        x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(
        net.parameters(), lr=LEARNING_RATE, eps=1e-3, amsgrad=True)
    # for storing the experience
    batch = []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer,
                                             batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue
                states_v, actions_v, vals_ref_v = unpack_batch(
                    batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, values_v = net(states_v)
                advantage_v = vals_ref_v - values_v.squeeze(-1)

                value_loss_v = advantage_v.pow(2).mean()

                log_probs_v, probs_v = F.log_softmax(
                    logits_v, dim=1), F.softmax(logits_v, dim=1)

                loss_policy_v = advantage_v * \
                    log_probs_v[range(BATCH_SIZE), actions_v]
                loss_policy_v = -loss_policy_v.mean()

                entropy_loss_v = ENTROPY_BETA * \
                    (probs_v * log_probs_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                loss_v = entropy_loss_v + value_loss_v
                loss_v.backward()
                # clip the gradients to CLIP_GRAD
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()

                # get the full loss
                loss_v += loss_policy_v
                tb_tracker.track("advantage", advantage_v, step_idx)
                tb_tracker.track("values", values_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", value_loss_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(
                    np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)

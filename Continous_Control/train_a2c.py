"""Summary

Attributes
----------
BATCH_SIZE : int
Description
ENTROPY_BETA : float
Description
ENV_ID : str
Description
GAMMA : float
Description
LEARNING_RATE : float
Description
REWARD_STEPS : int
Description
TEST_ITERS : int
Description
"""
import gym
import pybullet_envs

import argparse
import os
import time

import ptan
import torch
import numpy as np

import math
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from lib import model, common

ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000


def test_net(net, env, count=10, device="cpu"):
    """Summary

    Parameters
    ----------
    net : TYPE
        Description
    env : TYPE
        Description
    count : int, optional
        Description
    device : str, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def log_gaussian_policy(mu_v, var_v, actions_v):
    """Summary

    Parameters
    ----------
    mu_v : TYPE
        Description
    var_v : TYPE
        Description
    actions_v : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    p1 = - ((mu_v - actions_v) ** 2 / (2 * var_v.clamp(min=1e-3)))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def entropy_gaussian(var_v):
    """Summary

    Parameters
    ----------
    var_v : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return (torch.log(2 * math.pi * var_v) + 1) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    net = model.ModelA2C(
        env.observation_space.shape, env.action_size.shape).to(device)
    print(net)
    agent = model.AgentA2C(device=device, net=net)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" %
                                  (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                optimizer.zero_grad()
                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net, GAMMA**REWARD_STEPS, device=device)

                mu_v, var_v, values_v = net(states_v)

                loss_value_v = F.mse_loss(values_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - values_v.detach()

                loss_policy_v = - (adv_v * log_gaussian_policy(
                    mu_v, var_v, actions_v)).mean()

                entropy_loss_v = ENTROPY_BETA * \
                    (- entropy_gaussian(var_v)).mean()

                loss_v = loss_value_v + loss_policy_v + entropy_loss_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", values_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

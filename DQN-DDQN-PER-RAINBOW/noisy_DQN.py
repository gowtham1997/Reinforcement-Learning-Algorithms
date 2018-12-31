import gym
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from ptan.common import wrappers
from ptan.actions import ArgmaxActionSelector
from ptan.agent import DQNAgent, TargetNet
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from lib.common import HYPERPARAMS, RewardTracker, \
    calc_loss_dqn
from lib.model import NoisyDQN
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='n-step DQN')
    parser.add_argument('-n',
                        default=1,
                        type=int,
                        help='Enter the number of steps to unroll bellman eq')
    args = parser.parse_args()

    print('Starting...')
    params = HYPERPARAMS['pong']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on Device {}'.format(device))
    writer = writer = SummaryWriter(
        comment="-" + params['run_name'] + "-%d-step" % args.n)
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)
    # print(env.observation_space.shape, env.action_space.n)
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = TargetNet(net)

    agent = DQNAgent(net, ArgmaxActionSelector(), device)

    experience_source = ExperienceSourceFirstLast(
        env, agent, params['gamma'], steps_count=args.n)
    buffer = ExperienceReplayBuffer(
        experience_source, buffer_size=params['replay_size'])

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    frame_idx = 0
    with RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            # get latest rewards
            new_rewards = experience_source.pop_total_rewards()
            # new_rewards are empty till the end of the episode
            # so we need to check if the list is empty to pass it to
            # reward_tracker
            if new_rewards:
                if reward_tracker.reward(new_rewards[0],
                                         frame_idx):
                    break
            # till buffer fills up and we can sample continue the loop
            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss = calc_loss_dqn(
                batch, net, target_net.target_model,
                params['gamma'] ** args.n, device)
            loss.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()
            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in \
                        enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx + 1),
                                      sigma_l2, frame_idx)

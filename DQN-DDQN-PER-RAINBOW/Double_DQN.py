import gym
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from ptan.common import wrappers
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import DQNAgent, TargetNet
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from lib.common import HYPERPARAMS, EpsilonTracker, RewardTracker, \
    calc_loss_dqn, calculate_mean_Q
from lib.model import DQN
import argparse

n = 1
EVAL_STATES_FRAMES = 100
NUM_EVAL_STATES = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='n-step DQN')
    parser.add_argument('-n',
                        default=n,
                        type=int,
                        help='Enter the number of steps to unroll bellman eq')
    parser.add_argument('--double', '-d',
                        default=True,
                        action="store_false",
                        type=bool,
                        help='Enable double DQN')
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
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = EpsilonTracker(selector, params)

    agent = DQNAgent(net, selector, device)

    experience_source = ExperienceSourceFirstLast(
        env, agent, params['gamma'], steps_count=args.n)
    buffer = ExperienceReplayBuffer(
        experience_source, buffer_size=params['replay_size'])

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    frame_idx = 0
    eval_states = None

    with RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            # get latest rewards
            new_rewards = experience_source.pop_total_rewards()
            # new_rewards are empty till the end of the episode
            # so we need to check if the list is empty to pass it to
            # reward_tracker
            if new_rewards:
                if reward_tracker.reward(new_rewards[0],
                                         frame_idx, selector.epsilon):
                    break
            # till buffer fills up and we can sample continue the loop
            if len(buffer) < params['replay_initial']:
                continue
            if eval_states is None:
                eval_states = buffer.sample(NUM_EVAL_STATES)
                eval_states = [np.array(transition.state, copy=False)
                               for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss = calc_loss_dqn(
                batch, net, target_net.target_model,
                params['gamma'] ** args.n, device, args.double)
            loss.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()

            if frame_idx % EVAL_STATES_FRAMES == 0:
                mean_Q = calculate_mean_Q(eval_states, net, device)
                writer.add_scalar('mean_Q', mean_Q, frame_idx)

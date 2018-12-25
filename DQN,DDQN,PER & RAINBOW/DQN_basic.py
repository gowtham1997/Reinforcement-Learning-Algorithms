import gym
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from ptan.common import wrappers
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import DQNAgent, TargetNet
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from lib.common import HYPERPARAMS, EpsilonTracker, RewardTracker, \
    calc_loss_dqn
from lib.model import DQN

if __name__ == "__main__":
    print('Starting...')
    params = HYPERPARAMS['pong']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on Device {}'.format(device))
    writer = SummaryWriter(comment='- dqn_basic')
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)
    # print(env.observation_space.shape, env.action_space.n)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = EpsilonTracker(selector, params)

    agent = DQNAgent(net, selector, device)

    experience_source = ExperienceSourceFirstLast(
        env, agent, params['gamma'], steps_count=1)
    buffer = ExperienceReplayBuffer(
        experience_source, buffer_size=params['replay_size'])

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    frame_idx = 0
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

            optimizer.step()
            batch = buffer.sample(params['batch_size'])
            loss = calc_loss_dqn(
                batch, net, target_net.target_model, params['gamma'], device)
            loss.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()

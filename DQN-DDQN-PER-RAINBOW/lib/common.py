import numpy as np
import sys
import torch
import torch.nn as nn
import time


HYPERPARAMS = {
    'pong': {
        'env_name': "PongNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 100000,
        'replay_initial': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10**5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    },
    'breakout-small': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout-small',
        'replay_size': 3 * 10 ** 5,
        'replay_initial': 20000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64
    },
    'breakout': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []

    for exp in batch:
        states.append(np.array(exp.state, copy=False))
        actions.append(np.array(exp.action))
        rewards.append(np.array(exp.reward))
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            # we just add prev_state in case last_state is none
            # This will be masked with dones array during the loss calculation
            last_states.append(np.array(exp.state, copy=False))
        else:
            last_states.append(np.array(exp.last_state, copy=False))

    return np.array(states, copy=False), \
        np.array(actions), \
        np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), \
        np.array(last_states, copy=False)


def calculate_mean_Q(states, net, device="cpu"):
    mean_Q_vals = []
    for batch in np.array_split(states, 64):
        batch_v = torch.tensor(batch).to(device)
        Q_vals_v = net(batch_v)
        best_actionsQ_v = Q_vals_v.max(1)[0]
        mean_Q_vals.append(best_actionsQ_v.mean().item())
    return np.mean(mean_Q_vals)


def calc_loss_dqn(batch, net, target_net, gamma, device='cpu', double=False):
    states, actions, rewards, dones, last_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_mask_v = torch.ByteTensor(dones).to(device)
    last_states_v = torch.tensor(last_states).to(device)
    predicted_Q_v = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        # use the net to get actions
        # note max returns both max and argmax and we take the argmax
        next_state_actions_v = net(last_states_v).max(1)[1]
        next_state_Q_v = target_net(last_states_v).gather(
            1, next_state_actions_v.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_Q_v = target_net(last_states_v).max(1)[0]
    # use the dones_mask to 0 out values where last_state is none
    next_state_Q_v[dones_mask_v] = 0.0
    expected_Q_v = next_state_Q_v.detach() * gamma + rewards_v
    return nn.MSELoss()(predicted_Q_v, expected_Q_v)


class RewardTracker:

    def __init__(self, writer, stop_reward):
        self.stop_reward = stop_reward
        self.writer = writer

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self):
        self.writer.close()

    def reward(self, reward, frame_idx, epsilon=None):
        speed = (frame_idx - self.ts_frame) / (time.time() - self.ts)
        self.total_rewards.append(reward)
        # only use last 100 episodes to calculate mean reward
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame_idx, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar('epsilon', epsilon, frame_idx)
        self.writer.add_scalar('mean_reward', mean_reward, frame_idx)
        self.writer.add_scalar('speed', speed, frame_idx)
        self.writer.add_scalar('reward', np.mean(
            self.total_rewards), frame_idx)
        self.ts_frame = frame_idx
        self.ts = time.time()
        if mean_reward >= self.stop_reward:
            print('solved in %d frames' % frame_idx)
            return True
        else:
            return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        # we update the selector's epsilon directly here
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start -
                frame / self.epsilon_frames)

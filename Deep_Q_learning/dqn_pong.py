import torch
import collections
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn
import time
import torch.optim as optim
# import argparse

import wrappers
import model


ENV = 'PongNoFrameskip-v4'
MEAN_REWARD_BOUND = 19.5

# Training params
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4

# Target Network params
SYNC_TARGET_FRAMES = 1000
# Replay buffer params
REPLAY_BUFFER_SIZE = 10000
REPLAY_START_SIZE = 10000

# Epsilon-greedy params
EPSILON_MAX = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY_LAST_FRAME = 10**5

# create namedtuple to store experience
Experience = collections.namedtuple('Experience', field_names=[
                                    'state',
                                    'action',
                                    'reward',
                                    'done',
                                    'new_state'])


class ExperienceBuffer:

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size=BATCH_SIZE):
        batch_indices = np.random.choice(
            len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(
            *[self.buffer[idx] for idx in batch_indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(new_states)


class Agent:

    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_values_v = net(state_v)
            # print(q_values_v)
            _, action_v = torch.max(q_values_v, dim=1)
            action = int(action_v.item())

        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, done, new_state)
        self.buffer.append(exp)
        self.state = new_state
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calculate_loss(batch, net, target_net, device='cpu'):
    states, actions, rewards, dones, new_states = batch
    # vectorise all vars and push to device
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_mask = torch.ByteTensor(dones).to(device)
    new_states_v = torch.tensor(new_states).to(device)
    # gather requires indices in shape -> (num_indices, 1) so we use unsqueeze
    # along last dim and gather returns a tensor shaped -> (num_indices, 1) so
    # we use squeeze to reduce shape to (num_indices)
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    # max returns both max and argmax
    new_state_action_values = target_net(new_states_v).max(dim=1)[0]
    # make q_value for completed states to be 0
    new_state_action_values[dones_mask] = 0.0
    # detach new_states from computational graph to avoid backprop to the
    # target net
    new_state_action_values = new_state_action_values.detach()

    expected_q = rewards_v + GAMMA * new_state_action_values
    # MSE between Q(s,a) and (r + gamma * Q^*(s', a'))
    return nn.MSELoss()(state_action_values, expected_q)


if __name__ == '__main__':
    writer = SummaryWriter(comment='- DQN ' + ENV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device {}'.format(device))

    env = wrappers.make_env(ENV)
    net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = model.DQN(env.observation_space.shape,
                           env.action_space.n).to(device)
    print(net)

    buffer = ExperienceBuffer(capacity=REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_MAX

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    # some vars to measure speed
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    print_str = "%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s"
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_MIN, EPSILON_MAX -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])

            print(print_str % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(),
                           'checkpoints/' + ENV + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f" %
                          (best_mean_reward, mean_reward))
                    print("model saved")
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if frame_idx < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            print('Syncing Frames')
            target_net.load_state_dict(net.state_dict())
        # print('Backproping and updating net')
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calculate_loss(batch, net, target_net, device)
        loss_v.backward()
        optimizer.step()
    writer.close()
    env.env.close()

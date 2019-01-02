import ptan
import gym
from lib import common, model
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np

# additional hyperparams apart from the ones defined in lib/common.py
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:

    def __init__(self, exp_source, capacity, alpha):
        self.exp_source_iter = iter(exp_source)
        self.buffer = []
        # used to track number of elements
        self.pos = 0
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        # we will assign this max_prio to new elements in the buffer and later
        # update them with the loss
        max_prio = np.max(self.priorities) if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # convert priorities to probabilities and scale them by alpha
        probs = prios ** self.alpha
        probs /= probs.sum()
        # try with both False and True for replace
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        # since the samples with high priority will get oversampled we will
        # compensate this by weighting them less
        weights = (len(self.buffer) * prios[indices]) ** (-beta)
        # normalise the weights
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def calc_loss(batch, batch_weights, net, target_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_mask_v = torch.ByteTensor(dones).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    predicted_Q_v = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_Q_v = target_net(next_states_v).max(1)[0]
    # use the dones_mask to 0 out values where last_state is none
    next_state_Q_v[dones_mask_v] = 0.0
    expected_Q_v = next_state_Q_v.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (predicted_Q_v - expected_Q_v) ** 2

    return losses_v.mean(), losses_v + 1e-5


if __name__ == "__main__":

    params = common.HYPERPARAMS['pong']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(comment='PER ' + params['env_name'])

    env = gym.make(params['env_name'])
    # this doesn't normalise image so make sure to implement that in the
    # model part
    env = ptan.common.wrappers.wrap_dqn(env)
    net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device)

    experience_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, params['gamma'], steps_count=1)
    buffer = PrioReplayBuffer(
        experience_source, params['replay_size'], PRIO_REPLAY_ALPHA)

    optimizer = optim.Adam(net.parameters(), params['learning_rate'])
    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            # use to update selector's epsilon
            epsilon_tracker.frame(frame_idx)
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx *
                       (1.0 - BETA_START) / BETA_FRAMES)
            new_rewards = experience_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx,
                                         selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(
                params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch,
                                               batch_weights,
                                               net,
                                               target_net.target_model,
                                               params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(
                batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()

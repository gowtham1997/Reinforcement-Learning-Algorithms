"""
Utils for unpacking batches
"""
import numpy as np
import torch


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Unpacks batch into states, actions, rewards and last_states and sends them
    to specfied device

    Parameters
    ----------
    batch : list
        batch containing s,a,r,s' in a packed form
    net : torch nn.module
        a2c net
    last_val_gamma : int
        discount factor for the last state
    device : str, optional
        device used "cpu" or "cuda"

    Returns
    -------
    states_v : torch float32 Tensor
        states vector
    actions_v : torch float32 Tensor
        actions vector
    ref_vals_v : torch float32 Tensor
        reference value vector
    """
    states, actions, rewards, last_states, not_dones = [], [], [], [], []

    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)

        if exp.last_state is not None:
            last_states.append(exp.last_state)
            not_dones.append(idx)

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)

    if not_dones:
        last_states_v = torch.FloatTensor(last_states).to(device)
        # net returns mu, sigma and value
        last_state_values_v = net(last_states_v)[2]
        last_state_values_np = last_state_values_v.data.cpu().numpy()[:, 0]
        rewards_np[not_dones] += last_state_values_np * last_val_gamma

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_v, ref_vals_v

"""Different algorithms(A2C, DDPG, etc) and their model definitions

Attributes
----------
HID_SIZE : int
The hidden size for the fully connected layers
"""
import ptan
import torch
import numpy as np
import torch.nn as nn

HID_SIZE = 128


class ModelA2C(nn.Module):

    """Defines the A2C model

    Attributes
    ----------
    fc : torch nn Sequential
    Fully connected layer
    mu : torch nn Sequential
    Fully connected layer with tanh to output mean
    value : torch nn Sequential
    Fully connected layer with Softplus to output variance
    var : torch nn Sequential
    Fully connected layer to output value of a state
    """

    def __init__(self, obs_size, action_size):
        """Constructor

        Parameters
        ----------
        obs_size : tuple of ints
            shape of the obs(inputs)
        action_size : tuple of ints
            shape of the actions(outputs)
        """
        self.fc = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU())
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, action_size),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, action_size),
            nn.Softplus()
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, obs):
        """Forward function of the net

        Parameters
        ----------
        obs : torch FloatTensor
            observations(inputs)

        Returns
        -------
        torch FloatTensor
            The mean and variance of the action, and value for the state
        """
        fc_out = self.fc(obs)
        return self.mu(fc_out), self.var(fc_out), self.value(fc_out)


class AgentA2C(ptan.agent.BaseAgent):

    """Agent which takes a net and samples actions using mu, sigma from normal
    distribution

    Attributes
    ----------
    device : string
    The device to move or perform the operations on
    net : torch nn Module
    The network
    """

    def __init__(self, net, device="cpu"):
        """Constructor

        Parameters
        ----------
        net : torch nn Module
        The network
        device : string
        The device to move or perform the operations on
        """
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        """Takes states and gets mu, sigmas which are used to sample actions
        from the normal distribution

        Parameters
        ----------
        states : list of floats
            obserations
        agent_states : TYPE
            Description

        Returns
        -------
        np array of floats
            sampled actions
        """
        states_v = torch.FloatTensor(states).to(self.device)
        mu_v, var_v, _ = self.net(states_v)
        mu_np = mu_v.data.cpu().numpy()
        sigma_np = torch.sqrt(var_v).data.cpu().numpy()
        # sample from normal distribution
        actions = np.random.normal(mu_np, sigma_np)
        actions = np.clip(-1, 1)

        return actions, agent_states

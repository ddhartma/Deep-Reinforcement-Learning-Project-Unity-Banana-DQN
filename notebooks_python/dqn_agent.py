import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import operator

BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate
UPDATE_EVERY = 4            # how often to update the network
PER_ALPHA = 0.6             # importance sampling exponent
PER_BETA = 0.4              # prioritization exponent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weighted_mse_loss(input, target, weights):
    """ Return the weighted mse loss to be used by Prioritized experience replay

        INPUTS:
        ------------
            input - (torch.Tensor)
            target - (torch.Tensor)
            weights - (torch.Tensor)

        OUTPUTS:
        ------------
            loss - (torch.Tensor)
    """

    # source: http://
    # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
    out = (input-target)**2
    out = out * weights.expand_as(out)
    loss = out.mean(0)  # or sum over whatever dimensions

    return loss


class DQNAgent():
    """ Interacts with and learns from the environment
    """

    def __init__(self, state_size, action_size, seed, state_dict=None):
        """ Initialize an Agent object.

            INPUTS:
            ------------
                state_size - (int) dimension of each state
                action_size - (int) dimension of each action
                seed - (int) random seed

            OUTPUTS:
            ------------
                no direct
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        print("Local Netwrok: ")
        print(self.qnetwork_local)
        print("Target Netwrok: ")
        print(self.qnetwork_target)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        # In case of a trained model
        if state_dict != None:
            weights = torch.load(state_dict)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)


        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)


        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

            INPUTS:
            ------------
                state - (numpy array_like) the previous state of the environment (37,)
                action - (int) the agent's previous choice of action
                reward - (float) last reward received
                next_state - (torch tensor) the current state of the environment
                done - (bool) whether the episode is complete (True or False)

            OUTPUTS:
            ------------
                no direct
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                #print('experiences')
                #print(experiences)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ Returns actions for given state as per current policy.

            INPUTS:
            ------------
                state - (numpy array_like) current state
                eps - (float) epsilon, for epsilon-greedy action selection

            OUTPUTS:
            ------------
                act_select - (int) next epsilon-greedy action selection
        """

        # Convert state from numpy array to torch tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set model to evaluation (no optimizer step, no backpropagation), get action values as Fixed Targets
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Set model back to train, use these Fixed Targets to initiate an epsilon greedy action selection
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            act_select = np.argmax(action_values.cpu().data.numpy())
            return act_select
        else:
            act_select = random.choice(np.arange(self.action_size))
            return act_select

    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.

            INPUTS:
            ------------
                experiences - (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples
                gamma - (float) discount factor

            OUTPUTS:
            ------------
                no direct
        """

        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #print(Q_targets_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            INPUTS:
            ------------
                local_model - (PyTorch model) weights will be copied from
                target_model - (PyTorch model) weights will be copied to
                tau - (float) interpolation parameter

            OUTPUTS:
            ------------
                no direct

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DDQNAgent(DQNAgent):
    """ Implementation of a DDQN agent that interacts with and learns from the environment
    """

    def __init__(self, state_size, action_size, seed, state_dict=None):
        """ Initialize an DoubleDQNAgent object

           INPUTS:
            ------------
                state_size - (int) dimension of each state
                action_size - (int) dimension of each action
                seed - (int) random seed

            OUTPUTS:
            ------------
                no direct
        """
        super(DDQNAgent, self).__init__(state_size, action_size, seed, state_dict=None)

        # In case of a trained model
        if state_dict != None:
            weights = torch.load(state_dict)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples

            INPUTS:
            ------------
                experiences - (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples
                gamma - (float) discount factor

            OUTPUTS:
            ------------
                no direct
        """

        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss

        # arg max_{a} \hat{Q}(s_{t+1}, a, θ_t)
        argmax_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        # Q_targets_next :=  \hat{Q}(s_{t+1}, argmax_actions, θ^−)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions)
        #print(Q_targets_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)


        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class PRBDDQNAgent(DQNAgent):
    """ Implementation of a DDQN agent that used prioritized experience replay
    """

    def __init__(self, state_size, action_size, seed, state_dict=None, alpha=PER_ALPHA, initial_beta=PER_BETA, max_t=1000):
        """ Initialize an DDQNPREAgent object

            INPUTS:
            ------------
                state_size - (int) dimension of each state
                action_size - (int) dimension of each action
                seed - (int) random seed
                alpha - (float) importance sampling exponent
                initial_beta - (float) prioritization exponent
                max_t - (int) maximum time step

            OUTPUTS:
            ------------
                no direct
        """
        super(PRBDDQNAgent, self).__init__(state_size, action_size, seed, state_dict=None)

        # In case of a trained model
        if state_dict != None:
            weights = torch.load(state_dict)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size,
                                              BUFFER_SIZE,
                                              BATCH_SIZE,
                                              seed,
                                              alpha)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.alpha = alpha
        self.initial_beta = initial_beta
        self.max_t = max_t
        self.t_step = 0

    def get_beta(self, t):
        """ Return the current exponent β based on its schedul. Linearly anneal β
            from its initial value β0 to 1, at the end of learning.

            INPUTS:
            ------------
                t - (int) Current time step in the episode

            OUTPUTS:
            ------------
                current_beta - (float) Current exponent beta
        """
        f_frac = min(float(t) / self.max_t, 1.0)
        current_beta = self.initial_beta + f_frac * (1. - self.initial_beta)
        return current_beta

    def learn(self, experiences, gamma, t=1000):
        """ Update value parameters using given batch of experience tuples.

            INPUTS:
            ------------
                experiences - (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples
                gamma - (float) discount factor
                t - (int) current time step in the episode

            OUTPUTS:
            ------------
                no direct
        """
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #print(Q_targets_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #print(Q_expected)

        # compute importance-sampling weight wj
        f_currbeta = self.get_beta(t)
        weights = self.memory.get_is_weights(current_beta=f_currbeta)

        # compute TD-error δj and update transition priority pj
        td_errors = Q_targets - Q_expected
        self.memory.update_priorities(td_errors)

        # perform gradient descent step
        # Accumulate weight-change ∆←∆+wj x δj x ∇θQ(Sj−1,Aj−1)
        # loss = F.mse_loss(Q_expected*weights, Q_target*weights)
        loss = weighted_mse_loss(Q_expected, Q_targets, weights)
        # loss = F.mse_loss(Q_expected, Q_target)*weights
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize a ReplayBuffer object.

        INPUTS:
        ------------
            action_size - (int) dimension of each action
            buffer_size - (int) maximum size of buffer
            batch_size - (int) size of each training batch
            seed - (int) random seed

        OUTPUTS:
        ------------
            no direct
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory.

            INPUTS:
            ------------
                state - (array_like) the previous state of the environment
                action - (int) the agent's previous choice of action
                reward - (int) last reward received
                next_state - (int) the current state of the environment
                done - (bool) whether the episode is complete (True or False)

            OUTPUTS:
            ------------
                no direct

        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory.

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                states - (torch tensor) the previous states of the environment
                actions - (torch tensor) the agent's previous choice of actions
                rewards - (torch tensor) last rewards received
                next_states - (torch tensor) the next states of the environment
                dones - (torch tensor) bools, whether the episode is complete (True or False)

        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Return the current size of internal memory.

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                mem_size - (int) current size of internal memory

        """
        mem_size = len(self.memory)
        return mem_size

class PrioritizedReplayBuffer(object):
    ''' Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """ Initialize a PrioritizedReplayBuffer object

            INPUTS:
            ------------
                action_size - (int) dimension of each action
                buffer_size - (int) maximum size of buffer
                batch_size - (int) size of each training batch
                seed - (int) random seed
                alpha - (float) prioritized replay buffer hyperparameter, 0~1 indicating how much prioritization is used

            OUTPUTS:
            ------------
                no direct
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        # specifics
        self.alpha = max(0., alpha)  # alpha should be >= 0
        self.priorities = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self.cum_priorities = 0.
        self.eps = 1e-6
        self._indexes = []
        self.max_priority = 1.**self.alpha

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory

            INPUTS:
            ------------
                state - (array_like) the previous state of the environment
                action - (int) the agent's previous choice of action
                reward - (int) last reward received
                next_state - (int) the current state of the environment
                done - (bool) whether the episode is complete (True or False)

            OUTPUTS:
            ------------
                no direct

        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # exclude the value that will be discareded
        if len(self.priorities) >= self._buffer_size:
            self.cum_priorities -= self.priorities[0]
        # include the max priority possible initialy
        self.priorities.append(self.max_priority)  # already use alpha
        # accumulate the priorities abs(td_error)
        self.cum_priorities += self.priorities[-1]

    def sample(self):
        """ Sample a batch of experiences from memory according to importance-sampling weights

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                tuple[torch.Tensor]. Sample of past experiences
        """
        i_len = len(self.memory)
        na_probs = None
        if self.cum_priorities:
            na_probs = np.array(self.priorities)/self.cum_priorities
        l_index = np.random.choice(i_len,
                                   size=min(i_len, self.batch_size),
                                   p=na_probs)
        self._indexes = l_index

        experiences = [self.memory[ii] for ii in l_index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def _calculate_is_w(self, f_priority, current_beta, max_weight, i_n):
        #  wi= ((N x P(i)) ^ -β)/max(wi)
        f_wi = (i_n * f_priority/self.cum_priorities)
        return (f_wi ** -current_beta)/max_weight

    def get_is_weights(self, current_beta):
        """ Return the importance sampling (IS) weights of the current sample based on the beta passed

            INPUTS:
            ------------
                current_beta - (float) fully compensates for the non-uniform probabilities P(i) if β = 1

            OUTPUTS:
            ------------
                torch tensor of is_weights
        """
        # calculate P(i) to what metters
        i_n = len(self.memory)
        max_weight = (i_n * min(self.priorities) / self.cum_priorities)
        max_weight = max_weight ** -current_beta

        this_weights = [self._calculate_is_w(self.priorities[ii],
                                             current_beta,
                                             max_weight,
                                             i_n)
                        for ii in self._indexes]



        return torch.tensor(this_weights,
                            device=device,
                            dtype=torch.float).reshape(-1, 1)

    def update_priorities(self, td_errors):
        """ Update priorities of sampled transitions inspiration: https://bit.ly/2PdNwU9

            INPUTS:
            ------------
                td_errors - (tuple of torch.tensors) TD-Errors of last samples

            OUTPUTS:
            ------------
                no direct
        """

        for i, f_tderr in zip(self._indexes, td_errors):
            f_tderr = float(f_tderr)
            self.cum_priorities -= self.priorities[i]
            # transition priority: pi^α = (|δi| + ε)^α
            self.priorities[i] = ((abs(f_tderr) + self.eps) ** self.alpha)
            self.cum_priorities += self.priorities[i]
        self.max_priority = max(self.priorities)
        self._indexes = []

    def __len__(self):
        """ Return the current size of internal memory

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                len(self.memory) - (int) Length of accumulated memory
        """
        return len(self.memory)

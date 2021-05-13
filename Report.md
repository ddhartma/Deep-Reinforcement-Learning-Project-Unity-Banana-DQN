[image1]: assets/scoring_result.png "image1"

# Deep Reinforcement Learning Project - Unitiy-Banana-DQN - Implementation Report

## Content
- [Implementation - Navigation_Training.ipynb](#impl_notebook_train)
- [Implementation - dqn_agent.py](#impl_agent)
- [Implementation - model.py](#impl_model)
- [Implementation - Navigation_Trained_Agent.ipynb](#impl_notebook_trained_agent)
- [Ideas for future work](#ideas_future)

## Implementation - Navigation_Training.ipynb <a name="impl_notebook_train"></a>
- Open jupyter notebook file ```Navigation_Training.ipynb```
    ### Import important libraries
    - modul ***unityagents*** provides the Unity Environment. This modul is part of requirements.txt. Check the README.md file for detailed setup instructions.
    - modul **dqn_agent** contains the implementation of an DQN agent. Check the description of **dqn_agent.py** for further details. 
    ```
    import random
    import torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    from unityagents import UnityEnvironment
    from dqn_agent import Agent
    ```
    ### Instantiate the Environment
    - Load the UnityEnvironment and store it in **env**
    - Environments contain brains which are responsible for deciding the actions of their associated agents.
    ```
    env = UnityEnvironment(file_name="Banana.app")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    ```
    ### Instantiate the Agent
    ```
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    ```
    ### Watch an untrained agent
    - **state** - (numpy arrray) with 37 float values, contains the **agent's velocity**, along with **ray-based perception of objects** around the agent's forward direction
    - **action** - (int) four actions for four directions (0...3) are possible. See README.md for further details
    - **env_info** - (unityagent instance)
    - **next_state** - (numpy array) next state (here chosen by random action)
    - **reward** - (int/float) +1 is provided for collecting a yellow banana and -1 is provided for collecting a blue banana
    - **done** - (bool) if True episode is over
    - **score** - (float) cumulative reward, scoring after each action
    ```
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = np.random.randint(action_size)        # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break

    print('state')
    print(state)
    print()
    print('action ', action)  
    print()
    print('env_info')
    print(env_info)
    print()
    print('next_state')
    print(next_state)
    print()
    print('reward')
    print(reward)
    print()
    print('done')
    print(done)
    print()
    print("Score: {}".format(score))

    RESULT:
    -------------
    state
    [0.         1.         0.         0.         0.21772696 1.
    0.         0.         0.         0.07809805 0.         0.
    1.         0.         0.08886749 1.         0.         0.
    0.         0.05297319 1.         0.         0.         0.
    0.1608755  1.         0.         0.         0.         0.05958167
    0.         0.         0.         1.         0.         0.
    0.        ]

    action  0

    env_info
    <unityagents.brain.BrainInfo object at 0x116ff4208>

    next_state
    [0.         1.         0.         0.         0.21772696 1.
    0.         0.         0.         0.07809805 0.         0.
    1.         0.         0.08886749 1.         0.         0.
    0.         0.05297319 1.         0.         0.         0.
    0.1608755  1.         0.         0.         0.         0.05958167
    0.         0.         0.         1.         0.         0.
    0.        ]

    reward
    0.0

    done
    True

    Score: 0.0
    ```
    ### Start Training
    - function which implements the agent training
    - 2 for loops: outer loop --> loop over episodes and inner loop --> loop over timesteps per episode (TD learning algorithm)
        - for the actual episode:
            - reset the environment
            - get the current state
            - initialize the score 
            - for the actual time step:
                - return actions for current state and policy
                - send the action to the environment
                - get the next state
                - get the reward 
                - see if episode has finished
                - update the agent's knowledge, using the most recently sampled tuple
            - save most recent score
            - decrease epsilon, turn slowly from an equiprobable action choice of exploration to steadily increasing greedy exploration with each episode
    ```
    def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """ Deep Q-Learning.
        
            INPUTS: 
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                eps_start - (float) starting value of epsilon, for epsilon-greedy action selection
                eps_end - (float) minimum value of epsilon
                eps_decay - (float) multiplicative factor (per episode) for decreasing epsilon
                
            OUTPUTS:
            ------------
                scores - (list) list of scores
        """
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
            state = env_info.vector_observations[0]                 # get the current state
            score = 0                                               # initialize the score
            for t in range(max_t):
                action = agent.act(state, eps)                      # return actions for current state and policy
                env_info = env.step(action)[brain_name]             # send the action to the environment
                next_state = env_info.vector_observations[0]        # get the next state
                reward = env_info.rewards[0]                        # get the reward
                done = env_info.local_done[0]                       # see if episode has finished
                agent.step(state, action, reward, next_state, done) # update the agent's knowledge
                state = next_state                                  # set next state as current state 
                score += reward                                     # update score with the return for this time step
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores

    scores = dqn()
    ```
    ### Plot the cumulated scores as a function of episodes
    ```
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```
    ![image1]
    ### Close the environment
    ```
    env.close()
    ```

## Implementation - dqn_agent.py <a name="impl_agent"></a>
- Open Python file ```dqn_agent.py```
    ### Load important libraries
    ```
    import numpy as np
    import random
    from collections import namedtuple, deque

    from model import QNetwork

    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    ```
    ### Hyperparameters
    ```
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
    ### Main class 'Agent' to train an agent getting smart
    - **init** function:
        - design the QNetwork based on the numbers state_size=37 and action_size=4, see model.py for more information
        - use Adam optimizer (minibatch SGD with adaptive learning rate, momentum)
        - Initialize Replaybuffer: for storing each experienced tuple in this buffer. This buffer helps to reduce correlation between consecutive experience tuples, by random sampling of experiences from this buffer

    - **step** function:
        - update the agent's knowledge, using the most recently sampled tuple
        - save this experience in replay memory **self.memory**
        - structure of **experiences** (tuple of torch tensors):
            ```
            (
                tensor([[37x floats for state], x 64 for minibatch]),
                tensor([[1x int for action], x 64 for minibatch]),
                tensor([[1x float for reward], x 64 for minibatch]),
                tensor([[37x floats for next_state], x 64 for minibatch]),
                tensor([[1x int for done], x 64 for minibatch])
            )
            ```
        - learn every UPDATE_EVERY time steps (by doing random sampling from Replaybuffer)
    
    - **act** function:
        - returns actions for given state as per current policy
        - convert state as numpy arrray to torch tensor
        - set model to evaluation mode (no optimizer step, no backpropagation), get action values as Fixed Targets
        - set model back to train mode, use these Fixed Targets to initiate an epsilon greedy action selection

    - **learn** function:
        - update value parameters using given batch of experience tuples
        - compute and minimize the loss
        - get max predicted Q values (for next states) from target model

            ```
            RESULT structure for Q_targets_next:
            tensor([[1x float for Q value], x 64 for minibatch])
            ```
        - compute Q targets for current states 
            ```
            RESULT structure for Q_targets:
            tensor([[1x float for Q value], x 64 for minibatch])
            ```
        - get expected Q values from local model
            RESULT structure for Q_expected:
            ```
            tensor([[1x float for Q value], x 64 for minibatch])
            ```
        - compute loss
        - minimize loss
        - update target network, see soft_update function

    - **soft_update** function:
        - soft update model parameters
    ```
    class Agent():
        """ Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, seed):
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
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

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
                    next_state - (array_like) the current state of the environment
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
            """
            states, actions, rewards, next_states, dones = experiences

            # Compute and minimize the loss

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            print(Q_targets_next)
            
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            print(Q_targets)

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            print(Q_expected)

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

    ```
    ### ReplayBuffer class to reduce consecutive experience tuple correlations
    - **init** function: initialize a ReplayBuffer object
    - **add** function: add an experience tuple to self.memory object
    - **sample** function: 
        - choose per random one **experiences** tuple with batch_size experiences
        - return states, actions, rewards, next_states and dones each as torch tensors
    - **__len__**: return the current size of internal memory
    ```
    class ReplayBuffer:
        """ Fixed-size buffer to store experience tuples."""

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
    ```

## Implementation - model.py <a name="impl_model"></a>
- Open Python file ```model.py```
    ### Import important libraries
    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ```
    ### Create a Pytorch model as a deep QNetwork
    - **init** function: Initialize parameters and build model
    - **forward** function: create a forward pass, i.e. build a network that maps state -> action values
    - **Architecture**:
        - Three fully connected layers
            - 1st hidden layer: fully connected, 37 input unit units, 64 output units, rectified via ReLU
            - 2nd hidden layer: fully connected, 64 input unit units, 64 output units, rectified via ReLU
            - 3rd hidden layer: fully connected, 64 input unit units, 4 output units
    ```
    class QNetwork(nn.Module):
        """ Actor (Policy) Model
        """

        def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
            """ Initialize parameters and build model.
                
                INPUTS:
                ------------
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
                    seed (int): Random seed
                    fc1_units (int): Number of nodes in first hidden layer
                    fc2_units (int): Number of nodes in second hidden layer
                    
                OUTPUTS:
                ------------
                    no direct
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
           
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            

        def forward(self, state):
            """ Build a network that maps state -> action values.
                
                INPUTS:
                ------------
                    state - (array-like) actual 
                    
                OUTPUTS:
                ------------
                    output - (array-like) action values for given state set
            """
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            output = self.fc3(x)
            return output
    ```
## Implementation - Navigation_Trained_Agent.ipynb <a name="impl_notebook_trained_agent"></a> 
- Open Jupyter Notebook ```Navigation_Trained_Agent.ipynb```
    ### Import important libraries
    - modul ***unityagents*** provides the Unity Environment. This modul is part and installed via requirements.txt. Check the README.md file for detailed setup instructions.
    - modul **dqn_agent** is the own implementation of an DQN agent. Check the description of **dqn_agent.py** for further details. 
    ```
    import random
    import torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    from unityagents import UnityEnvironment
    from dqn_agent import Agent
    ```
    ### Instantiate the Environment
    ```
    # Load the Unity environment
    env = UnityEnvironment(file_name="Banana.app")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get an instance of an agent from the Agent class (see module dqn_agent)
    agent = Agent(state_size=37, action_size=4, seed=0)

    # Load the weights from the pytorch state_dict file checkpoint.pth
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score
    ```
    ### Watch a smart agent in action
    ```
    while True:
        action = agent.act(state)                      # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))
    ```

## Ideas for future work <a name="ideas_future"></a> 
- Implement Deep Q-Learning Improvements like:
    - [Double Q-Learning](https://arxiv.org/abs/1509.06461): Deep Q-Learning [tends to overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) action values. In early stages, the Q-values are still evolving. This can result in an overestimation of Q-values, since the maximum values are chosen from noisy numbers. Solution: Select the best action using one set of weights w, but evaluate it using a different set of weights w'. It's basically like having two separate function approximators.

    - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

    - [Dueling DQN](https://arxiv.org/abs/1511.06581): Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action. The core idea of dueling networks is to use two streams, one that estimates the state value function and one that estimates the advantage for each action.

    - [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298): A Rainbow DQN algorithm combines the upper three modificartions (Double Q-Learning, Prioritized Experience Replay, Dueling DQN) together with:
        - Learning from [multi-step bootstrap targets](https://arxiv.org/abs/1602.01783) 
        - [Distributional DQN](https://arxiv.org/abs/1707.06887)
        - [Noisy DQN](https://arxiv.org/abs/1706.10295)

- Further Readings for DQN optimizations:
    - [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://shmuma.medium.com/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)
    - [Advanced DQNs: Playing Pac-man with Deep Reinforcement Learning by mapping pixel images to Q values](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814)
    - Interesting GitHub repo based on [Prioritized Experience Replay](https://github.com/rlcode/per)
    - [Conquering OpenAI Retro Contest 2: Demystifying Rainbow Baseline](https://medium.com/intelligentunit/conquering-openai-retro-contest-2-demystifying-rainbow-baseline-9d8dd258e74b)
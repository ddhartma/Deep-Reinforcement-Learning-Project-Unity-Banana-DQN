[image1]: assets/banana_agent.gif "image1"

# Deep Reinforcement Learning Project - Unitiy-Banana-DQN

## Content
- [Introduction](#intro)
- [Unity Environment](#unitity_env)
- [Files in the Repo](#files_in_repo)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a id="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

- This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=12906460312_c&utm_term=121838875579&utm_keyword=deep%20reinforcement%20udacity_e&gclid=CjwKCAjw-e2EBhAhEiwAJI5jg7Ycb934lFlosCFVpvwKRD_U5ESjMX18faGkkTTUkIyZVJ6yU4HkohoCyfIQAvD_BwE) for more information.

## Unity Environment <a id="unitity_env"></a>
- [Unity Machine Learning Agents (ML-Agents)](https://github.com/Unity-Technologies/ml-agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. 
- Implementations (based on PyTorch) of state-of-the-art algorithms to enable game developers and hobbyists to easily train intelligent agents for 2D, 3D and VR/AR games
- For this project, an agent to navigate (and collect bananas!) in a large, square world will be trained.
- A **reward** of 
    - +1 is provided for collecting a yellow banana
    - -1 is provided for collecting a blue banana. 
- **Goal**: Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.
- The **state space** has **37 dimensions** and contains the **agent's velocity**, along with **ray-based perception of objects** around the agent's forward direction. Given this information, the agent has to learn how to best select actions. 
- **Four discrete actions** are available:
    - 0 - move forward
    - 1 - move backward
    - 2 - turn left
    - 3 - turn right

    ```
    INFO:unityagents:
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
            
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 
    ```
- The task is **episodic**, and in order to solve the environment, your agent **must get an average score of +13 over 100 consecutive episodes**.

    ![image1]

## Files in the repo <a id="files_in_repo"></a>
The workspace contains the following files:
- **README.md**: Markdown file, the readme of this repo. 
- **Report.md**: Markdown file, a detailed description of the code implementation.
- **/notebooks_python/Navigation_Training.ipynb**: Main notebook file to implement DQN and to train the agent.
- **/notebooks_python/Navigation_Trained_Agent.ipynb**: Second notebook file to watch the behaviour of a trained agent.
- **/notebooks_python/dqn_agent.py**: Python file containing the implementation of the deep reinforcement learning agent.
- **/notebooks_python/model.py**: Python file containing the PyTorch Deep Learning model,  a deep neural network that acts as a function approximator, i.e. it defines a neural network architecture that maps states to action values.
- **/notebooks_python/checkpoint_dqn.pth**: PyTorch state_dict file containing trained weights for simple DQN.
- **/notebooks_python/checkpoint_ddqn.pth**: PyTorch state_dict file containing trained weights for Double DQN.
- **/notebooks_python/checkpoint_prbddqn.pth**: PyTorch state_dict file containing trained weights for PrioritizedReplayBuffer Double DQN.
- **/notebooks_python/Banana.app/Contents**: Downloaded Unity Environment needed to watch a trained agent.
- **/notebooks_python/unity-environment.log**: Log file repoting the last interaction with the environment.


## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python - 64 Bit

- Upgrade Anaconda via
    ```
    $ conda upgrade conda
    $ conda upgrade --all
    ```

- Optional: In case of trouble add Anaconda to your system path. Write in your Command Line Interface (CLI)
    ```
    $ export PATH="/path/to/anaconda/bin:$PATH"
    ```

### Clone the project <a id="Clone_the_project"></a>
- Open your CLI
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder via:
    ```
    $ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Project-Unity-Banana-DQN.git
    ```

- Change Directory
    ```
    $ cd Unitiy-Banana-DQN-Project
    ```

### Create (and activate) a new environment:
- Create a Python 3.6 environment via conda
    ```
    $ conda create --name drlnd python=3.6
    $ conda activate drlnd
    ```

- To install ```requirements.txt``` in the new environment, use **pip** installed within the environment. Install pip first by
    ```
    $ conda install pip
    ```

- Install all packages provided in ```requirements.txt``` (via pip) needed to train and watch a smart agent
    ```
    $ pip install -r requirements.txt
    ```

- Check the environment installation via
    ```
    $ conda list
    ```
### Download the Unity environment
- You need to select the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   -  Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   -  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

- Unzip this file and place it in ```/Unitiy-Banana-DQN-Project````
- Replace **<file_name>** in the following line of ```Navigation_Training.ipynb``` and ```Navigation_Trained_Agent.ipynb```
    ```
    env = UnityEnvironment(file_name="<file_name>"
    ```
    with the file name of the Unity evironment (**Banana...**).
    Save the notebooks.

### To Start Agent Training
- Navigate via CLI to ```Navigation_Training.ipynb```
- Type in terminal
    ```
    jupyter notebook Navigation_Training.ipynb
    ```
- Run each cell in the notebook to train the agent.

### To Watch a Smart Agent
- Navigate via CLI to ```Navigation_Trained_Agent.ipynb```
- Type in terminal
    ```
    jupyter notebook Navigation_Trained_Agent.ipynb
    ```
- Run each cell in the notebook to watch a trained and smart agent.

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>
Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

Optimize DQNs:
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

    - Repo1 based on a Navigation solution including DQN optimizations [silviomori](https://github.com/silviomori/udacity-deep-reinforcement-learning-p1-navigation)

    - Repo2 based on a Navigation solution including DQN optimizations [vaanxy](https://github.com/vaanxy/drlnd-p1-navigation)
    - Repo2 based on a Navigation solution including DQN optimizations [ucaiado](https://github.com/ucaiado/banana-rl)
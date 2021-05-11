[image1]: assets/banana_agent.gif "image1"

# Deep Reinforcement Learning Project - Unitiy-Banana-DQN

## Content
- [Introduction](#intro)
- [Unity Environment](#unitity_env)
- [Files in the Repo](#files)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## Unity Environment <a name="unitity_env"></a>
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

## Files in the repo <a name="files"></a>
The workspace contains three files:
- **Navigation_Training.ipynb**: Main notebook file to implement DQN and traiin the agent.
- **Navigation_Trained_Agent.ipynb**: Second notebook file to watch the behaviour of a trained agent.
- **dqn_agent.py**: Python file to develop the reinforcement learning agent.
- **model.py**: Contains the pytorch Deep Learning model,  a deep neural network that acts as a function approximator, i.e. it define a neural network architecture that maps states to action values.
- **checkpoint.pth**: Pytorch state_dict file containing trained weights.
- **Banana.app**: Downloaded Unity Environment needed to watch a trained agent.
- **unity-environment.log**: Log file repoting the last interaction with the environment.
- **Report.md**: Markdown file with a detailed description of the implementation.

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
    ```
    $ conda upgrade conda
    $ conda upgrade --all
    ```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
    ```
    $ export PATH="/path/to/anaconda/bin:$PATH"
    ```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
    ```
    $ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Project-Unity-Banana-DQN.git
    ```

- Change Directory
    ```
    $ cd Unitiy-Banana-DQN-Project
    ```

### Create (and activate) a new environment:
- Use Python 3.6 for the environment
    ```
    $ conda create --name drlnd python=3.6
    $ conda activate drlnd
    ```

- To install ```requirements.txt``` in the new environment, use the **pip** installed within the environment. Thus we should install pip first by
    ```
    $ conda install pip
    ```

- Install the all packages in ```requirements.txt``` (via pip and conda) needed to train and watch a smart agent
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
    with the file name of the Unity evironment (**Banana...**)
- Save the notebooks

### To Start Agent Training
- Navigate via CLI to ```Navigation_Training.ipynb```
- Type in terminal
    ```
    jupyter notebook Navigation_Training.ipynb
    ```
- Run each cell in the notebook to train the agent

### To Watch a Smart Agent
- Navigate via CLI to ```Navigation_Trained_Agent.ipynb```
- Type in terminal
    ```
    jupyter notebook Navigation_Trained_Agent.ipynb
    ```
- Run each cell in the notebook to train the agent

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
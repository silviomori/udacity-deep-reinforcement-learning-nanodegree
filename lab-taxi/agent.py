import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.alpha = 0.01
        self.gamma = 1.0

        # These values have reached:
        # Best average reward 9.0943
        #self.eps = 0.2
        #self.eps_decay = 0.99
        #self.eps_min = 0.01

        # These values have reached:
        # Best average reward 9.1285
        # self.eps = 1
        # self.eps_decay = 0.66
        # self.eps_min = 0.001

        # These values have reached:
        # Best average reward 9.1343        
        self.eps = 1
        self.eps_decay = 0.5
        self.eps_min = 0.001

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        actions = self.Q[state]
        
        # Identify the greedy action
        greedy_action = np.argmax(actions)
        
        # Initialize equiprobable policy
        policy = np.ones(self.nA) * (self.eps / self.nA)
        
        # Emphasizes the greedy action
        policy[greedy_action] += 1 - self.eps
        
        # Choose the next action
        action = np.random.choice(np.arange(self.nA), p=policy)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        ## Calculate the expected reward
        # Calculate the reward for the equiprobable policy
        next_actions = self.Q[next_state]
        expected_reward = sum(next_actions * (self.eps / len(next_actions)))
        
        # Identify the greedy action
        greedy_action = np.argmax(next_actions)
        
        # Add the reward for the greedy action
        expected_reward += next_actions[greedy_action] * (1 - self.eps)

        ## Calculate the reward
        reward += (self.gamma * expected_reward) - self.Q[state][action]

        ## Update Q-Table
        self.Q[state][action] += self.alpha * reward
        
        
        if done:
            # Decay epsilon after each episode
            self.eps = max(self.eps*self.eps_decay, self.eps_min)

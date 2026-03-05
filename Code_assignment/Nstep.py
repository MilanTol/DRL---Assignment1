#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
    
    def _single_update(self, states, actions, rewards, done):
        """
        updates Q_sa for a single state and action, using Nstep. 
        """
        n = len(actions) # store total number of actions taken
        gammas = self.gamma**np.arange(n) #create array containing [gamma^0, gamma^1, ..., gamma^{n-1}]
        sum_term = np.sum(gammas*rewards) #compute estimated total reward for current policy up till t + (n-1)

        #if final state is terminal, dont add estimated future reward after final state
        if done:
            target = sum_term 
        else:
            target = sum_term + self.gamma**n * np.max(self.Q_sa[states[n]])

        self.Q_sa[states[0], actions[0]] = (
            self.Q_sa[states[0], actions[0]] + 
            self.learning_rate * (target - self.Q_sa[states[0], actions[0]])
            )
        return 
    
    def update(self, states, actions, rewards, done, T_ep):
        ''' 
        states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state 
        '''
        for T in range(T_ep):
            self._single_update(states[T:], actions[T:], rewards[T:], done=done) #state always contains 1 more element
        return 
           

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an Nstep rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    t=0
    while t < n_timesteps:    
        t+=1
        s = env._location_to_state(env.start_location)

        states = []
        states.append(s)
        actions = []
        rewards = []
        #collect episode
        for T in range(max_episode_length):
            a = agent.select_action(s, policy, epsilon=epsilon, temp=temp)
            #overwrite state s 
            s, r, done = env.step(a) 
            states.append(s)
            actions.append(a)
            rewards.append(r)
            if done: #check whether agent reached terminal state, then end episode
                break
        T_ep = T 
        agent.update(states, actions, rewards, done=done, T_ep=T_ep)
            
        if plot:
            env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
            
        if t % eval_interval == 0:
            eval_timesteps.append(t)
            eval = agent.evaluate(eval_env)
            eval_returns.append(eval) #use evaluation environment to not affect the testing environment!

    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.3
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,  
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()

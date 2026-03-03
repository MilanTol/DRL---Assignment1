#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states:int, n_actions:int, gamma, threshold=0.01):
        self.n_states = n_states 
        self.n_actions = n_actions
        self.gamma = gamma #discount rate ([0,1]): quantifies how much future rewards are valued.
        self.Q_sa = np.zeros((n_states,n_actions)) #instantiates table containing all Q(s,a) vals
        
    def select_action(self, s:int) -> int:
        ''' 
        Returns the greedy best action in state s 
        
            s (int): integer corresponding to state    
        ''' 
        # greedy policy is a discrete policy that only considers the best action:
        a = argmax(self.Q_sa[s]) # Replace this with correct action selection
        return a
        
    def update(self, s:int, a:int, p_sas:np.ndarray, r_sas:np.ndarray) -> None:
        ''' 
        Function updates Q(s,a) using p_sas and r_sas 
        p_sas must have a particular ordering!
            p_sas: shape (n_states,) -> p(s'|s,a)
            r_sas: shape (n_states,) -> r(s,a,s')
        '''
        max_Q_next_turn = np.max(self.Q_sa, axis=1) 
        estimated_reward = r_sas + self.gamma * max_Q_next_turn # reward gained + Q(s',a') for optimal next action a' 
        self.Q_sa[s, a] = np.sum(p_sas * estimated_reward, axis=0)
        #we print the error in Q_value_iteration
        return 
    
    
def Q_value_iteration(env: StochasticWindyGridworld, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    env._construct_model()

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    states = range(env.n_states)
    actions = range(env.n_actions)
    
    i = 0
    max_error = 1+threshold
    while max_error > threshold:
        Q_old = QIagent.Q_sa.copy() # store old Q_sa to compute error
        i += 1
        for s in states:
            for a in actions:
                p_sas, r_sas = env.model(s,a)
                QIagent.update(s=s, a=a, p_sas=p_sas, r_sas=r_sas)
                # Plot current Q-value estimates & print max error
                max_error = np.max(np.abs(QIagent.Q_sa - Q_old))
        del Q_old
        print(f"Q-value iteration, iteration {i}, max error {max_error}")
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.1)
 
    return QIagent


def experiment():
    gamma = 1
    threshold = 1e-3
    env = StochasticWindyGridworld(initialize_model=True, reward_per_step=-1, wind_blows_proportion=0.9)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    total_reward = 0
    timesteps = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)

        total_reward += r
        timesteps += 1

        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.3)
        s = s_next

    mean_reward_per_timestep = total_reward/timesteps    
    print(f"Mean reward per timestep under optimal policy: {mean_reward_per_timestep}")
    

if __name__ == '__main__':
    experiment()

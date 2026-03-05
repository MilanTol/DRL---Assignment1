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

import matplotlib.pyplot as plt

class QLearningAgent(BaseAgent):
    def update(self, s, a, r, s_next, done):
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (target - self.Q_sa[s,a])
        return 
    
def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    s = env.reset()
    t=0
    while t < n_timesteps:
        t += 1 
        a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp) #selection action based on some randomness policy
        s_next, r, done = env.step(a) # perform action in environment to observe next state, gained reward, and termination condition
        agent.update(s, a, r, s_next, done) #update Q, based on observed reward
        if done: #if the run is done, start over.
            s = env.reset()
        else:
            s = s_next

        if t % eval_interval == 0:
            eval_timesteps.append(t)
            eval = agent.evaluate(eval_env)
            eval_returns.append(eval) #use evaluation environment to not affect the testing environment!

            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.0001) # Plot the Q-value estimates during Q-learning execution   

    return np.array(eval_returns), np.array(eval_timesteps)   

def q_learning_experiment(
        policy = 'egreedy', epsilon=None, temp=None, gamma = 1, 
        learning_rate=0.1,eval_interval = 1000, n_timesteps = 50001,
        plot = False
        ):
    
    n_timesteps = n_timesteps
    eval_interval= eval_interval
    gamma = gamma
    learning_rate = learning_rate

    # Exploration
    policy = policy # 'egreedy' or 'softmax' 
    epsilon = epsilon
    temp = temp
    
    # Plotting parameters
    plot = plot
    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    return eval_returns, eval_timesteps



if __name__ == '__main__':

    repetitions = 20

    for epsilon in [0.03, 0.1, 0.3]:
        eval_returns_list = []
        for i in range(repetitions):
            eval_returns, eval_timesteps = q_learning_experiment(policy='egreedy', epsilon=epsilon, n_timesteps=50001, eval_interval=1000)
            eval_returns_list.append(eval_returns)

        mean_returns = np.mean(eval_returns_list, axis=0)
        std_returns = np.std(eval_returns_list, axis=0)
        stderr = std_returns / np.sqrt(repetitions)
        ci95 = 1.96 * stderr

        plt.plot(eval_timesteps, np.mean(eval_returns_list, axis=0), label='egreedy:' + r'$\epsilon=$' + f'{epsilon}')
        plt.fill_between(
            eval_timesteps[:len(mean_returns)],
            mean_returns - ci95,
            mean_returns + ci95,
            alpha=0.25
        )

    plt.legend()
    plt.savefig("/home/milan/Desktop/DRL/Assignment1/plots/Q_learning_egreedy.pdf")
    plt.close()

    for temp in [0.01, 0.1, 1]:

        eval_returns_list = []
        for i in range(repetitions):
            eval_returns, eval_timesteps = q_learning_experiment(policy='softmax', temp=temp, n_timesteps=50001, eval_interval=1000)
            eval_returns_list.append(eval_returns)

        mean_returns = np.mean(eval_returns_list, axis=0)
        std_returns = np.std(eval_returns_list, axis=0)
        stderr = std_returns / np.sqrt(repetitions)
        ci95 = 1.96 * stderr

        plt.plot(eval_timesteps, mean_returns, label='softmax:' + r'$T=$' + f'{temp}')
        plt.fill_between(
            eval_timesteps[:len(mean_returns)],
            mean_returns - ci95,
            mean_returns + ci95,
            alpha=0.25
        )

    plt.legend()
    plt.savefig("/home/milan/Desktop/DRL/Assignment1/plots/Q_learning_softmax.pdf")

    plt.xlabel('episode')
    plt.close()

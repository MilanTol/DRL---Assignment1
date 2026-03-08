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

    def __init__(self, n_states: int, n_actions: int, learning_rate, gamma):
        super().__init__(n_states, n_actions, learning_rate, gamma)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        '''
        Select an action using the specified policy.

        s (int): state index
        policy (str): 'greedy', 'egreedy', or 'softmax'
        epsilon (float): exploration parameter for epsilon-greedy
        temp (float): temperature parameter for softmax
        '''

        if policy == 'greedy':
            a = np.argmax(self.Q_sa[s])

        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() < epsilon:
                a = np.random.randint(self.n_actions)
            else:
                a = np.argmax(self.Q_sa[s])

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            q_values = self.Q_sa[s]
            q_shifted = q_values - np.max(q_values)  # numerical stability
            probs = np.exp(q_shifted / temp)
            probs /= np.sum(probs)
            a = np.random.choice(self.n_actions, p=probs)

        else:
            raise ValueError("Unknown policy")

        return a

    def update(self, states, actions, rewards, done, n):
        '''
        states is a list of states observed in the episode, of length T_ep + 1
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final state is terminal
        '''
        T = len(actions)

        for t in range(T):
            end = min(t + n, T)

            G = 0.0
            for i in range(t, end):
                G += (self.gamma ** (i - t)) * rewards[i]

            if t + n < T or (t + n == T and not done):
                G += (self.gamma ** n) * np.max(self.Q_sa[states[t + n]])

            s_t = states[t]
            a_t = actions[t]
            self.Q_sa[s_t, a_t] += self.learning_rate * (G - self.Q_sa[s_t, a_t])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None,
             plot=True, n=5, eval_interval=500):
    '''
    Runs a single repetition of an n-step Q-learning agent.
    Returns:
        eval_returns: array of evaluation returns
        eval_timesteps: array of timesteps at which evaluation happened
    '''

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = []
    eval_returns = []

    timestep = 0

    while timestep < n_timesteps:
        s = env.reset()

        states = [s]
        actions = []
        rewards = []

        done = False
        t = 0

        while not done and t < max_episode_length and timestep < n_timesteps:
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)

            states.append(s_next)
            actions.append(a)
            rewards.append(r)

            s = s_next
            t += 1
            timestep += 1

            if timestep % eval_interval == 0:
                mean_return = pi.evaluate(
                    eval_env,
                    n_eval_episodes=30,
                    max_episode_length=max_episode_length
                )
                eval_timesteps.append(timestep)
                eval_returns.append(mean_return)

        pi.update(states, actions, rewards, done, n)

        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    plot = True

    eval_returns, eval_timesteps = n_step_Q(
        n_timesteps=n_timesteps,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        gamma=gamma,
        policy=policy,
        epsilon=epsilon,
        temp=temp,
        plot=plot,
        n=n
    )

    print("Evaluation timesteps:", eval_timesteps)
    print("Evaluation returns:", eval_returns)


if __name__ == '__main__':
    test()
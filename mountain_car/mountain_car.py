import os
from ddpg import DDPG
import gym
import numpy as np


class MountainCar():

    def __init__(self):
        self.agent = DDPG(gym.make('MountainCarContinuous-v0'))

    def run_epoch(self, training=True):
        state = self.agent.reset()

        total_reward = 0
        steps = 0
        done = False
        while not done:
            steps += 1
            noisy_action, pure_action = self.agent.act(state)

            # Use action with noise if training
            action = noisy_action if training else pure_action

            next_state, reward, done, _ = self.agent.task.step(action)
            if training:
                self.agent.step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward, steps

    def run_model(self, max_epochs=100, plot_Q=False):
        """
        Train the learner with max epochs.

        Args:
            max_epochs (int): Maximum number of training episodes
            plot_Q (bool): If true, plot Q values heatmaps
        """
        for epoch in range(1, max_epochs+1):
            train_reward, train_steps = self.run_epoch()
            test_reward, test_steps = self.run_epoch(training=False)

            print('Epoch:{:3}\n'
                  'Train: reward:{:6.1f}\tsteps:{:6.0f}\n'
                  'Test:  reward:{:6.1f}\tsteps:{:6.0f}'
                  .format(epoch, train_reward, train_steps, test_reward,
                          test_steps))


if __name__ == '__main__':
    learner = MountainCar()
    learner.run_model(max_epochs=10)

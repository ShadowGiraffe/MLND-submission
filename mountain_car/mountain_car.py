import os
from ddpg import DDPG
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MountainCar():
    #TODO(peizhao): refactor plot_Q().
    #TODO(peizhao): develop new methods that help debuging.

    def __init__(self):
        self.agent = DDPG(gym.make('MountainCarContinuous-v0'))

    def plot_Q(self):
        """Plots 4 heatmaps related to Q values.

        1. Q-max: Max of Q at each state.
        2. Q-std: Std of Q at each state.
        3. Q action: Action at each state that produces max Q value.
        4. Policy: Action for each state
        """
        state_step = 0.1
        action_step = 0.1
        plot_range = np.arange(-1, 1 + state_step, state_step)
        action_range = np.arange(-1, 1 + action_step, action_step)
        shape = plot_range.shape[0]
        matrix_Q = np.zeros((shape, shape))
        matrix_mQ = np.zeros((shape, shape))
        matrix_sQ = np.zeros((shape, shape))
        matrix_A = np.zeros((shape, shape))
        for i in range(shape):
            for j in range(shape):
                pos = plot_range[j]
                vel = plot_range[i]
                state = np.array([pos, vel]).reshape(-1, 2)
                Q_list = []
                for a in action_range:
                    action = np.array(a).reshape(-1, 1)
                    Q_list.append(
                        self.agent.critic_local.model.predict([state, action]))
                matrix_Q[i][j] = np.max(Q_list)
                matrix_sQ[i][j] = np.std(Q_list)
                matrix_mQ[i][j] = action_range[np.argmax(Q_list)]
                matrix_A[i][j] = self.agent.actor_local.model.predict(state)
        extent = [plot_range[0], plot_range[-1], plot_range[0], plot_range[-1]]

        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].set_title('Q value max')
        ax[0, 0].set_ylabel('Velocity')
        ax[0, 0].set_xlabel('Position')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 0].imshow(matrix_Q, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[0, 1].set_title('Q value std')
        ax[0, 1].set_ylabel('Velocity')
        ax[0, 1].set_xlabel('Position')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 1].imshow(matrix_sQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[1, 0].set_title('Action with Q max')
        ax[1, 0].set_ylabel('Velocity')
        ax[1, 0].set_xlabel('Position')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 0].imshow(matrix_mQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[1, 1].set_title('Predicted Action')
        ax[1, 1].set_ylabel('Velocity')
        ax[1, 1].set_xlabel('Position')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 1].imshow(matrix_A, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        plt.subplots_adjust(top=0.92, right=0.95, hspace=0.25, wspace=0.4)

        plt.show()

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

        if plot_Q:
            self.plot_Q()


if __name__ == '__main__':
    learner = MountainCar()
    learner.run_model(max_epochs=10)

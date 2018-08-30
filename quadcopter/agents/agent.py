import copy
from collections import namedtuple, deque
from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import random


class DDPG():
    """Reinforcement Learning agent implemented with DDPG algorithm."""

    def __init__(self, task):
        # Task/Env
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(
            self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(
            self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu,
                             self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.999  # discount factor
        self.tau_actor = 0.01  # for soft update of target parameters
        self.tau_critic = 0.01

        # Scores
        self.score = 0

    def reset(self):
        self.score = 0
        self.noise.reset()
        return self.task.reset()

    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.score += reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        """Return actions for given state(s) as per current policy.

        Args:
            state (array): A state array.

        Returns:
            A tuple of (noisy_action, pure_action).
        """
        state = np.reshape(state, [-1, self.state_size])
        pure_normalized_action = self.actor_local.model.predict(state)[0]
        noisy_normalized_action = np.clip(
            pure_normalized_action + self.noise.sample(), -1, 1)
        # Scale [-1, 1] output for each action dimension to proper range
        return tuple(
            (x + 1) * self.action_range / 2 + self.action_low
            for x in (noisy_normalized_action, pure_normalized_action))

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experiences.
        """
        # Convert experience tuples to separate arrays for each element
        # (states, actions, rewards, etc.)
        experiences = [e for e in experiences if e is not None]
        states = np.vstack([e.state for e in experiences])
        actions = np.array(
            [e.action for e in experiences]
        ).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([
            e.reward for e in experiences]).astype(np.float32).reshape(-1, 1)
        dones = np.array(
            [e.done for e in experiences]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch(
            [next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(
            [states, actions, 0]), (-1, self.action_size))
        # custom training function
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model,
                         self.tau_critic)
        self.soft_update(self.actor_local.model, self.actor_target.model,
                         self.tau_actor)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), (
            'Local and target model parameters must have the same size')

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Add final output layer
        actions = layers.Dense(
            units=self.action_size,
            activation='tanh',
            kernel_initializer=initializers.RandomUniform(
                minval=-3e-3, maxval=3e-3),
            name='actions')(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.001)
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """
        Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic network that maps (state, action) pairs -> Q-values.
        """
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(
            units=1,
            kernel_initializer=initializers.RandomUniform(
                minval=-3e-4, maxval=3e-4),
            name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with
        # built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be
        # used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.done = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses vertical velocity to return reward."""
        reward = np.clip(self.sim.v[2]*0.2, -1, 1)
        if (np.sum(np.exp2(self.sim.pose[:2] - self.target_pos[:2])) > 25 or
            self.sim.pose[2] < 1):
            self.done = True
            reward = -100
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_list = []
        for _ in range(self.action_repeat):
            self.done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            state_list.append(np.concatenate((self.sim.pose, self.sim.v)))
        next_state = np.concatenate(state_list)
        return next_state, reward, self.done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        return np.tile(np.concatenate((self.sim.pose, self.sim.v)),
                       self.action_repeat)

import tella
import torch
import numpy as np
from repr import RePR


class RePRAgent(tella.ContinualRLAgent):
    def __init__(
        self, rng_seed, observation_space, action_space, num_envs, config_file
    ):
        super(RePRAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        self.repr_model = RePR()
        self.frames_per_update = 4
        self.env_steps = 0

    def block_start(self, is_learning_allowed):
        pass

    def task_start(self, task_name):
        pass

    def task_variant_start(self, task_name, variant_name):
        pass

    def choose_actions(self, observations):
        if self.env_steps < self.frames_per_update:
            self.prev_action = 0
            return 0

        if self.env_steps % self.frames_per_update == 0:
            # Sample new Action
            self.curr_action = self.repr_model.sample_action(self.buffer_obs)

        # Keep sending current action
        return self.curr_action

    def receive_transitions(self, transitions):
        if self.env_steps % self.frames_per_update == 0:
            total_r = sum([r for _, r, _, _ in self.buffer_observations])
            one_last_frame = self.buffer_observations[-2][0]
            last_frame, _, done, info = self.buffer_observations[-1]
            self.repr_model.add_transition(
                (np.maximum(one_last_frame, last_frame), total_r, done, info)
            )
            self.buffer_observations = []
            if done:
                self.env_steps = -1

        self.buffer_observations.append(transitions)
        self.env_steps += 1

    def task_variant_end(self, task_name, variant_name):
        pass

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass

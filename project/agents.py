import collections
import logging
import tella
import torch
import cv2
import numpy as np
from repr import RePR
from curriculums.atari import SimpleAtariSequenceCurriculum


def preprocess(one_last_frame, last_frame):
    x = np.maximum(one_last_frame, last_frame)
    x = x.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (82, 82), interpolation=cv2.INTER_LINEAR)
    return x


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
        self.buffer_observations = []
        self.buffer_sample_action = collections.deque(maxlen=4)
        self.trainning = False

        self.logger = logging.getLogger("RePR Agent")

    def block_start(self, is_learning_allowed):
        self.env_steps = 0
        self.buffer_observations = []
        self.buffer_sample_action = collections.deque(maxlen=4)

        self.trainning = is_learning_allowed
        self.repr_model.learning(is_learning_allowed)
        self.logger.info(f"Block with learning: {self.trainning}")

    def task_start(self, task_name):
        self.logger.info(f"Start task {task_name}")

    def task_variant_start(self, task_name, variant_name):
        if self.trainning:
            if "0" in variant_name:
                self.repr_model.set_mode("stm")
            else:
                self.repr_model.set_mode("ltm")
        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        if self.env_steps < self.frames_per_update:
            self.curr_action = 0
        elif self.env_steps % self.frames_per_update == 0:
            # Sample new Action
            x = list(self.buffer_sample_action)
            x = np.array([x[0]] * (4-len(x)) + x)
            x = torch.from_numpy(x).float().unsqueeze(0)
            self.curr_action = self.repr_model.sample_action(x)

        self.env_steps += 1
        # print(f"Log| Selected action: {self.curr_action}")
        # Keep sending current action
        return [self.curr_action]

    def receive_transitions(self, transitions):
        self.buffer_observations.append(transitions[0])

        if self.env_steps % self.frames_per_update == 0:
            one_last_frame = self.buffer_observations[-2][-1]
            _, action, _, done, last_frame = self.buffer_observations[-1]
            observation = preprocess(one_last_frame, last_frame)
            self.buffer_sample_action.append(observation)

            if self.trainning:
                total_r = sum([r for _, _, r, _, _ in self.buffer_observations])
                self.repr_model.add_transition(
                    (observation, action, total_r, done))

            self.buffer_observations = []
            if done:
                self.env_steps = 0

    def task_variant_end(self, task_name, variant_name):
        pass

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.curriculum.curriculum_registry["SimpleAtari"] = SimpleAtariSequenceCurriculum
    tella.rl_cli(RePRAgent)

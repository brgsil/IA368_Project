import collections
import os
import logging
import tella
import torch
import cv2
import numpy as np

from repr import RePR
from curriculums.atari import SimpleAtariSequenceCurriculum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"##### DEVICE: {device}")


def preprocess(one_last_frame, last_frame):
    x = np.maximum(one_last_frame, last_frame)
    x = x.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_LINEAR)
    # return (x / 255.) * 2 - 1
    return x


def preprocess2(last_frame):
    x = last_frame.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_LINEAR)
    # return (x / 255.) * 2 - 1
    return x


class RePRAgent(tella.ContinualRLAgent):
    def __init__(
        self, rng_seed, observation_space, action_space, num_envs, config_file
    ):
        super(RePRAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        self.repr_model = RePR()
        self.frames_per_update = 1
        self.env_steps = 0
        self.total_steps = 0
        self.buffer_observations = collections.deque(maxlen=4)
        self.buffer_sample_action = collections.deque(maxlen=4)
        self.prev_observation = np.zeros((4, 84, 84))
        self.trainning = False
        self.first_ltm_train = True

        self.checkpoint_count = 0
        self.logger = logging.getLogger("RePR Agent")
        self.test_video = {}
        self.curr_task = ""
        self.prev_obs_is_done = False

    def block_start(self, is_learning_allowed):
        self.trainning = is_learning_allowed
        self.repr_model.learning(is_learning_allowed)
        # self.logger.info(f"Block with learning: {self.trainning}")

    def task_start(self, task_name):
        # self.logger.info(f"Start task {task_name}")
        pass

    def task_variant_start(self, task_name, variant_name):
        self.env_steps = 0
        self.total_steps = 0
        self.repr_model.task = task_name
        self.buffer_observations = collections.deque(maxlen=4)
        self.buffer_sample_action = collections.deque(maxlen=4)
        self.curr_task = task_name

        if "Checkpoint" in variant_name:
            if not self.trainning:
                checkpoint_path = (
                    f"./logs/repr_dqn/latest/checkpoint_{self.checkpoint_count}/"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                self.repr_model.save_checkpoint(dir=checkpoint_path)
                self.checkpoint_count += 1

        if "STM" in variant_name:
            self.repr_model.set_mode("stm")
        elif "LTM" in variant_name:
            if self.trainning:
                self.repr_model.first_ltm_train = self.first_ltm_train
                self.repr_model.tasks_seen += 1
                if self.first_ltm_train:
                    self.first_ltm_train = False
            self.repr_model.set_mode("ltm")

        if not self.trainning:
            self.test_video[self.curr_task] = []

        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        if self.env_steps < 4:
            self.curr_action = 0
        elif self.env_steps % self.frames_per_update == 0:
            # Sample new Action
            x = list(self.buffer_sample_action)
            x = np.array([x[0]] * (4 - len(x)) + x)
            x = 2 * x / 255.0 - 1
            x = torch.from_numpy(x).float().unsqueeze(0)
            with torch.no_grad():
                self.curr_action = self.repr_model.sample_action(x)

        self.env_steps += 1
        self.total_steps += 1
        # print(f"Log| Selected action: {self.curr_action}")
        # Keep sending current action
        return [self.curr_action]

    def receive_transitions(self, transitions):
        # self.logger.info(f"Receiving transition - Step {self.env_steps}")
        if transitions[0] is not None:
            s, a, r, done, s_ = transitions[0]
            self.buffer_observations.append((s, a, r, done, s_))

            if not self.trainning:
                if self.prev_obs_is_done:
                    self.test_video[self.curr_task] = []

                self.test_video[self.curr_task].append(s)
                self.prev_obs_is_done = done

            if (done or self.env_steps % self.frames_per_update == 0) and len(
                self.buffer_observations
            ) >= 1:

                # one_last_frame = self.buffer_observations[-2][-2]
                _, action, _, done, last_frame = self.buffer_observations[-1]
                # observation = preprocess(one_last_frame, last_frame)
                observation = preprocess2(last_frame)
                self.prev_observation = np.array(self.buffer_sample_action)
                self.buffer_sample_action.append(observation)
                curr_observation = np.array(self.buffer_sample_action)

                if self.trainning and self.prev_observation.shape[0] == 4:
                    total_r = sum(
                        [r for _, _, r, _, _ in self.buffer_observations])
                    self.repr_model.add_transition(
                        (
                            self.prev_observation,
                            action,
                            total_r,
                            done,
                            curr_observation,
                        )
                    )

                if done:
                    self.env_steps = 0

    def task_variant_end(self, task_name, variant_name):
        if not self.trainning:
            frames = self.test_video[task_name]
            out = cv2.VideoWriter(
                f"output_{task_name}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (160, 250),
            )
            for frame in frames:
                out.write(frame)
            out.release()

        if self.trainning:
            if "Last" in variant_name:
                self.repr_model.set_mode("gan")
                self.repr_model.train_step()

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_experiment(
        RePRAgent,
        SimpleAtariSequenceCurriculum,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="./logs",
    )
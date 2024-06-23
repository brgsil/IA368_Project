import collections
import os
import logging
import tella
import cv2
import torch
import numpy as np

from repr import RePR
from curriculums.lunar import LunarCurriculum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"##### DEVICE: {device}")


class RePRAgent(tella.ContinualRLAgent):
    def __init__(
        self, rng_seed, observation_space, action_space, num_envs, config_file
    ):
        super(RePRAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        self.repr_model = RePR()
        self.trainning = False
        self.first_ltm_train = True

        self.checkpoint_count = 0
        self.logger = logging.getLogger("RePR Agent")
        self.test_video = []
        self.prev_obs_is_done = False
        self.train_task = "Init Model"
        self.test_r = 0
        self.test_ep_r = []

    def block_start(self, is_learning_allowed):
        self.trainning = is_learning_allowed
        self.repr_model.learning(is_learning_allowed)
        # self.logger.info(f"Block with learning: {self.trainning}")

    def task_start(self, task_name):
        pass

    def task_variant_start(self, task_name, variant_name):
        if self.trainning:
            self.train_task = task_name
        self.repr_model.task = task_name

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

        del self.test_video
        self.test_video = []

        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        # Sample new Action
        if isinstance(observations[0]['state'], np.ndarray):
            x = observations[0]['state'].squeeze()
            if (x.shape[0] == 8):
                x = torch.from_numpy(x).float().unsqueeze(0)
                with torch.no_grad():
                    self.curr_action = self.repr_model.sample_action(x)

        # Keep sending current action
        return [self.curr_action]

    def receive_transitions(self, transitions):
        if transitions[0] is not None:
            s, a, r, done, s_ = transitions[0]

            if not self.trainning:
                self.test_r += r
                if done:
                    self.test_ep_r.append(self.test_r)
                    self.test_r = 0

                if self.prev_obs_is_done:
                    self.test_video = []

                self.test_video.append(s['pixels'][0])
                self.prev_obs_is_done = done

            if self.trainning:
                assert s['state'].shape == (8,)
                assert s_['state'].shape == (8,)
                self.repr_model.add_transition(
                    (
                        s['state'],
                        a,
                        r/100.,
                        done,
                        s_['state'],
                    )
                )

    def task_variant_end(self, task_name, variant_name):
        if not self.trainning:
            with open("eval.txt", "a") as f:
                f.write(f"{self.repr_model.mode} | {self.train_task} - {self.repr_model.task} | {sum(self.test_ep_r)/len(self.test_ep_r):.2f}\n")
            self.test_ep_r = []

            frames = self.test_video
            out = cv2.VideoWriter(
                f"output_dqn_{task_name}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (600, 400),
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
    #logging.basicConfig(level=logging.INFO)
    tella.rl_experiment(
        RePRAgent,
        LunarCurriculum,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="./logs",
    )

import collections
import os
import logging
import tella
import torch
import cv2
import numpy as np
from ppo import PPO
from curriculums.lunar import LunarCurriculumPPO


def preprocess(one_last_frame, last_frame):
    x = np.maximum(one_last_frame, last_frame)
    x = x.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_LINEAR)
    # return (x / 255.) * 2 - 1
    return x


class PPOAgent(tella.ContinualRLAgent):
    def __init__(
        self, rng_seed, observation_space, action_space, num_envs, config_file
    ):
        super(PPOAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        self.action_space = 4
        self.model = PPO(action_space=self.action_space)
        self.trainning = False

        self.env_steps = 0
        self.total_steps = 0
        self.buffer_observations = collections.deque(maxlen=4)
        self.buffer_sample_action = collections.deque(maxlen=4)
        self.prev_observation = np.zeros((4, 84, 84))
        self.action_probs = 1. / self.action_space
        self.checkpoint_count = 0
        self.losses = []
        self.train_r = []
        self.train_ep_r = []
        self.entropy = []
        self.logger = logging.getLogger("PPO Agent")
        self.ppo_horizon = 1000
        self.task = ""
        self.curr_task = ""
        self.prev_obs_is_done = False

    def block_start(self, is_learning_allowed):
        self.trainning = is_learning_allowed

    def task_start(self, task_name):
        # self.logger.info(f"Start task {task_name}")
        pass

    def task_variant_start(self, task_name, variant_name):
        self.env_steps = 0
        if not self.task == task_name and 'Train' in variant_name:
            self.total_steps = 0
            self.task = task_name
            self.loss = []
        self.curr_task = task_name
        self.buffer_observations = collections.deque(maxlen=4)
        self.buffer_sample_action = collections.deque(maxlen=4)
        self.action_probs = 1. / self.action_space

        if "Checkpoint" in variant_name:
            if not self.trainning:
                checkpoint_path = (
                    f"./logs/ppo/latest/checkpoint_{self.checkpoint_count}/"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                self.model.save_checkpoint(dir=checkpoint_path)
                self.checkpoint_count += 1

        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        if isinstance(observations[0], np.ndarray):
            # Sample new Action
            x = observations[0].squeeze()
            if (x.shape[0] == 8):
                x = torch.from_numpy(x).float().unsqueeze(0)
                assert x.shape == (1, 8), f"Actual: {x.shape}"
                with torch.no_grad():
                    self.curr_action, self.action_probs = self.model.sample_action(
                        x)

        self.env_steps += 1
        self.total_steps += 1
        # print(f"Log| Selected action: {self.curr_action}")
        # Keep sending current action
        return [self.curr_action]

    def receive_transitions(self, transitions):
        # self.logger.info(f"Receiving transition - Step {self.env_steps}")
        if transitions[0] is not None:
            s, a, r, done, s_ = transitions[0]
            self.buffer_observations.append(
                (s, a, r, done, s_, self.action_probs))

            if isinstance(s, np.ndarray):
                s = s[:].squeeze()
                s_ = s_[:].squeeze()

            if self.trainning:
                if s.shape == (8,):
                    self.model.put_data(
                        (
                            s,
                            a,
                            r/100.,
                            s_,
                            self.action_probs,
                            done,
                        )
                    )
                self.train_r.append(r)
                if done:
                    self.train_ep_r.append(sum(self.train_r))
                    self.train_r = []
                if done or (self.env_steps % self.ppo_horizon == 0):
                    l, e = self.model.train_net()
                    self.losses.append(l)
                    self.entropy.append(e)

            # self.prev_observation = observation

            if done:
                self.env_steps = 0

        if self.trainning and self.total_steps % 20_000 == 0:
            log = f"{self.task} Train [{self.total_steps/self.frames_per_update/1_000_000.:.2f}M] |"+\
                f"Loss {sum(self.losses)/len(self.losses):.4f}"+\
                f" | Entropy: {sum(self.entropy)/len(self.entropy):.4f}" +\
                f" | Reward {sum(self.train_ep_r)/len(self.train_ep_r):.1f}"
            print(log)
            with open("terminal_ppo.txt", "a") as f:
                f.write(log + "\n")
            self.train_ep_r = []
            self.losses = []
            self.entropy = []

    def task_variant_end(self, task_name, variant_name):
        pass

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_experiment(
        PPOAgent,
        LunarCurriculumPPO,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="./logs/ppo",
    )

import collections
import os
import logging
import tella
import torch
import numpy as np
from ppo import PPO
from curriculums.lunar import LunarCurriculumPPO


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

        self.action_probs = 1.0 / self.action_space
        self.checkpoint_count = 0
        self.losses = []
        self.train_r = []
        self.train_ep_r = []
        self.entropy = []
        self.test_r = 0
        self.test_ep_r = []
        self.logger = logging.getLogger("PPO Agent")
        self.ppo_horizon = 1000
        self.task = "Init Model"
        self.curr_task = ""
        self.prev_obs_is_done = False
        self.total_steps = 0

    def block_start(self, is_learning_allowed):
        self.trainning = is_learning_allowed

    def task_start(self, task_name):
        pass

    def task_variant_start(self, task_name, variant_name):
        self.env_steps = 0
        if not self.task == task_name and "Train" in variant_name:
            self.total_steps = 0
            self.task = task_name
            self.loss = []
            self.model.reset_lr()
        self.curr_task = task_name
        self.action_probs = 1.0 / self.action_space

        if "Checkpoint" in variant_name:
            if not self.trainning:
                checkpoint_path = (
                    f"./logs/ppo_agent/latest/checkpoint_{self.checkpoint_count}/"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                self.model.save_checkpoint(dir=checkpoint_path)
                self.checkpoint_count += 1

        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        x = observations[0]
        if isinstance(x, np.ndarray):
            # Sample new Action
            x = x.squeeze()
            if x.shape[0] == 8:
                x = torch.from_numpy(x).float().unsqueeze(0)
                assert x.shape == (1, 8), f"Actual: {x.shape}"
                with torch.no_grad():
                    self.curr_action, self.action_probs = self.model.sample_action(
                        x)

        self.env_steps += 1
        if self.trainning:
            self.total_steps += 1

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

            if self.trainning:
                if s.shape == (8,):
                    self.model.put_data(
                        (
                            s,
                            a,
                            r / 100.0,
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
                    self.entropy.extend(e)

            if done:
                self.env_steps = 0

        if self.trainning and self.total_steps % 2_000 == 0:
            log = (
                f"PPO | {self.task} Train [{self.total_steps/1_000_000.:.3f}M] |"
                + f"Loss: {sum(self.losses)/len(self.losses):.4f}"
                + f" | Entropy: {sum(self.entropy)/len(self.entropy):.4f}"
                + f" | Reward: {sum(self.train_ep_r)/len(self.train_ep_r):.1f}"
            )
            print(log)
            with open("train_ppo.txt", "a") as f:
                f.write(log + "\n")
            self.train_ep_r = []
            self.losses = []
            self.entropy = []

    def task_variant_end(self, task_name, variant_name):
        if not self.trainning:
            with open("eval_ppo.txt", "a") as f:
                f.write(
                    f"PPO | {self.task} - {self.curr_task} | {sum(self.test_ep_r)/len(self.test_ep_r):.2f}\n"
                )
            self.test_ep_r = []

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass


if __name__ == "__main__":
    tella.rl_experiment(
        PPOAgent,
        LunarCurriculumPPO,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="./logs/ppo",
    )

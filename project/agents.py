import collections
import os
import logging
import tella
import torch
import numpy as np

# from repr import RePR
from repr_ppo import RePR

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

        self.action_space = 4
        self.repr_model = RePR(action_space=self.action_space)
        self.env_steps = 0
        self.total_steps = 0
        self.action_probs = 1. / self.action_space
        self.trainning = False
        self.first_ltm_train = True

        self.checkpoint_count = 0
        self.logger = logging.getLogger("RePR Agent")
        self.test_video = []
        self.curr_task = ""
        self.prev_obs_is_done = False
        self.curr_action = 0

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
        self.curr_action = 0
        self.action_probs = 1 / self.action_space
        if task_name != self.curr_task:
            self.repr_model.tasks_seen += 1
        self.curr_task = task_name

        if "Checkpoint" in variant_name:
            if not self.trainning:
                checkpoint_path = (
                    f"./logs/repr_ppo/latest/checkpoint_{self.checkpoint_count}/"
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

        self.test_video = []

        self.logger.info(f"Start variant {variant_name}")

    def choose_actions(self, observations):
        # if self.env_steps < 4:
        #    self.curr_action = 0
        #    self.action_probs = 1 / 18.0
        # elif self.env_steps % self.frames_per_update == 0:
        # Sample new Action
        # x = list(self.buffer_sample_action)
        # x = np.array([x[0]] * (4 - len(x)) + x)
        # x = 2 * x / 255.0 - 1
        if isinstance(observations[0]['state'], np.ndarray):
            x = observations[0]['state'].squeeze()
            if (x.shape[0] == 8):
                x = torch.from_numpy(x).float().unsqueeze(0)
                #x = 2 * x / 255.0 - 1
                assert x.shape == (1, 8), f"Actual: {x.shape}"
                with torch.no_grad():
                    if self.repr_model.mode == "stm":
                        self.curr_action, self.action_probs = self.repr_model.sample_action(
                            x
                        )
                    else:
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
            if isinstance(s['state'], np.ndarray):

                if not self.trainning:
                    if self.prev_obs_is_done:
                        self.test_video = []

                    self.test_video.append(s['pixels'][0])
                    self.prev_obs_is_done = done

                s = s['state'][:].squeeze()
                s_ = s_['state'][:].squeeze()
                # self.buffer_observations.append((s, a, r, done, s_, self.action_probs))
                # if (done or self.env_steps % self.frames_per_update == 0) \
                #        and len(self.buffer_observations) >= 1:

                # one_last_frame = self.buffer_observations[-2][-2]
                # _, action, _, done, last_frame, prob_a = self.buffer_observations[-1]
                # observation = preprocess(one_last_frame, last_frame)
                # observation = preprocess2(last_frame)
                # self.prev_observation = np.array(self.buffer_sample_action)
                # self.buffer_sample_action.append(observation)
                # curr_observation = np.array(self.buffer_sample_action)

                # if self.trainning and self.prev_observation.shape[0] == 4:
                if self.trainning:
                    if s.shape == (8,):
                        if self.repr_model.mode == "stm":
                            self.repr_model.add_transition(
                                (
                                    s,
                                    a,
                                    r/100.,
                                    s_,
                                    self.action_probs,
                                    done,
                                )
                            )
                        else:
                            self.repr_model.add_transition(
                                (
                                    s,
                                    a,
                                    r/100.,
                                    done,
                                    s_,
                                )
                            )

                if done:
                    self.env_steps = 0

    def task_variant_end(self, task_name, variant_name):
        #if not self.trainning:
        #    frames = self.test_video
        #    out = cv2.VideoWriter(
        #        f"output_{task_name}.mp4",
        #        cv2.VideoWriter_fourcc(*"mp4v"),
        #        10,
        #        (84, 84),
        #        False,
        #    )
        #    for frame in frames:
        #        out.write(frame)
        #    out.release()

        if self.trainning:
            if "Last" in variant_name:
                self.repr_model.set_mode("gan")
                self.repr_model.train_step()
        pass

    def task_end(self, task_name):
        pass

    def block_end(self, is_learning_allowed):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_experiment(
        RePRAgent,
        LunarCurriculum,
        num_lifetimes=1,
        num_parallel_envs=1,
        log_dir="./logs",
    )

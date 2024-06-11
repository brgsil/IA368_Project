import random
import torch
from dqn import DQN, Qnet, ReplayBuffer
from gan import GAN


class RePR:
    def __init__(self, mode="stm", batch_size=32, alpha=0.5):
        self.stm_dqn = DQN()
        self.ltm_net = Qnet()
        self.ltm_replay = ReplayBuffer(size=200_000)
        self.gan = GAN()
        self.new_gan = GAN()

        assert mode in ["stm", "ltm", "gan"]
        self.mode = mode

        self.batch_size = batch_size
        self.first_ltm_train = True
        self.alpha = alpha

        self.ltm_optimizer = torch.optim.RMSprop(
            self.ltm_net.parameters(), lr=0.00025, eps=1e-6
        )
        self.trainning = False

    def learning(self, learn):
        self.trainning = learn

    def add_transition(self, obs):
        if self.mode == "stm":
            self.stm_dqn.receive_transition(obs)
        else:
            self.ltm_replay.put(obs)

    def sample_action(self, obs):
        if self.mode == "stm":
            return self.stm_dqn.sample_action(obs)
        else:
            out = self.ltm_net(obs)
            coin = random.random()
            if coin < 0.1:
                return random.randint(0, 18)
            else:
                return out.argmax().item()

    def train_stm_step(self):
        self.stm_dqn.train_step()
        pass

    def train_step(self):
        if self.mode == "stm":
            self.train_stm_step()
        elif self.mode == "ltm":
            self.train_ltm_step()
        else:
            self.train_gan()

    def set_mode(self, mode):
        if not mode == self.mode:
            if mode == "stm":
                self.stm_dqn = DQN()
            if mode == "ltm":
                self.gan = self.new_gan
                self.prev_ltm_net = Qnet()
                self.prev_ltm_net.load_state_dict(self.ltm_net.state_dict())
                self.ltm_replay = ReplayBuffer(size=200_000)
            if mode == "gan":
                self.new_gan = GAN()

    def train_ltm_step(self):
        if self.first_ltm_train:
            self.ltm_net.load_state_dict(self.stm_dqn.q_net.state_dict())
            self.first_ltm_train = False
        else:
            s, _, _, _, _ = self.ltm_replay.sample(self.batch_size)

            ltm_q_out = self.ltm_net(s)
            stm_q_out = self.stm_dqn.logits(s)

            loss_curr_task = torch.nn.functional.mse_loss(ltm_q_out, stm_q_out)

            gen_obs = self.gan.sample(self.ltm_net.batch_size)
            ltm_q_out_gen = self.ltm_net(gen_obs)
            prev_ltm_q_out_gen = self.prev_ltm_net(gen_obs)

            loss_prev_task = torch.nn.functional.mse_loss(
                ltm_q_out_gen, prev_ltm_q_out_gen
            )

            loss = self.alpha * loss_curr_task + \
                (1 - self.alpha) * loss_prev_task

            self.ltm_optimizer.zero_grad()
            loss.backward()
            self.ltm_optimizer.step()

    def train_gan(self):
        if random.random() < 1 / self.tasks_seen:
            real_samples = self.ltm_replay.sample(self.batch_size)
        else:
            real_samples = self.gan.sample(batch=self.batch_size)

        self.new_gan.train_step(real_samples)

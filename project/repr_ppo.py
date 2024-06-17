import random
import collections
from copy import deepcopy
import torch
from dqn import DQN, Qnet, ReplayBuffer
from ppo import PPO
from gan import GAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RePR:
    def __init__(self, mode="stm", batch_size=32, alpha=0.5):
        self.stm_model = PPO()
        self.ltm_net = Qnet().to(device)
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
        self.task = ""
        self.tasks_seen = 0
        self.train_loss = collections.deque(maxlen=10_000)
        self.train_r = []
        self.train_ep_r = []

        self.stm_steps = 0
        self.ltm_steps = 0
        self.env_steps = 0

    def learning(self, learn):
        # self.stm_model.train = learn
        self.trainning = learn

    def add_transition(self, obs):
        self.env_steps += 1
        if self.trainning:
            self.train_r.append(obs[2])
            if obs[5]:
                self.train_ep_r.append(sum(self.train_r))
                self.train_r = []

            if self.env_steps % 10_000 == 0:
                print(
                    f"{self.mode} - {self.task} Train [{self.env_steps/10_000.0:.2f}M steps] |"
                    + f" Loss:{sum(self.train_loss)/len(self.train_loss):.5f}"
                    + f" | Reward: {sum(self.train_ep_r)/len(self.train_ep_r):.4f}"
                )
                with open("terminal.txt", "a") as f:
                    f.write(
                        f"{self.mode} - {self.task} Train [{self.env_steps/10_000.0:.2f}M steps] |"
                        + f" Loss:{sum(self.train_loss)/len(self.train_loss):.5f}"
                        + f" | Reward: {sum(self.train_ep_r)/len(self.train_ep_r):.4f}\n"
                    )
                self.train_ep_r = []

        if self.mode == "stm":
            self.stm_model.put_data(obs)
            if self.trainning:
                if obs[5] or len(self.stm_model.data) >= 1000:
                    self.train_step()
        else:
            self.ltm_replay.put(obs)
            if self.trainning:
                self.train_step()

    def sample_action(self, obs):
        if self.mode == "stm":
            return self.stm_model.sample_action(obs)
        else:
            return self.ltm_net.sample_action(
                obs, mode="train" if self.trainning else "eval"
            )

    def set_mode(self, mode):
        if not mode == self.mode:
            if mode == "stm":
                print("CHANGE")
                self.stm_steps = 0
                self.env_steps = 0
                self.stm_model = PPO()
            if mode == "ltm":
                self.env_steps = 0
                self.ltm_steps = 0
                self.gan.copy_from(self.new_gan)
                self.prev_ltm_net = Qnet().to(device)
                self.prev_ltm_net.load_state_dict(self.ltm_net.state_dict())
                self.ltm_replay = ReplayBuffer(size=200_000)
            if mode == "gan":
                self.new_gan = GAN()
        self.mode = mode

    def train_step(self):
        # print("Training RePR")
        if self.mode == "stm":
            self.train_stm_step()
        elif self.mode == "ltm":
            self.train_ltm_step()
        else:
            self.train_gan()

    def train_stm_step(self):
        loss = self.stm_model.train_net()
        self.train_loss.append(loss)
        self.stm_steps += 1

    def train_ltm_step(self):
        self.ltm_steps += 1
        # if self.first_ltm_train:
        #    self.ltm_net.load_state_dict(self.stm_dqn.q_net.state_dict())
        #    self.ltm_replay = deepcopy(self.stm_dqn.replay)
        # else:
        if self.ltm_replay.size() > 1000:
            s, _, _, _, _ = self.ltm_replay.sample(self.batch_size)
            s = s.to(device)

            with torch.no_grad():
                stm_q_out = self.stm_model.logits(s)

            ltm_q_out = self.ltm_net(s)

            loss_curr_task = torch.nn.functional.mse_loss(ltm_q_out, stm_q_out)

            if not self.first_ltm_train:
                with torch.no_grad():
                    gen_obs = self.gan.sample(self.batch_size)
                    prev_ltm_q_out_gen = self.prev_ltm_net(gen_obs)

                ltm_q_out_gen = self.ltm_net(gen_obs)

                loss_prev_task = torch.nn.functional.mse_loss(
                    ltm_q_out_gen, prev_ltm_q_out_gen
                )
            else:
                loss_prev_task = 0

            loss = self.alpha * loss_curr_task + (1 - self.alpha) * loss_prev_task

            self.ltm_optimizer.zero_grad()
            loss.backward()
            self.ltm_optimizer.step()
            self.train_loss.append(loss.detach().item())
            print(f"LTM Train | Loss:{loss.detach().item():.8f}", end="\r")

    def train_gan(self):
        print(f"Buffer size: {self.ltm_replay.size()} - Batch: {self.batch_size}")
        avg_disc_loss = 0
        avg_gen_loss = 0
        for i in range(10_000):
            if random.random() < 1 / self.tasks_seen:
                real_samples = self.ltm_replay.sample(100)[0]
            else:
                real_samples = self.gan.sample(batch=100)

            disc_loss, gen_loss = self.new_gan.train_step(real_samples)
            avg_disc_loss += disc_loss / 20
            avg_gen_loss += gen_loss / 20
            print(
                f"GAN TRAIN [{i}/10_000] | Disc: {avg_disc_loss:.4f} - Gen: {avg_gen_loss:.4f}",
                end="\r",
            )
        print(
            f"GAN TRAIN [10_000/10_000] | Disc: {avg_disc_loss:.4f} - Gen: {avg_gen_loss:.4f}"
        )

    def save_checkpoint(self, dir="./logs/checkpoints/"):
        # Save stm
        self.stm_model.save_checkpoint(dir)
        # Save ltm
        torch.save(
            {
                "iter": self.ltm_steps,
                "replay": self.ltm_replay.buffer,
                "model_state_dict": self.ltm_net.state_dict(),
                "optimizer_state_dict": self.ltm_optimizer.state_dict(),
            },
            dir + "ltm.pt",
        )
        # Save GAN
        torch.save(
            {
                "gen_state_dict": self.gan.gen.state_dict(),
                "disc_state_dict": self.gan.disc.state_dict(),
                "optimizer_gen_state_dict": self.gan.gen_optim.state_dict(),
                "optimizer_disc_state_dict": self.gan.disc_optim.state_dict(),
            },
            dir + "gan.pt",
        )

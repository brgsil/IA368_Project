import random
import collections
import torch
from torch.distributions import Categorical
from dqn import LinearQnet, ReplayBuffer
from ppo import PPO
from gan import GAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RePR:
    def __init__(self, mode="stm", batch_size=128, alpha=0.5, action_space=18):
        self.action_space = action_space
        self.stm_model = PPO(action_space=action_space)
        self.ltm_net = LinearQnet(action_space=action_space).to(device)
        self.ltm_replay = ReplayBuffer(size=20_000)
        self.gan = GAN()
        self.new_gan = GAN()
        self.kkk = True

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
        self.train_loss = collections.deque(maxlen=20_000)
        self.train_r = []
        self.train_ep_r = []
        self.train_entropy = []

        self.stm_steps = 0
        self.ltm_steps = 0
        self.env_steps = 0

    def learning(self, learn):
        # self.stm_model.train = learn
        self.trainning = learn

    def add_transition(self, obs):
        self.env_steps += 1

        if self.mode == "stm":
            self.stm_model.put_data(obs)
            if self.trainning:
                self.train_r.append(obs[2])
                self.log(obs[5])
                if obs[5] or len(self.stm_model.data) >= 1000:
                    self.train_step()
        else:
            self.ltm_replay.put(obs)
            if self.trainning:
                self.train_r.append(obs[2])
                self.log(obs[3])
                self.train_step()

    def log(self, end_ep):
        if end_ep:
            self.train_ep_r.append(sum(self.train_r))
            self.train_r = []

        if self.env_steps % 2_000 == 0:
            entropy = ""
            if self.mode in ['stm', 'ltm']:
                entropy = f" | Entropy: {sum(self.train_entropy)/len(self.train_entropy):.5f}"
                self.train_entropy = []

            print(
                f"{self.mode} - {self.task} Train [{self.env_steps/1_000_000.0:.3f}M steps] |"
                + f" Loss:{sum(self.train_loss)/len(self.train_loss):.5f}"
                + entropy
                + f" | Reward: {sum(self.train_ep_r)/len(self.train_ep_r):.4f}"
            )
            with open("terminal.txt", "a") as f:
                f.write(
                    f"{self.mode} - {self.task} Train [{self.env_steps/1_000_000.0:.3f}M steps] |"
                    + f" Loss:{sum(self.train_loss)/len(self.train_loss):.5f}"
                    + entropy
                    + f" | Reward: {sum(self.train_ep_r)/len(self.train_ep_r):.4f}\n"
                )
            self.train_ep_r = []

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
                self.stm_model = PPO(action_space=self.action_space)
                self.ltm_replay = ReplayBuffer(size=20_000)
            if mode == "ltm":
                print("CHANGE TO LTM")
                self.train_loss.clear()
                self.env_steps = 0
                self.ltm_steps = 0
                self.gan.copy_from(self.new_gan)
                self.prev_ltm_net = LinearQnet(action_space=self.action_space).to(device)
                self.prev_ltm_net.load_state_dict(self.ltm_net.state_dict())
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
        loss, entropy = self.stm_model.train_net()
        self.train_loss.append(loss)
        self.train_entropy.extend(entropy)
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
                stm_q_out = torch.nn.functional.softmax(self.stm_model.logits(s) / 0.5, dim=-1)

            ltm_q_out = torch.nn.functional.softmax(self.ltm_net(s), dim=-1)

            loss_curr_task = torch.nn.functional.mse_loss(ltm_q_out, stm_q_out)

            if not self.first_ltm_train:
                if self.kkk:
                    print("FIRST GAN TRANFER")
                    self.kkk = False
                with torch.no_grad():
                    gen_obs = self.gan.sample(self.batch_size)
                    prev_ltm_q_out_gen = torch.nn.functional.softmax(self.prev_ltm_net(gen_obs) / 0.2, dim=-1)

                ltm_q_out_gen = torch.nn.functional.softmax(self.ltm_net(gen_obs), dim=-1)

                loss_prev_task = torch.nn.functional.mse_loss(
                    ltm_q_out_gen, prev_ltm_q_out_gen
                )
            else:
                loss_prev_task = 0

            loss = self.alpha * loss_curr_task + (1 - self.alpha) * loss_prev_task

            self.ltm_optimizer.zero_grad()
            loss.backward()
            self.ltm_optimizer.step()
            self.train_loss.append(10000*loss.detach().item())
            self.train_entropy.append(Categorical(ltm_q_out).entropy().mean().detach().item())
            print(f"LTM Train | Loss:{10000*loss.detach().item():.8f}", end="\r")

    def train_gan(self):
        print(f"Buffer size: {self.ltm_replay.size()} - Batch: {self.batch_size}")
        avg_disc_loss = []
        avg_gen_loss = []
        total_iter = 10_000
        for i in range(total_iter):
            if random.random() < 1 / self.tasks_seen:
                real_samples = self.ltm_replay.sample(32)[0]
            else:
                real_samples = self.gan.sample(batch=32)

            disc_loss, gen_loss = self.new_gan.train_step(real_samples)
            avg_disc_loss.append(disc_loss)
            avg_gen_loss.append(gen_loss)
            print(
                f"GAN TRAIN [{i}/{total_iter}] | Disc: {sum(avg_disc_loss)/len(avg_disc_loss)} - Gen: {sum(avg_gen_loss)/len(avg_gen_loss)}",
                end="\r",
            )
            if (i+1) % 1000 == 0:
                with open("terminal.txt", "a") as f:
                    f.write(
                        f"GAN TRAIN [{i}/{total_iter}] | Disc: {sum(avg_disc_loss)/len(avg_disc_loss)} - Gen: {sum(avg_gen_loss)/len(avg_gen_loss)}\n"
                    )
                print(
                    f"GAN TRAIN [{i}/{total_iter}] | Disc: {sum(avg_disc_loss)/len(avg_disc_loss)} - Gen: {sum(avg_gen_loss)/len(avg_gen_loss)}",
                )
        print(
            f"GAN TRAIN [{i}/{total_iter}] | Disc: {sum(avg_disc_loss)/len(avg_disc_loss)} - Gen: {sum(avg_gen_loss)/len(avg_gen_loss)}"
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

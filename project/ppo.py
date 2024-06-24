import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 4e-4
gamma = 0.99
lmbda = 0.97
eps_clip = 0.2
K_epoch = 5


class PPO(nn.Module):
    def __init__(self, action_space=18):
        super(PPO, self).__init__()
        self.data = []
        print("INIT PPO")

        self.fc_pi = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

        self.fc_v = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.last_entropy = 0.0

    def pi(self, x, softmax_dim=-1):
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def logits(self, x):
        return self.fc_pi(x)

    def v(self, x):
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = (
            torch.from_numpy(np.array(s_lst)).float(),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.from_numpy(np.array(s_prime_lst)).float(),
            torch.tensor(done_lst, dtype=torch.float),
            torch.tensor(prob_a_lst),
        )

        del self.data
        self.data = []

        return s, a, r, s_prime, done_mask, prob_a

    def sample_action(self, x):
        prob = self.pi(x)
        m = Categorical(prob[0])
        a = m.sample()
        return a.detach().item(), prob[0, a].detach().item()

    def reset_lr(self):
        self.last_entropy = 0.
        self.optimizer.param_groups[0]["lr"] = learning_rate * 0.8


    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        acc_loss = []
        entropy_lst = []
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=-1)
            entropy = Categorical(pi).entropy()
            pi_a = pi.gather(1, a)
            ratio = torch.exp(
                torch.log(pi_a) - torch.log(prob_a)
            )  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(
                self.v(s), td_target.detach()
            )  # - 0.0001 * entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            acc_loss.append(loss.mean().detach().item())
            entropy_lst.extend(entropy.detach().tolist())

        curr_entropy = sum(entropy_lst) / len(entropy_lst)
        if self.last_entropy == 0.0:
            self.last_entropy = curr_entropy
        if self.last_entropy * 0.95 > curr_entropy:
            self.optimizer.param_groups[0]["lr"] *= 0.95
            curr_lr = self.optimizer.param_groups[0]["lr"]
            print(f"UPDATE LR: {curr_lr*1e4:.4f} E-4")
            self.last_entropy = curr_entropy
        return sum(acc_loss) / len(acc_loss), entropy_lst

    def save_checkpoint(self, dir):
        torch.save(
            {
                "pi_mode": self.fc_pi.state_dict(),
                "v_model": self.fc_v.state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            dir + "ppo.pt",
        )

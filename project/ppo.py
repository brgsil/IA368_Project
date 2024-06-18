import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.15
K_epoch = 1


class PPO(nn.Module):
    def __init__(self, action_space=18):
        super(PPO, self).__init__()
        self.data = []

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
        )

        self.fc_pi = nn.Linear(512, action_space)
        self.fc_v = nn.Linear(512, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=-1):
        assert x.max() <= 1
        assert x.min() >= -1
        x = F.relu(self.features(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def logits(self, x):
        assert x.max() <= 1
        assert x.min() >= -1
        return self.fc_pi(F.relu(self.features(x)))

    def v(self, x):
        assert x.max() <= 1
        assert x.min() >= -1
        x = F.relu(self.features(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(2 * s / 255.0 - 1)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(2 * s_prime / 255.0 - 1)
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
        #x = 2 * x / 255.0 - 1
        prob = self.pi(x)
        m = Categorical(prob[0])
        a = m.sample().item()
        return a, prob[0, a].item()

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        acc_loss = []
        entropy = 0
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

            pi = self.pi(s, softmax_dim=1)
            entropy = Categorical(pi).entropy().mean().detach().item()
            pi_a = pi.gather(1, a)
            ratio = torch.exp(
                torch.log(pi_a) - torch.log(prob_a)
            )  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                self.v(s), td_target.detach()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            acc_loss.append(loss.mean().detach().item())

        return sum(acc_loss) / len(acc_loss), entropy

    def save_checkpoint(self, dir):
        torch.save(
            {
                "features_model": self.features.state_dict(),
                "pi_mode": self.fc_pi.state_dict(),
                "v_model": self.fc_v.state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            dir + "ppo.pt",
        )


"""
def main():
    env = gym.make("CartPole-v1")
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data(
                    (s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score / print_interval
                )
            )
            score = 0.0

    env.close()


if __name__ == "__main__":
    main()
"""

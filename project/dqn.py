import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, size=100):
        self.buffer = [()] * size
        self.max_size = size
        self.idx = 0

    def put(self, transition):
        # Transition in the form (action, reward, terminal, next_state)
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.max_size

    def sample(self, n):
        idxs = random.sample(range(self.size - 4), n)
        s_lst, a_lst, r_lst, done_mask_lst, s_prime_lst = [], [], [], [], []

        for idx in idxs:
            a, r, done_mask, _ = self.buffer[idx]
            s = [s_ for _, _, _, s_ in self.bufer[idx - 4: idx]]
            s_prime = [s_ for _, _, _, s_ in self.bufer[idx - 3: idx + 1]]
            s_lst.append(torch.tensor(s))
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(torch.tensor(s_prime))
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, action_space=18):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3))
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(
        self,
        lr=0.00025,
        gamma=0.99,
        buffer=200_000,
        start_train=1000,
        batch_size=32,
        action_space=18,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_delta=100,
        train=True
    ):
        self.q_net = Qnet(action_space=action_space)

        # Initialize target QNet with base QNet parameters
        self.q_target = Qnet(action_space=action_space)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.replay = ReplayBuffer(size=buffer)
        self.start_buffer_size = start_train
        self.batch_size = batch_size

        # Epsilon annealing scheduling function
        self.epsilon = lambda step: max(
            epsilon_end,
            epsilon_start - 1.0 * step *
            (epsilon_start - epsilon_end) / epsilon_delta,
        )
        self.step_count = 0
        self.action_space = action_space

        self.train = train
        self.gamma = gamma
        self.optimizer = optim.RMSprop(
            self.q_net.parameters(), lr=lr, eps=1e-6)

    def sample_action(self, obs):
        if self.train:
            self.step_count += 1
            coin = random.random()
            if coin < self.epsilon(self.step_count):
                return random.randint(0, self.action_space - 1)
            else:
                out = self.q_net(obs)
                return out.argmax().item()
        else:
            coin = random.random()
            if coin < 0.1:
                return random.randint(0, self.action_space - 1)
            else:
                out = self.q_net(obs)
                return out.argmax().item()


    def logits(self, obs):
        return self.q_net(obs)

    def receive_transition(self, obs):
        self.replay.put(obs)

    def copy_nets(self, other_dqn):
        self.q_net.load_state_dict(other_dqn.q_net.state_dict())
        self.q_target.load_state_dict(other_dqn.q_target.state_dict())

    def train_step(self):
        if self.replay.size() > self.start_buffer_size:
            s, a, r, s_prime, done_mask = self.replay.sample(self.batch_size)

            with torch.no_grad():
                max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done_mask

            q_out = self.q_net(s)
            q_a = q_out.gather(1, a)
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

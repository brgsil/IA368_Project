import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, size=100):
        self.buffer = collections.deque(maxlen=size)

    def put(self, transition):
        # Transition in the form (state, action, reward, terminal, next_state)
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, done_mask_lst, s_prime_lst = [], [], [], [], []

        for transition in mini_batch:
            # Create samples
            s, a, r, done_mask, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([0 if done_mask else 1])

        return (
            torch.from_numpy(np.array(s_lst)).float(),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.from_numpy(np.array(s_prime_lst)).float(),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


class LinearQnet(nn.Module):
    def __init__(self, action_space=18):
        super(LinearQnet, self).__init__()
        self.action_space = action_space
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        return self.net(x)

    def sample_action(self, x, epsilon=0.1, mode="train"):
        if mode == "eval":
            epsilon = 0.0

        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            probs = self(x)
            return probs.argmax().item()


class DQN:
    def __init__(
        self,
        lr=0.00015,
        gamma=0.99,
        buffer=150_000,
        start_train=5_000,
        update_freq=2,
        update_target=2_000,
        batch_size=128,
        action_space=18,
        epsilon_start=1.0,
        epsilon_end=0.1,
        start_epsilon_decay=5_000,
        epsilon_delta=200_000,
        train=True,
    ):
        self.q_net = LinearQnet(action_space=action_space).to(device)

        # Initialize target QNet with base QNet parameters
        self.q_target = LinearQnet(action_space=action_space).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.replay = ReplayBuffer(size=buffer)
        self.buffer_size = buffer
        self.start_buffer_size = start_train
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.update_target = update_target

        # Epsilon annealing scheduling function
        self.epsilon = lambda step: min(
            1.0,
            max(
                epsilon_end,
                epsilon_start
                - 1.0
                * (step - start_epsilon_decay)
                * (epsilon_start - epsilon_end)
                / epsilon_delta,
            ),
        )
        self.step_count = 0
        self.last_loss = 0
        self.last_entropy = [
            Categorical(torch.tensor([1.0 / action_space] * action_space))
            .entropy()
            .item()
        ]
        self.action_space = action_space

        self.train = train
        self.gamma = gamma
        self.optimizer = optim.RMSprop(
            self.q_net.parameters(), lr=lr, eps=1e-6)

    def sample_action(self, obs):
        obs = obs.to(device)
        if self.train:
            self.step_count += 1
            return self.q_net.sample_action(obs, self.epsilon(self.step_count), "train")
        else:
            return self.q_net.sample_action(obs, self.epsilon(self.step_count), "eval")

    def logits(self, obs):
        obs = obs.to(device)
        return self.q_net(obs)

    def receive_transition(self, obs):
        self.replay.put(obs)

    def reset_buffer(self):
        self.replay = ReplayBuffer(size=self.buffer_size)

    def copy_nets(self, other_dqn):
        self.q_net.load_state_dict(other_dqn.q_net.state_dict())
        self.q_target.load_state_dict(other_dqn.q_target.state_dict())

    def train_step(self):
        # print(f"Train DQN step {self.replay.size()}")
        if self.replay.size() > self.start_buffer_size:
            if self.step_count % self.update_target == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())

            if self.step_count % self.update_freq == 0:
                s, a, r, s_prime, done_mask = self.replay.sample(
                    self.batch_size)

                with torch.no_grad():
                    max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
                    target = r + self.gamma * max_q_prime * done_mask

                q_out = self.q_net(s)
                self.last_entropy = (
                    Categorical(logits=q_out).entropy().detach().tolist()
                )
                q_a = q_out.gather(1, a)
                loss = F.smooth_l1_loss(q_a, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.last_loss = 1000*loss.detach().item()

        return self.last_loss, self.last_entropy

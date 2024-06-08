import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, size=100):
        self.buffer = collections.deque(maxlen=size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
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
    def __init__(self, action_space=2):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, (8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3))
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, action_space)
        else:
            return out.argmax().item()


class DQN:
    def __init__(self, lr=0.00025, gamma=0.99, buffer=200_000, batch_size=32, action_space=18, epsilon_start=1.0, epsilon_end=0.1, epsilon_delta=100):
        self.q_net = Qnet(action_space=action_space)

        # Initialize target QNet with base QNet parameters
        self.q_target = Qnet(action_space=action_space)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.replay = ReplayBuffer(size=buffer)
        self.batch_size = batch_size

        # Epsilon annealing scheduling function
        self.epsilon = lambda step: max(
            epsilon_end,
            epsilon_start - 1.0 * step * (epsilon_start - epsilon_end) / epsilon_delta)
        self.step_count = 0

        self.optimizer = optim.RMSprop(self.q_target.parameters(), lr=lr)

    def sample_action(self, obs):
        self.step_count += 1
        return self.q_net.sample_action(obs, self.epsilon(self.step_count))

    def receive_transition(self, obs):
        self.replay.put(obs)

    def train_step(self):
        s, a, r, s_prime, done_mask = self.replay.sample(self.batch_size)

        with torch.no_grad():
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        q_out = self.q_net(s)
        q_a = q_out.gather(1, a)
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

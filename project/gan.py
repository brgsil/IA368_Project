import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN:
    def __init__(self, tradeoff=0.1, lr=0.0002, drift=1e-6):
        super(GAN, self).__init__()

        # self.gen = nn.Sequential(
        #    nn.Linear(100, 256 * 7 * 7),
        #    nn.BatchNorm1d(256 * 7 * 7, momentum=0.9, eps=1e-5),
        #    nn.ReLU(),
        #    nn.Unflatten(-1, (256, 7, 7)),
        #    nn.ConvTranspose2d(256, 256, kernel_size=(5, 5),
        #                       stride=(3, 3), padding=1),
        #    nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(256, 128, kernel_size=(5, 5),
        #                       stride=(2, 2), padding=1),
        #    nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(128, 64, kernel_size=(5, 5),
        #                       stride=(2, 2), padding=2),
        #    nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(64, 4, kernel_size=(4, 4),
        #                       stride=(1, 1), padding=2),
        #    nn.Tanh(),
        # ).to(device)
        self.gen = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        # self.disc = nn.Sequential(
        #    nn.Conv2d(4, 64, kernel_size=(5, 5), stride=(3, 3)),
        #    nn.LeakyReLU(0.2),
        #    nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2)),
        #    nn.LeakyReLU(0.2),
        #    nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2)),
        #    nn.LeakyReLU(0.2),
        #    nn.Flatten(),
        #    nn.Linear(256 * 4 * 4, 1),
        # ).to(device)
        self.disc = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

        self.gen_optim = optim.Adam(self.gen.parameters(), lr=lr)
        self.disc_optim = optim.Adam(self.disc.parameters(), lr=lr)

        self.tradeoff = tradeoff
        self.drift = drift

    def sample(self, batch=1):
        noise = torch.rand((batch, 2))
        noise = noise.to(device)
        samples = self.gen(noise)
        return samples

    def classify(self, x):
        x = x.to(device)
        return self.disc(x)

    def copy_from(self, other):
        self.gen.load_state_dict(other.gen.state_dict())
        self.disc.load_state_dict(other.disc.state_dict())

    def grad_penality(self, real_samples, fake_samples):
        batch = real_samples.shape[0]
        alpha = torch.rand(batch, 1).expand(real_samples.shape)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated = interpolated.requires_grad_(True)

        interpolated_pred = self.classify(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones(interpolated_pred.shape),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch, -1)

        grad_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        return ((grad_norm - 1) ** 2).mean()

    def train_step(self, real_samples):
        real_samples = real_samples.to(device)
        batch_size = real_samples.shape[0]

        fake_samples = self.sample(batch=batch_size).to(device)

        real_pred = self.classify(real_samples)
        fake_pred = self.classify(fake_samples)

        disc_loss = (
            fake_pred.mean()
            - real_pred.mean()
            + self.tradeoff * self.grad_penality(real_samples, fake_samples)
            + self.drift * ((real_pred**2).mean() + (fake_pred**2).mean())
        )

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        fake_samples = self.sample(batch=batch_size)
        fake_pred = self.classify(fake_samples)

        gen_loss = -fake_pred.mean()

        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        return disc_loss.detach().item(), gen_loss.detach().item()

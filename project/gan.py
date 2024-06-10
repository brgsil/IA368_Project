import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GAN:
    def __init__(self, lambda=0.1, lr=0.001, drift=1e-6):
        super(GAN, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(100, 256 * 7 * 7),
            nn.BatchNorm1d(momentum=0.9, eps=1e-5),
            nn.ReLU(),
            nn.Unflatten(-1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 256, kernel_size=(5, 5), stride=(3, 3)),
            nn.BatchNorm1d(momentum=0.9, eps=1e-5),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm1d(momentum=0.9, eps=1e-5),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm1d(momentum=0.9, eps=1e-5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
        )

        self.disc = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(5, 5), stride=(3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(256 * 4 * 4, 1),
        )

        self.gen_optim = optim.Adam(
            self.gen.parameters(), lr=lr, betas=(0, 0.99))
        self.disc_optim = optim.Adam(
            self.disc.parameters(), lr=lr, betas=(0, 0.99))

        self.lambda = lamda
        self.drift = drift

    def sample(self, batch=1):
        noise = torch.rand((batch, 100)) * 2 - 1
        samples = self.gen(noise)
        return samples

    def classify(self, x):
        return self.disc(x)

    def grad_penality(self, real_samples, fake_samples):
        batch = real_samples.shape[0]
        alpha = torch.rand(batch, 1, 1, 1).expand(real_samples.shape)
        interpolated = alpha * real_samples + (1-alpha) * fake_samples
        interpolated = interpolated.requires_grad_(True)

        interpolated_pred = self.classify(interpolated)

        gradients = torch.autograd.grad(outputs=interpolated_pred,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones(
                                            interpolated_pred.shape),
                                        create_graph=True,
                                        retain_graph=True)[0]
        gradients = gradients.reshape(batch, -1)

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def train_step(self, real_samples):
        batch_size = real_samples.shape[0]

        fake_samples = self.sample(batch=batch_size)

        real_pred = self.classify(real_samples)
        fake_pred = self.classify(fake_samples)

        disc_loss = (fake_pred.mean() - real_pred.mean()
                     + self.lambda * self.grad_penality(real_samples, fake_samples)
                     + self.drift * ((real_pred**2).mean() + (fake_pred**2).mean()))
        gen_loss = - fake_pred.mean()

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

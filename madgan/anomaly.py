import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(object):

    def __init__(self,
                 *,
                 discriminator: nn.Module,
                 generator: nn.Module,
                 latent_space_dim: int,
                 device: torch.device,
                 res_weight: float = .2,
                 anomaly_threshold: float = 0.9,
                 max_iter_for_reconstruct: int = 1000) -> None:
        self.discriminator = discriminator
        self.generator = generator
        self.threshold = anomaly_threshold
        self.latent_space_dim = latent_space_dim
        self.res_weight = res_weight
        self.device = device
        self.max_iter_for_reconstruct = max_iter_for_reconstruct

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        predict = self.predict_proba(tensor)
        print(
            f'Pred float range: {predict.max().item(), predict.min().item()}')
        return (predict > self.threshold).float()

    def predict_proba(self, tensor: torch.Tensor) -> torch.Tensor:
        discriminator_score = self.compute_anomaly_score(tensor)
        discriminator_score *= 1. - self.res_weight
        reconstruction_loss = self.compute_reconstruction_loss(tensor).view_as(
            discriminator_score)
        reconstruction_loss *= self.res_weight
        return reconstruction_loss + discriminator_score

    def compute_anomaly_score(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            discriminator_score, logits = self.discriminator(tensor)
        return discriminator_score

    def compute_reconstruction_loss(self,
                                    tensor: torch.Tensor) -> torch.Tensor:
        best_reconstruct = self._generate_best_reconstruction(tensor)
        return (best_reconstruct - tensor).abs().sum(dim=(-1))

    def _generate_best_reconstruction_v1(self, x):
        z = torch.randn(x.size(0), x.size(
            1), self.latent_space_dim).to(self.device)
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=0.01)
        loss_fn = nn.MSELoss(reduction="mean")

        for _ in range(10):  # Perform 1000 optimization steps
            optimizer.zero_grad()
            self.generator.train()
            x_gen = self.generator(z)
            loss = loss_fn(x_gen, x)  # MSE loss
            loss.backward()
            optimizer.step()

        self.generator.eval()

        return self.generator(z).detach()

    def _generate_best_reconstruction(self, tensor: torch.Tensor) -> None:
        # The goal of this function is to find the corresponding latent space for the given
        # input and then generate the best possible reconstruction.
        Z = torch.randn(
            (tensor.size(0), tensor.size(1), self.latent_space_dim)).to(self.device)
        Z.requires_grad = True
        optimizer = torch.optim.Adam([Z], lr=0.01)
        # optimizer = torch.optim.RMSprop(params=[Z], lr=0.01)
        loss_fn = nn.MSELoss(reduction="sum")
        normalized_target = F.normalize(tensor, dim=1, p=2)

        for _ in range(self.max_iter_for_reconstruct):
            optimizer.zero_grad()
            self.generator.train()
            generated_samples = self.generator(Z)
            normalized_input = F.normalize(generated_samples, dim=1, p=2)
            reconstruction_error = loss_fn(normalized_input,
                                           normalized_target)  # .sum(dim=(1, 2))
            reconstruction_error.backward()
            optimizer.step()

        self.generator.eval()

        with torch.no_grad():
            best_reconstruct = self.generator(Z)
        return best_reconstruct

import random
from typing import Callable, Dict, Iterator, List

import numpy as np
import torch
import torch.nn as nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(generator: nn.Module,
                    discriminator: nn.Module,
                    loss_fn: LossFn,
                    device: torch.device,
                    real_dataloader: Iterator[torch.Tensor],
                    latent_dataloader: Iterator[torch.Tensor],
                    discriminator_optimizer: torch.optim.Optimizer,
                    generator_optimizer: torch.optim.Optimizer,
                    normal_label: int = 0,
                    anomaly_label: int = 1,
                    epoch: int = 0,
                    epochs: int = 0,
                    log_every: int = 30,
                    g_loss_records: List[float] = None,
                    d_loss_records: List[float] = None) -> None:
    """Trains a GAN for a single epoch.

    Args:
        generator (nn.Module): Torch module implementing the GAN generator.
        discriminator (nn.Module): Torch module implementing the GAN
            discriminator.
        loss_fn (LossFn): Loss function, should return a reduced value.
        real_dataloader (Iterator[torch.Tensor]): Iterator to go over real data
            samples.
        latent_dataloader (Iterator[torch.Tensor]): Iterator to go through
            generated samples from the latent space.
        discriminator_optimizer (torch.optim.Optimizer): Optimizer for the
            discrimninator.
        generator_optimizer (torch.optim.Optimizer): Oprimizer for the generator
            module.
        normal_label (int): Label for samples with normal behaviour
            (real or non-anomaly). Defaults to 0.
        anomaly_label (int): Label that identifies generate samples
            (anomalies when running inference). Defaults to 1.
        epoch (int, optional): Current epoch (just for logging purposes).
            Defaults to 0.
        log_every (int, optional): Log the training progess every n steps.
            Defaults to 30.
    """
    generator.train()
    discriminator.train()

    for i, (real, z) in enumerate(zip(real_dataloader, latent_dataloader)):
        bs = real.size(0)
        # At the end of the epoch, the last batch might be smaller
        if z.shape[0] > bs:
            z = z[:bs]
        real = real.float().to(device)
        z = z.float().to(device)

        # Generate fake samples with the generator
        fake = generator(z)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        discriminator.zero_grad()
        real_output, real_logits = discriminator(real)
        fake_output, fake_logits = discriminator(fake.detach())
        # Create labels for the real and fake samples
        real_labels = torch.full(
            real_output.shape, normal_label, dtype=torch.float, device=device)
        fake_labels = torch.full(
            fake_output.shape, anomaly_label, dtype=torch.float, device=device)

        # Discriminator tries to identify the true nature of each sample
        # (anomaly or normal)
        d_real_loss = loss_fn(real_output, real_labels)
        d_fake_loss = loss_fn(fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        # Compute the gradients of discriminator's loss
        d_loss.backward()
        D_x = real_output.mean().item()
        D_G_z1 = fake_output.mean().item()
        # Update the discriminator
        discriminator_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        fake_out_g, fake_logits_g = discriminator(fake)

        # Generator will improve so it can cheat the discriminator
        cheat_loss = loss_fn(fake_out_g, real_labels)
        cheat_loss.backward()
        D_G_z2 = fake_out_g.mean().item()
        # Update the generator
        generator_optimizer.step()

        d_loss_records.append(d_loss.item())
        g_loss_records.append(cheat_loss.item())
        if (i + 1) % log_every == 0:
            # Evaluate the training accuracy
            d_output = torch.cat([real_output, fake_output])
            all_labels = torch.cat([real_labels, fake_labels])
            d_preds = (d_output > .5).float()
            discriminator_acc = (d_preds == all_labels).float().mean()

            g_preds = (fake_output > .5).float()
            generator_acc = (g_preds == real_labels).float().mean()

            print(
                f"Epoch [{epoch}/{epochs}], Step [{i}/{len(real_dataloader)}], "
                f"Loss_D: {d_loss.item():.4f}, Loss_G: {cheat_loss.item():.4f}, "
                f"D(x): {D_x: .6f}, D(G(z)): {D_G_z1: .6f} / {D_G_z2: .6f},"
                f" Discriminator acc: {discriminator_acc: .6f}, Generator acc: {generator_acc: .6f}")

    discriminator.zero_grad()
    generator.zero_grad()


@torch.no_grad()
def evaluate(generator: nn.Module,
             device: torch.device,
             discriminator: nn.Module,
             loss_fn: LossFn,
             real_dataloader: Iterator[torch.Tensor],
             latent_dataloader: Iterator[torch.Tensor],
             normal_label: int = 0,
             anomaly_label: int = 1) -> Dict[str, float]:
    """Evaluates a trained GAN.

    Reports the real and fake losses for the discriminator, as well as the
    accuracies.

    Args:
        generator (nn.Module): Torch module implementing the GAN generator.
        discriminator (nn.Module): Torch module implementing the GAN 
            discriminator.
        loss_fn (LossFn): Loss function, should return a reduced value.
        real_dataloader (Iterator[torch.Tensor]): Iterator to go over real data 
            samples.
        latent_dataloader (Iterator[torch.Tensor]): Iterator to go through 
            generated samples from the latent space.
        normal_label (int): Label for samples with normal behaviour 
            (real or non-anomaly). Defaults to 0.
        anomaly_label (int): Label that identifies generate samples 
            (anomalies when running inference). Defaults to 1.

    Returns:
        Dict[str, float]: Aggregated metrics.
    """
    generator.eval()
    discriminator.eval()

    agg_metrics: Dict[str, float] = {}
    for real, z in zip(real_dataloader, latent_dataloader):
        bs = real.size(0)
        # At the end of the epoch, the last batch might be smaller
        if z.shape[0] > bs:
            z = z[:bs]
        real = real.float().to(device)
        z = z.float().to(device)
        real_labels = torch.full((bs, ), normal_label).float().to(device)
        fake_labels = torch.full((bs, ), anomaly_label).float().to(device)
        all_labels = torch.cat([real_labels, fake_labels])

        # Generate fake samples with the generator
        fake = generator(z)

        # Try to classify the real and generated samples
        real_output, real_logits = discriminator(real)
        fake_output, fake_logits = discriminator(fake.detach())
        d_output = torch.cat([real_output, fake_output])
        real_labels = torch.full(
            real_output.shape, normal_label, dtype=torch.float, device=device)
        fake_labels = torch.full(
            fake_output.shape, anomaly_label, dtype=torch.float, device=device)
        all_labels = torch.cat([real_labels, fake_labels])

        # Discriminator tries to identify the true nature of each sample
        # (anomaly or normal)
        d_real_loss = loss_fn(real_output, real_labels)
        d_fake_loss = loss_fn(fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss

        # Discriminator accuracy
        discriminator_acc = (
            (d_output > .5) == all_labels).float().mean()

        # Generator accuracy
        generator_acc = ((fake_output > .5)
                         == real_labels).float().sum()

        log = {
            "discriminator_real_loss": d_real_loss.item(),
            "discriminator_fake_loss": d_fake_loss.item(),
            "discriminator_loss": d_loss.item(),
            "discriminator_acc": discriminator_acc.item(),  # Add this line
            "generator_acc": generator_acc.item(),  # Add this line
        }

        if not agg_metrics:
            agg_metrics = log
        else:
            agg_metrics = {
                metric_name: (agg_metrics[metric_name] + metric_value) / 2.
                for metric_name, metric_value in log.items()
            }

    log_metrics = " ".join(
        f"{metric_name}:{metric_value:.4f}"
        for metric_name, metric_value in agg_metrics.items())
    print("Evaluation metrics:", log_metrics)
    return agg_metrics

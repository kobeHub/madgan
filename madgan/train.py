from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
import torch
import pytorch_model_summary as pms

import madgan
from madgan import constants

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    input_data: str,
    batch_size: int = constants.BATCH_SIZE,
    epochs: int = constants.EPOCHS,
    lr: float = constants.LEARNING_RATE,
    hidden_dim: int = constants.HIDDEN_DIM,
    window_size: int = constants.WINDOW_SIZE,
    window_stride: int = constants.WINDOW_STRIDE,
    add_batch_mean: bool = constants.ADD_BATCH_MEAN,
    random_seed: int = constants.RANDOM_SEED,
    model_dir: Path = Path("models"),
    log_every: int = 30,
) -> None:

    madgan.engine.set_seed(random_seed)

    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_data)
    train_dl, test_dl = _prepare_data(df=df,
                                      batch_size=batch_size,
                                      window_size=window_size,
                                      window_stride=window_stride)
    latent_space = madgan.data.LatentSpaceIterator(noise_shape=[
        batch_size,
        window_size,
        constants.LATENT_SPACE_DIM,
        # df.shape[-1],
    ])

    generator = madgan.models.Generator(
        latent_space_dim=constants.LATENT_SPACE_DIM,
        hidden_units=hidden_dim,
        output_dim=df.shape[-1])
    generator.to(DEVICE)
    pms.summary(generator, torch.zeros((batch_size, window_size, constants.LATENT_SPACE_DIM)).to(DEVICE),
                show_input=True, batch_size=batch_size, print_summary=True)

    # Handle adding batch mean to the discriminator
    input_d = df.shape[-1]
    if add_batch_mean:
        input_d *= 2
    discriminator = madgan.models.Discriminator(input_dim=input_d,
                                                hidden_units=hidden_dim,
                                                add_batch_mean=add_batch_mean)
    generator.to(DEVICE)
    pms.summary(discriminator, torch.zeros((batch_size, window_size,
                                            df.shape[-1])).to(DEVICE), batch_size=batch_size, show_input=True, print_summary=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    generator_optim = torch.optim.Adam(generator.parameters(), lr=lr)

    criterion_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        madgan.engine.train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loss_fn=criterion_fn,
            device=DEVICE,
            real_dataloader=train_dl,
            latent_dataloader=latent_space,
            discriminator_optimizer=discriminator_optim,
            generator_optimizer=generator_optim,
            normal_label=constants.REAL_LABEL,
            anomaly_label=constants.FAKE_LABEL,
            epoch=epoch,
            log_every=log_every)

    madgan.engine.evaluate(generator=generator,
                           discriminator=discriminator,
                           real_dataloader=test_dl,
                           latent_dataloader=latent_space,
                           loss_fn=criterion_fn,
                           normal_label=constants.REAL_LABEL,
                           anomaly_label=constants.FAKE_LABEL)

    generator.save(model_dir / f"generator_{epoch}.pt")
    discriminator.save(model_dir / f"discriminator_{epoch}.pt")


def _prepare_data(
    df: pd.DataFrame,
    batch_size: int,
    window_size: int,
    window_stride: int,
) -> Tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]]:
    dataset = madgan.data.WindowDataset(df,
                                        window_size=window_size,
                                        window_slide=window_stride)

    indices = torch.randperm(len(dataset))
    train_len = int(len(dataset) * .8)
    train_dataset = torch.utils.data.Subset(dataset,
                                            indices[:train_len].tolist())
    test_dataset = torch.utils.data.Subset(dataset,
                                           indices[train_len:].tolist())

    train_dl = madgan.data.prepare_dataloader(train_dataset,
                                              batch_size=batch_size)
    test_dl = madgan.data.prepare_dataloader(test_dataset,
                                             batch_size=batch_size)
    return train_dl, test_dl

from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
import torch
import pytorch_model_summary as pms

import madgan
from madgan.visual import plot_losses

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    config_file: str,
) -> None:
    config = madgan.utils.read_config(config_file)
    print(f'Config for the training: {config}')
    random_seed = config.get('random_seed')
    batch_size = config.get('batch_size')
    window_size = config.get('window_size')
    window_stride = config.get('window_stride')

    madgan.engine.set_seed(random_seed)

    model_dir = Path(config.get('model_dir'))
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config['input_data'])
    if df.columns.str.contains("label").any():
        df.drop(columns=["label"], inplace=True)
        print(f"Dropped label column for training")
    print(
        f'Data shape: {df.shape}, batch size: {batch_size}, window size: {window_size}, window stride: {window_stride}\nCol: {df.columns}')
    # The output_dim of the generator and the input_dim of the discriminator
    n_features = config['n_features']
    latent_space_dim = config['latent_space_dim']
    train_dl, test_dl = _prepare_data(df=df,
                                      batch_size=batch_size,
                                      window_size=window_size,
                                      window_stride=window_stride,
                                      n_skip_size=config['n_skip_size'],
                                      n_features=n_features)
    latent_space = madgan.data.LatentSpaceIterator(noise_shape=[
        batch_size,
        window_size,
        latent_space_dim,
    ])

    hidden_dim = config['hidden_dim']
    add_batch_mean = config.get('add_batch_mean', False)
    generator = madgan.models.Generator(
        window_size=window_size,
        latent_space_dim=latent_space_dim,
        hidden_units=hidden_dim,
        output_dim=n_features,
        n_lstm_layers=latent_space_dim)
    generator.to(DEVICE)
    print(f'Generator summary:')
    pms.summary(generator, torch.zeros((batch_size, window_size, latent_space_dim)).to(DEVICE),
                show_input=True, batch_size=batch_size, print_summary=True)

    # Handle adding batch mean to the discriminator
    input_d = n_features
    if add_batch_mean:
        input_d *= 2
    discriminator = madgan.models.Discriminator(input_dim=input_d,
                                                hidden_units=hidden_dim,
                                                n_lstm_layers=config['d_lstm_layers'],
                                                add_batch_mean=add_batch_mean)
    discriminator.to(DEVICE)
    print(f'\nDiscriminator summary:')
    pms.summary(discriminator, torch.zeros((batch_size, window_size, n_features)).to(DEVICE),
                batch_size=batch_size, show_input=True, print_summary=True)

    discriminator_optim = torch.optim.Adam(
        discriminator.parameters(), lr=config['d_lr'])
    generator_optim = torch.optim.Adam(
        generator.parameters(), lr=config['g_lr'])

    criterion_fn = torch.nn.BCELoss()

    # Record loss to plot later
    g_loss_records = []
    d_loss_records = []

    epochs = config.get('epochs')
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
            epoch=epoch,
            config=config,
            g_loss_records=g_loss_records,
            d_loss_records=d_loss_records)

    madgan.engine.evaluate(generator=generator,
                           device=DEVICE,
                           discriminator=discriminator,
                           real_dataloader=test_dl,
                           latent_dataloader=latent_space,
                           loss_fn=criterion_fn,
                           normal_label=config['normal_label'],
                           anomaly_label=config['anomaly_label'])

    generator.save(model_dir / f"generator_{epoch}.pt")
    discriminator.save(model_dir / f"discriminator_{epoch}.pt")
    plot_losses(g_loss_records, d_loss_records,
                model_dir / f"loss_{epoch}.png")


def _prepare_data(
    df: pd.DataFrame,
    batch_size: int,
    window_size: int,
    window_stride: int,
    n_skip_size: int,
    n_features: int,
) -> Tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]]:
    # PCA feature extraction
    df = madgan.data.feature_extract_without_label(
        df, skip_size=n_skip_size, n_features=n_features)
    print(f'Feature extracted data shape: {df.shape}')
    dataset = madgan.data.WindowDataset(df,
                                        window_size=window_size,
                                        window_slide=window_stride)
    # Split the dataset into training and testing
    indices = torch.randperm(len(dataset))
    train_len = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(
        dataset, indices[:train_len].tolist())
    test_dataset = torch.utils.data.Subset(
        dataset, indices[train_len:].tolist())

    train_dl = madgan.data.prepare_dataloader(
        train_dataset, batch_size=batch_size)
    test_dl = madgan.data.prepare_dataloader(
        test_dataset, batch_size=batch_size)
    return train_dl, test_dl

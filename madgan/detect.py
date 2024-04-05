from pathlib import Path
from typing import Iterator, Tuple, Union
from madgan import constants
import os
import torch
import madgan
from madgan.models import Discriminator, Generator
from madgan.anomaly import AnomalyDetector

import pandas as pd


def detect(model_path: str = './models/madgan',
           attack_csv: str = './data/swat_data_attack.csv',
           batch_size: int = 32,
           anomaly_threshold: float = 1.0):
    """Detect anomalies in the input data using the MAD-GAN model.

    Args:
        model_path (str): Path to the MAD-GAN model.
        anomaly_threshold (float, optional): Anomaly threshold. Defaults to 1.0.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test data
    df = pd.read_csv(attack_csv)
    print(f"Test data shape: {df.shape}")
    features = df.columns.tolist().pop(-1)
    samples, labels = df[features], df['label']
    print(f'Class distribution: {labels.value_counts()}')
    print(f"Test data samples: {samples.shape}, labels: {labels.shape}")
    test_dl = _prepare_data(df, batch_size=batch_size, window_size=constants.WINDOW_SIZE,
                            window_stride=constants.WINDOW_STRIDE)

    pts_files = [os.path.join(model_path, f) for f in os.listdir(model_path)]
    pts_files.sort(key=os.path.getmtime, reverse=True)
    latest_files = pts_files[:2]
    dis_path = [file for file in latest_files if 'discriminator' in file][0]
    gen_path = [model for model in latest_files if 'generator' in model][0]
    print(f"Dis_path {dis_path}; Gen_path {gen_path}")

    generator = Generator.from_pretrained(gen_path, DEVICE)
    discriminator = Discriminator.from_pretrained(dis_path, DEVICE)

    print("Models loaded successfully.")

    detector = AnomalyDetector(discriminator=discriminator, generator=generator,
                               latent_space_dim=constants.LATENT_SPACE_DIM,
                               anomaly_threshold=anomaly_threshold)


def _prepare_data(df: pd.DataFrame, batch_size: int, window_size: int,
                  window_stride: int) -> Iterator[torch.Tensor]:
    dataset = madgan.data.WindowDataset(df, window_size=window_size,
                                        window_slide=window_stride)
    dl = madgan.data.prepare_dataloader(dataset, batch_size=batch_size)
    return dl

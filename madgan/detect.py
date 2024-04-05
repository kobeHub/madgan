from pathlib import Path
from typing import Tuple, Union
from madgan import constants
import os
import torch
from madgan.models import Discriminator, Generator
from madgan.anomaly import AnomalyDetector


def detect(model_path: str = './models/madgan', anomaly_threshold: float = 1.0):
    """Detect anomalies in the input data using the MAD-GAN model.

    Args:
        model_path (str): Path to the MAD-GAN model.
        anomaly_threshold (float, optional): Anomaly threshold. Defaults to 1.0.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pts_files = [os.path.join(model_path, f) for f in os.listdir(model_path)]
    pts_files.sort(key=os.path.getmtime, reverse=True)
    latest_files = pts_files[:2]
    dis_path = [file for file in latest_files if 'discriminator' in file][0]
    gen_path = [model for model in latest_files if 'generator' in model][0]
    print(f"Dis_path {dis_path}; Gan_path {gen_path}")

    generator = Generator.from_pretrained(gen_path, DEVICE)
    discriminator = Discriminator.from_pretrained(dis_path, DEVICE)

    print("Models loaded successfully.")

    detector = AnomalyDetector(discriminator=discriminator, generator=generator,
                               latent_space_dim=constants.LATENT_SPACE_DIM,
                               anomaly_threshold=anomaly_threshold)

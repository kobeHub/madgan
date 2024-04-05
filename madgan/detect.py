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
           anomaly_threshold: float = 0.95,
           max_iter_for_reconstruct: int = 1000,
           print_every: int = 30):
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

    detector = AnomalyDetector(discriminator=discriminator, generator=generator, device=DEVICE,
                               latent_space_dim=constants.LATENT_SPACE_DIM,
                               anomaly_threshold=anomaly_threshold,
                               res_weight=0.,
                               max_iter_for_reconstruct=max_iter_for_reconstruct)
    total_samples = 0
    correct_predictions = 0
    for i, (x, y) in enumerate(test_dl):
        x = x.float().to(DEVICE)
        y = y.float().to(DEVICE)
        detect_res = detector.predict(x)

        # Convert predictions to binary labels
        pred_labels = (detect_res > detector.anomaly_threshold).float()

        # Update counters
        total_samples += y.size(0)
        correct_predictions += (pred_labels == y).sum().item()

        if i % print_every == 0:
            # Calculate metrics
            accuracy = correct_predictions / total_samples
            precision, recall, f1 = calculate_metrics(pred_labels, y)

            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

            print(f"Anomaly Detection Batch [{i+1}/{len(test_dl)}]: Accuracy: {accuracy:.4f}"
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


def calculate_metrics(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score.

    Args:
        pred_labels (torch.Tensor): Predicted labels.
        true_labels (torch.Tensor): True labels.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """
    true_positives = ((pred_labels == 1) & (true_labels == 1)).sum().item()
    false_positives = ((pred_labels == 1) & (true_labels == 0)).sum().item()
    false_negatives = ((pred_labels == 0) & (true_labels == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def _prepare_data(df: pd.DataFrame, batch_size: int, window_size: int,
                  window_stride: int) -> Iterator[torch.Tensor]:
    dataset = madgan.data.WindowDataset(df, window_size=window_size,
                                        window_slide=window_stride, use_label=True)
    dl = madgan.data.prepare_dataloader(dataset, batch_size=batch_size)
    return dl

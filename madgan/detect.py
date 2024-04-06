from pathlib import Path
from typing import Iterator, Tuple, Union
from madgan import constants
import os
import torch
import madgan
from madgan.models import Discriminator, Generator
from madgan.anomaly import AnomalyDetector

import pandas as pd
import torchmetrics.functional as F


def detect(model_path: str = './models/madgan',
           attack_csv: str = './data/swat_test.csv',
           batch_size: int = 32,
           anomaly_threshold: float = 0.8,
           max_iter_for_reconstruct: int = 10,
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
    latest_files = pts_files[:3]
    dis_path = [file for file in latest_files if 'discriminator' in file][0]
    gen_path = [model for model in latest_files if 'generator' in model][0]
    print(f"Dis_path {dis_path}; Gen_path {gen_path}")

    generator = Generator.from_pretrained(gen_path, DEVICE).to(DEVICE)
    discriminator = Discriminator.from_pretrained(dis_path, DEVICE).to(DEVICE)

    print("Models loaded successfully.")

    detector = AnomalyDetector(discriminator=discriminator, generator=generator, device=DEVICE,
                               latent_space_dim=constants.LATENT_SPACE_DIM,
                               anomaly_threshold=anomaly_threshold,
                               res_weight=0.00002,
                               max_iter_for_reconstruct=max_iter_for_reconstruct)
    total_samples = 0
    correct_predictions = 0
    for i, (x, y) in enumerate(test_dl):
        x = x.float().to(DEVICE)
        y = y.float().to(DEVICE)

        detect_res = detector.predict(x)

        # Convert predictions to binary labels
        pred_labels = (detect_res > anomaly_threshold).float().view_as(y)

        # Update counters
        total_samples += y.size(0) * y.size(1)
        correct_predictions += (pred_labels == y).sum().item()
        # print(f'Cor: {correct_predictions}, total: {total_samples}')

        if i % print_every == 0:
            # Calculate metrics
            accuracy = correct_predictions / total_samples
            precision, recall, f1 = calculate_metrics(pred_labels, y)
            print(f"Anomaly Detection Batch [{i+1}/{len(test_dl)}]: Accuracy: {accuracy:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


def calculate_metrics(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score.

    Args:
        pred_labels (torch.Tensor): Predicted labels.
        true_labels (torch.Tensor): True labels.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """
    # Reshape labels
    pred_labels = pred_labels.view(-1)
    true_labels = true_labels.view(-1)
    if torch.any(true_labels == 1):
        print(f'The true labels count: {true_labels.count_nonzero()},'
              f' The predicted labels count: {pred_labels.count_nonzero()}, '
              f'True positive count: {(pred_labels == true_labels).sum()}, '
              f'False positive count: {(pred_labels != true_labels).sum()}, '
              f'False negative count: {(pred_labels != true_labels).sum()}, '
              f'True negative count: {(pred_labels == true_labels).sum()}')

    # Compute metrics
    precision = F.precision(pred_labels, true_labels,
                            num_classes=2, task='binary')
    recall = F.recall(pred_labels, true_labels, num_classes=2, task='binary')
    f1 = F.f1_score(pred_labels, true_labels, num_classes=2, task='binary')

    return precision, recall, f1


def _prepare_data(df: pd.DataFrame, batch_size: int, window_size: int,
                  window_stride: int) -> Iterator[torch.Tensor]:
    labels = df.pop('label')
    df = madgan.data.feature_extract(df, skip_size=0,
                                     n_features=constants.N_FEATURES)
    df.loc[:, 'label'] = labels
    print(f'Feature extracted data shape: {df.shape}')
    dataset = madgan.data.WindowDataset(df, window_size=window_size,
                                        window_slide=window_stride, use_label=True)
    dl = madgan.data.prepare_dataloader(dataset, batch_size=batch_size)
    return dl

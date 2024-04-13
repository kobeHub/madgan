from typing import Iterator, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import madgan
from madgan.models import Discriminator, Generator
from madgan.anomaly import AnomalyDetector

import pandas as pd
import torchmetrics.functional as F


def detect(config_file: str = './config/swat-test-config.yaml'):
    """Detect anomalies in the input data using the MAD-GAN model.

    Args:
        model_path (str): Path to the MAD-GAN model.
        anomaly_threshold (float, optional): Anomaly threshold. Defaults to 1.0.
    """
    config = madgan.utils.read_config(config_file)
    print(f'Config for the detection:\n{config}')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size']
    window_size = config['window_size']
    window_stride = config['window_stride']
    n_features = config['n_features']
    latent_space_dim = config['latent_space_dim']

    # Load the test data
    df = pd.read_csv(config['attack_data'])
    print(f"Test data shape: {df.shape}")
    features = df.columns.tolist().pop(-1)
    samples, labels = df[features], df['label']
    print(f'Class distribution: {labels.value_counts()}')
    print(f"Test data samples: {samples.shape}, labels: {labels.shape}")
    test_dl = _prepare_data(df, config)

    dis_path = config['d_model_path']
    gen_path = config['g_model_path']
    print(f"Dis_path {dis_path}; Gen_path {gen_path}")

    generator = Generator.from_pretrained(gen_path, DEVICE).to(DEVICE)
    discriminator = Discriminator.from_pretrained(dis_path, DEVICE).to(DEVICE)

    print("Models loaded successfully.")

    anomaly_threshold = config['anomaly_threshold']
    detector = AnomalyDetector(discriminator=discriminator, generator=generator, device=DEVICE,
                               latent_space_dim=latent_space_dim,
                               anomaly_threshold=config['anomaly_threshold'],
                               res_weight=config['reconstruction_weight'],
                               max_iter_for_reconstruct=config['max_iter_for_reconstruction'],)

    total_samples = 0
    correct_predictions = 0
    for i, (x, y) in enumerate(test_dl):
        print(f'x shape: {x.shape}, y shape: {y.shape}')
        x = x.float().to(DEVICE)
        y = y.float().to(DEVICE)

        detect_res = detector.predict(x)
        print(f'Datect res: {detect_res.values}, y: {y.values}')

        # Convert predictions to binary labels
        pred_labels = (detect_res > anomaly_threshold).float().view_as(y)

        # Update counters
        total_samples += y.size(0) * y.size(1)
        correct_predictions += (pred_labels == y).sum().item()
        # print(f'Cor: {correct_predictions}, total: {total_samples}')

        if i % config['print_every'] == 0:
            # Calculate metrics
            accuracy = correct_predictions / total_samples
            precision, recall, f1 = calculate_metrics(pred_labels, y)
            if precision > 0 and recall > 0 and f1 > 0:
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


def _prepare_data(df: pd.DataFrame, config: dict) -> Iterator[torch.Tensor]:
    labels = df.pop('label')
    n_features = config['n_features']
    window_size = config['window_size']
    window_stride = config['window_stride']
    batch_size = config['batch_size']

    df = madgan.data.feature_extract_without_label(df, skip_size=0,
                                                   n_features=n_features)
    df.loc[:, 'label'] = labels
    print(f'Feature extracted data shape: {df.shape}')
    if config['test_batches']:
        print(f'Preparing data for testing batches')
        dataset = madgan.data.WindowDataset(df, window_size=window_size,
                                            window_slide=window_stride, use_label=True)
        dl = madgan.data.prepare_dataloader(dataset, batch_size=batch_size)
    else:
        print(f'Preparing data for testing')
        start, end = config['test_data_range']
        samples = df.iloc[start:end, :].copy()
        dataset = madgan.data.WindowDataset(samples, window_size=window_size,
                                            window_slide=window_stride, use_label=True)
        dl = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False)
    return dl

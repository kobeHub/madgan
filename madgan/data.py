from typing import Any, Callable, Iterator, Optional, Sequence
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class WindowDataset(Dataset):
    """Dataset to iterate using sliding windows over a pandas DataFrame.

    Args:
        df (pd.DataFrame): Sorted DataFrame by time.
        window_size (int): Number of elements per window.
        window_slide (int): Step size between each window.

    """

    def __init__(self, df: pd.DataFrame, window_size: int,
                 window_slide: int, use_label: bool = False) -> None:
        self.use_label = use_label
        data = df.values
        if use_label:
            data = df.drop(columns=["label"]).values
        self.windows = _window_array(data, window_size, window_slide)
        if use_label:
            self.labels = _window_array(
                df.label.values, window_size, window_slide)
            assert self.windows.shape[0] == self.labels.shape[0]
        print(f'Windows shape: {self.windows.shape}')

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.use_label:
            return (torch.as_tensor(self.windows[index].copy()),
                    torch.as_tensor(self.labels[index].copy()))
        else:
            return torch.as_tensor(self.windows[index].copy())

    def __len__(self) -> int:
        return self.windows.shape[0]


class LatentSpaceIterator(object):
    """Iterator that generates random sliding windows."""

    def __init__(self,
                 noise_shape: Sequence[int],
                 noise_type: str = "uniform") -> None:
        self.noise_fn: Callable[[Any], torch.Tensor]
        self.noise_shape = noise_shape
        if noise_type == "uniform":
            self.noise_fn = torch.rand
        elif noise_type == "normal":
            self.noise_fn = torch.randn
        else:
            raise ValueError(f"Unexpected noise type {noise_type}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        return self.noise_fn(*self.noise_shape)


def prepare_dataloader(ds: Dataset,
                       batch_size: int,
                       is_distributed: bool = False,
                       **kwargs: Any) -> DataLoader:
    """Creates a dataloader for training.

    Args:
        ds (Dataset): Training dataset.
        batch_size (int): DataLoader batch size.
        is_distributed (bool): Is the training distributed over multiple nodes?
            Defaults to False.

    Returns:
        DataLoader: Data iterator ready to use.
    """
    sampler: Optional[DistributedSampler] = (DistributedSampler(ds)
                                             if is_distributed else None)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, **kwargs)


def _window_array(array: np.ndarray, window_size: int,
                  window_slide: int) -> np.ndarray:
    return np.array([
        array[i:i + window_size]
        for i in range(0, array.shape[0] - window_size + 1, window_slide)
    ])


def feature_extract(df: pd.DataFrame, skip_size: int, n_features: int) -> Iterator[torch.utils.data.Dataset]:
    """Extract features from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        skip_size (int): Number of rows to skip at the beginning.
        n_features (int): Number of features to consider.

    Returns:
        Iterator[torch.utils.data.Dataset]: Iterator to go over the extracted features.
    """
    features = df.columns.tolist()
    df = df.astype(float)
    for f in features:
        max_val = df[f].max()
        if max_val != 0:
            df.loc[:, f] /= max_val
            # [-1, 1]
            df.loc[:, f] = 2 * df[f] - 1

    samples = df.iloc[skip_size:, :-1].copy()
    x_n = samples.values
    pca = PCA(n_components=n_features, svd_solver='full')
    pca.fit(x_n)
    pc = pca.components_
    t_n = np.matmul(x_n, pc.transpose(1, 0))
    samples = pd.DataFrame(t_n, columns=[f"pc-{i}" for i in range(n_features)])
    return samples

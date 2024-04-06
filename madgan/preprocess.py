import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def convert_csv(raw_path: str, output_csv: str, sheet_name: str = 'Combined Data') -> None:
    """Convert the raw data to a CSV file.

    Args:
        raw_path (str): Path to the raw data file.
        output_csv (str): Path to the output CSV file.
    """
    df = pd.read_excel(raw_path, sheet_name=sheet_name, header=1)
    assert df.isnull().sum().sum() == 0, 'Missing values in the data'
    df.columns = df.columns.str.strip()
    df['Normal/Attack'] = df['Normal/Attack'].replace('\s+', '', regex=True)
    df.loc[:, 'label'] = df['Normal/Attack'].map({'Normal': 0, 'Attack': 1})
    df = df.drop(['Normal/Attack', 'Timestamp'], axis=1)
    print(f"Data shape: {df.shape},\nColumns: {df.columns}")
    df.to_csv(output_csv, index=False)


def swat(raw_data_path: str,
         sheet_name: str = 'Combined Data',
         n_features: int = 10,
         output_csv: str = './data/swat_data.csv',
         skip_cnt: int = 21600) -> None:
    """Preprocess the raw data from the SWaT dataset and generate a CSV file.

    Args:
        raw_data (str): Path to the raw data file.
        n_features (int): Number of features to consider.
        output_csv (str): Path to the output CSV file.
        skip_cnt (int, optional): Number of rows to skip at the beginning 
            of the file to maintain system stability. Default to 21600.
    """
    df = pd.read_excel(raw_data_path, sheet_name=sheet_name, header=1)
    assert df.isnull().sum().sum() == 0, 'Missing values in the data'
    df.columns = df.columns.str.strip()
    f_columns = df.columns.tolist()
    f_columns.remove('Normal/Attack')
    f_columns.remove('Timestamp')
    print(f"All features columns ({len(f_columns)}): {f_columns}")

    df['Normal/Attack'] = df['Normal/Attack'].replace('\s+', '', regex=True)
    df.loc[:, 'label'] = df['Normal/Attack'].map({'Normal': 0, 'Attack': 1})
    df = df.drop(['Normal/Attack', 'Timestamp'], axis=1)

    m, n = df.shape
    assert df.shape[0] > skip_cnt, f"Number of records is less than {skip_cnt}"
    print(f'Read excel data: {df.shape},\nColumns: {df.columns}')
    print(f"Class counts: {df['label'].value_counts()}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[f_columns] = scaler.fit_transform(df[f_columns])
    # Skip the first 21600 records to maintain system stability
    samples = df.iloc[skip_cnt:, :-1].copy()
    labels = df.iloc[skip_cnt:, -1].copy()
    print(f"Data shape after skipping {skip_cnt} records: {samples.shape}")

    pca = PCA(n_components=n_features, svd_solver='full')
    pca.fit(samples)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_

    # Project the data to the new feature space
    projected_sample = samples @ pc.T
    print(f"Projected data shape: {projected_sample.shape}")
    print(f"Explained variance: {ex_var},\nHead: {projected_sample[:5]}")
    projected_sample.columns = [f'PC-{i+1}' for i in range(n_features)]
    projected_sample.loc[:, 'label'] = labels
    projected_sample.to_csv(output_csv, index=False)

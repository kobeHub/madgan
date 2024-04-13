from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from sklearn.metrics import mean_squared_error
from torch.optim import Adam
from torch import nn
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from tqdm import tqdm

import madgan


def extract_features(data, window_size, window_slide):
    # Initialize the features
    features = []

    # Loop over the data with a stride of 1
    for i in range(0, data.shape[0] - window_size + 1, window_slide):
        # Extract the window
        window = data[i:i+window_size]

        # Compute features
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        trend = (window[-1] - window[0]) / window_size
        # print(f"Shape: {mean.shape}, {std.shape}, {trend.shape}")

        # Append the features
        features.append([mean, std, trend])

    return np.array(features)


def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return precision, recall, f1


def display_label_counts(y_test):
    label_counts = np.bincount(y_test)
    assert len(label_counts) == 2, f'Unexpected label counts: {label_counts}'
    print(f'True label counts: 0->{label_counts[0]}, 1->{label_counts[1]}')


def run_anomalies_ocsvm(train_data, test_data, config):
    # Access the config parameter
    print(f'Config: {config}')
    # Extract features
    # train_features = extract_features(train_data, window_size, window_slide)
    # test_features = extract_features(test_data, window_size, window_slide)
    x_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
    x_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

    print(
        f'OCSVM train features shape: {x_train.shape}, {y_train.shape}')
    # Train the OCSVM model
    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(
        nu=0.5, kernel="rbf", gamma=0.1))
    clf.fit(x_train)
    print(f'OCSVM model trained')

    # Predict the anomalies in the test data
    pred_label = clf.predict(x_test)
    print(f'Predicted labels: {pred_label}')
    # The inner label is 1 -> 0;
    pred_label = np.where(pred_label == 1, 0, pred_label)
    # The outer label is -1 -> 1;
    pred_label = np.where(pred_label == -1, 1, pred_label)
    print(f'Predicted labels: {pred_label}, label: {y_test}')
    # Compute the value counts of pred_label
    pred_counts = np.bincount(pred_label)
    print(
        f'Value counts of pred_label: 0->{pred_counts[0]}, 1->{pred_counts[1]}')
    print(f'Predicted labels: {pred_label.shape}, test labels: {y_test.shape}')

    return compute_metrics(y_test, pred_label)


def run_anomalies_knn(train_data, test_data, config, n_neighbors=5, anomaly_threshold=100.5):
    x_train = train_data[config['knn_train_skip']:, :-1]
    test_start, test_end = config['knn_test_range']
    x_test = test_data[test_start:test_end, :-1]
    y_test = test_data[test_start:test_end, -1].astype(int)
    display_label_counts(y_test)

    # Train the KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(x_train)

    # Compute the average distance to the k nearest neighbors for each instance in the test set
    distances, _ = knn.kneighbors(x_test)
    avg_distances = distances.mean(axis=1)

    # If the average distance is greater than the anomaly threshold, label the instance as an anomaly (1)
    pred_label = np.where(avg_distances > anomaly_threshold, 1, 0)

    pred_counts = np.bincount(pred_label)
    print(f'Predicted labels: 0->{pred_counts[0]}, 1->{pred_counts[1]}')

    return compute_metrics(y_test, pred_label)


# ==============================================================================
# Auto-encoder based anomaly detection
# ==============================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*2),
            nn.BatchNorm1d(encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*2),
            nn.BatchNorm1d(encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_anomalies_autoencoder(train_data, test_data, config):

    x_train = train_data[config['ae_train_skip']:, :-1]
    test_start, test_end = config['ae_test_range']
    x_test = test_data[test_start:test_end, :-1]
    y_test = test_data[test_start:test_end, -1].astype(int)

    # Create a StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    display_label_counts(y_test)

    # Define the autoencoder model
    input_dim = x_train.shape[1]
    autoencoder = Autoencoder(input_dim, config['ae_encoding_dim'])

    # Compile and train the autoencoder
    criterion = nn.MSELoss()
    optimizer = Adam(autoencoder.parameters())
    x_train_tensor = torch.from_numpy(x_train).float()
    for epoch in tqdm(range(config['ae_epochs'])):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(x_train_tensor)
        loss = criterion(outputs, x_train_tensor)
        loss.backward()
        optimizer.step()

    # Use the autoencoder to reconstruct the test data
    x_test_tensor = torch.from_numpy(x_test).float()
    x_test_reconstructed = autoencoder(x_test_tensor).detach().numpy()

    # Compute the reconstruction error for each instance in the test set

    reconstruction_error = np.array([mean_squared_error(
        t, r) for t, r in tqdm(zip(x_test, x_test_reconstructed))])
    print(
        f'x_test_reconstructed: {x_test_reconstructed.shape}, error: {reconstruction_error.shape}')
    print(f'Reconstruction error: {reconstruction_error}')

    # If the reconstruction error is greater than the anomaly threshold, label the instance as an anomaly (1)
    pred_label = np.where(reconstruction_error >
                          config['ae_anomaly_threshold'], 1, 0)

    pred_counts = np.bincount(pred_label)
    print(f'Predicted labels: 0->{pred_counts[0]}, 1->{pred_counts[1]}')

    return compute_metrics(y_test, pred_label)


# ==============================================================================
# Feature Bagging
# ==============================================================================
def run_anomalies_feature_bagging(train_data, test_data, config):
    x_train = train_data[config['fb_train_skip']:, :-1]
    test_start, test_end = config['fb_test_range']
    x_test = test_data[test_start:test_end, :-1]
    y_test = test_data[test_start:test_end, -1].astype(int)

    # Create a StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    display_label_counts(y_test)

    # Define the base detector
    base_detector = LOF()

    # Define the feature bagging detector
    fb_detector = FeatureBagging(
        base_estimator=base_detector, n_estimators=config['fb_n_estimators'],
        n_jobs=config['fb_n_jobs'], contamination=config['fb_contamination'], verbose=True)

    # Fit the detector
    fb_detector.fit(x_train)
    print(f'Feature Bagging model trained')
    # Get the anomaly scores
    y_test_scores = fb_detector.decision_function(x_test)  # anomaly scores
    threshold = config['fb_threshold']
    print(f'Threshold: {threshold}\nAnomaly scores: {y_test_scores}')

    # Get the prediction on the test data
    y_test_pred = np.where(y_test_scores > threshold, 1, 0)

    pred_counts = np.bincount(y_test_pred)
    print(f'Predicted labels: 0->{pred_counts[0]}, 1->{pred_counts[1]}')

    return compute_metrics(y_test, y_test_pred)


def run_baseline(config_file: str = './config/baseline-config.yaml'):
    config = madgan.utils.read_config(config_file)
    print(f'Config for the baseline:\n{config}')
    train_data = pd.read_csv(config['train_data']).values
    test_data = pd.read_csv(config['test_data']).values
    print(
        f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    if config['enable_svm']:
        svm_p, svm_r, svm_f = run_anomalies_ocsvm(train_data=train_data,
                                                  test_data=test_data,
                                                  config=config)
        print(
            f'Metrics for OCSVM: Precision: {svm_p:.4f}, Recall: {svm_r:.4f}, F1 Score: {svm_f:.4f}')

    if config['enable_knn']:
        knn_p, knn_r, knn_f = run_anomalies_knn(train_data=train_data,
                                                test_data=test_data,
                                                config=config)
        print(
            f'Metrics for KNN: Precision: {knn_p:.4f}, Recall: {knn_r:.4f}, F1 Score: {knn_f:.4f}')

    if config['enable_ae']:
        ae_p, ae_r, ae_f = run_anomalies_autoencoder(train_data=train_data,
                                                     test_data=test_data,
                                                     config=config)
        print(
            f'Metrics for Autoencoder: Precision: {ae_p:.4f}, Recall: {ae_r:.4f}, F1 Score: {ae_f:.4f}')

    if config['enable_fb']:
        fb_p, fb_r, fb_f = run_anomalies_feature_bagging(train_data=train_data,
                                                         test_data=test_data,
                                                         config=config)
        print(
            f'Metrics for Feature Bagging: Precision: {fb_p:.4f}, Recall: {fb_r:.4f}, F1 Score: {fb_f:.4f}')

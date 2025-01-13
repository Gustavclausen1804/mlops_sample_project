import os
import sys

import pytest

from src.mlops_sample_project.data import corrupt_mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/test_target.pt"), reason="Data files not found")
def test_data():
    N_train = 30000
    N_test = 5000
    train_dataset, test_dataset = corrupt_mnist()
    assert len(train_dataset) == N_train, f"Expected {N_train} training samples, but got {len(train_dataset)}"
    assert len(test_dataset) == N_test, f"Expected {N_test} test samples, but got {len(test_dataset)}"
    
    for data, label in train_dataset:
        assert list(data.shape) in ([1, 28, 28], [784]), f"Unexpected data shape: {data.shape}"
    
    for data, label in test_dataset:
        assert list(data.shape) in ([1, 28, 28], [784]), f"Unexpected data shape: {data.shape}"
    
    train_labels = set(int(label.item()) for _, label in train_dataset)
    test_labels = set(int(label.item()) for _, label in test_dataset)
    
    assert train_labels == set(range(10)), "Not all labels are represented in the training set"
    assert test_labels == set(range(10)), "Not all labels are represented in the test set"
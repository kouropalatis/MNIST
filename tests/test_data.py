import torch
import os
import pytest
from mnist.data import corrupt_mnist
from tests import _PATH_DATA

# We define the specific file we are looking for to verify data presence
DATA_FILE = os.path.join(_PATH_DATA, "train_0.npz")

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data folder not found")
@pytest.mark.skipif(not os.path.exists(DATA_FILE), reason="Specific data files (train_0.npz) not found")
def test_data():
    """Check that data gets correctly loaded with correct shapes and all labels."""
    train, test = corrupt_mnist()
    
    # Check dataset sizes
    n_train = len(train)
    n_test = len(test)
    assert n_train in [30000, 50000], f"Training set expected 30k or 50k samples, but got {n_train}"
    assert n_test == 5000, f"Test set expected 5000 samples, but got {n_test}"
    
    # Check shapes and labels
    for dataset_name, dataset in [("train", train), ("test", test)]:
        for i, (x, y) in enumerate(dataset):
            assert x.shape == (1, 28, 28), f"Sample {i} in {dataset_name} has shape {x.shape}, expected (1, 28, 28)"
            assert 0 <= y <= 9, f"Sample {i} in {dataset_name} has invalid label {y}"
            if i >= 100:
                break
    
    # Verify all classes are represented
    # Using list comprehension to get labels if it's a TensorDataset
    train_labels = torch.tensor([y for _, y in train])
    test_labels = torch.tensor([y for _, y in test])
    
    assert len(torch.unique(train_labels)) == 10, f"Train set missing digits. Found: {torch.unique(train_labels)}"
    assert len(torch.unique(test_labels)) == 10, f"Test set missing digits. Found: {torch.unique(test_labels)}"
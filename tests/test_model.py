import pytest
import torch

from mnist.model import MyAwesomeModel


# Test that the model works for common training batch sizes
@pytest.mark.parametrize("batch_size", [1, 32, 64, 128])
def test_model_output_shape(batch_size: int) -> None:
    model = MyAwesomeModel()
    model.eval()

    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch_size, 10), f"Output shape mismatch for batch_size {batch_size}"


# Test that our custom ValueError catches multiple types of wrong input
@pytest.mark.parametrize(
    "invalid_shape",
    [
        (1, 28, 28),  # Missing batch dimension (3D)
        (1, 1, 28),  # Missing width (3D)
        (1, 3, 28, 28),  # Wrong channels (RGB instead of Grayscale)
        (1, 1, 32, 32),  # Wrong resolution
    ],
)
def test_model_raises_error_on_invalid_shape(invalid_shape):
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected"):
        # Create a tensor with the invalid shape
        x = torch.randn(*invalid_shape)
        model(x)

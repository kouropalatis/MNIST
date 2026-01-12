import os
import time
import torch
import wandb
import pytest
# Replace 'mnist.model' and 'MyAwesomeModel' with your actual package/class names
from mnist.model import MyAwesomeModel 

def load_model(artifact_path: str):
    """Loads a model from a W&B artifact path."""
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    
    # Download the artifact to a local directory
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download(root="downloaded_model_checkpoints")
    
    # Identify the checkpoint file (assumes .pt or .pth)
    model_checkpoint = os.path.join(artifact_dir, "model.pth")
    
    # Initialize and load weights
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@pytest.mark.skipif(not os.getenv("MODEL_NAME"), reason="MODEL_NAME env var not set")
def test_model_speed():
    """Performance test: 100 predictions must finish under 1 second."""
    model_path = os.getenv("MODEL_NAME")
    if not model_path:
        pytest.fail("MODEL_NAME environment variable is required but not set.")

    model = load_model(model_path)
    
    # Create random input tensor (MNIST shape: [batch, channel, height, width])
    dummy_input = torch.rand(1, 1, 28, 28)
    
    # Warm-up (standard practice for benchmarking)
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)

    # Performance measurement using high-resolution counter
    start_time = time.perf_counter()
    with torch.inference_mode(): # Faster inference by disabling gradient tracking
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nTime for 100 predictions: {elapsed_time:.4f}s")
    
    # Assert that the total time is less than 1.0 second
    assert elapsed_time < 1.0, f"Model is too slow! Took {elapsed_time:.4f}s for 100 predictions."
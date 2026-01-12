import torch
from mnist.model import MyAwesomeModel

def test_training_step():
    """Assert that the loss decreases after one step of optimization."""
    model = MyAwesomeModel()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create one dummy batch
    data = torch.randn(16, 1, 28, 28)
    target = torch.randint(0, 10, (16,))
    
    # Step 1
    optimizer.zero_grad()
    output1 = model(data)
    loss1 = criterion(output1, target)
    loss1.backward()
    optimizer.step()
    
    # Step 2
    output2 = model(data)
    loss2 = criterion(output2, target)
    
    assert loss2 < loss1, f"Optimization failed: Loss did not decrease. Loss 1: {loss1.item():.4f}, Loss 2: {loss2.item():.4f}"

def test_overfit_batch():
    """Sanity check: Can the model reach high accuracy on a tiny 4-image dataset?"""
    model = MyAwesomeModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    
    data = torch.randn(4, 1, 28, 28)
    target = torch.randint(0, 10, (4,))
    
    # Train for 20 mini-iterations
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    
    # Evaluate on the same 4 images
    model.eval()
    with torch.no_grad():
        preds = model(data).argmax(dim=1)
        accuracy = (preds == target).float().mean()
    
    assert accuracy > 0.8, f"Sanity check failed: Model could not overfit 4 images. Accuracy: {accuracy:.2f}"
import torch
import matplotlib.pyplot as plt

# 1. Load the file
data = torch.load('C:\Users\lykos\DTU\MLOps\test\data\corruptmnist\test_images.pt')

# 2. Check the shape
# Common MNIST shapes are (10, 28, 28) or (10, 1, 28, 28)
print(f"Data shape: {data.shape}")

# 3. Plot the images
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i in range(10):
    # If the shape is (1, 28, 28), we use .squeeze() to make it (28, 28)
    img = data[i].squeeze() 
    
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Img {i}")

plt.show()
from data_loader2 import DenoMAEDataGenerator
from torch.utils.data import DataLoader

image_path = './data/noisyImg/'
signal_path = './data/signal/'
noise_path = './data/noise/'

batch_size = 16
image_size = (32, 32)

# Create dataset and data loader
dataset = DenoMAEDataGenerator(image_path=image_path, signal_path=signal_path,
                            noise_path = noise_path, image_size=image_size)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the data loader
for batch_idx, (images, signals, noises) in enumerate(data_loader):
    # images: Tensor of shape (batch_size, 3, image_size[0], image_size[1])
    # npy_data: Tensor of shape (batch_size, ...)
    print(f"Batch {batch_idx+1}")
    print("Images shape:", images.shape)
    print("Signals data shape:", signals.shape)
    print("Noises data shape:", noises.shape)
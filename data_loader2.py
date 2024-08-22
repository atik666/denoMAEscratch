import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class DenoMAEDataGenerator(Dataset):
    def __init__(self, image_path, signal_path, noise_path, image_size=(224, 224), transform=None):
        """
        Initializes the data generator for DenoMAE.
        
        Args:
            image_path (str): Directory containing image files (e.g., PNG).
            signal_path (str): Directory containing signal files.
            noise_path (str): Directory containing noise files.
            image_size (tuple): Size to which images will be resized.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_path = image_path
        self.signal_path = signal_path
        self.noise_path = noise_path
        self.image_size = image_size
        self.transform = transform

        self.image_filenames = sorted(os.listdir(image_path))
        self.signal_filenames = sorted(os.listdir(signal_path))
        self.noise_filenames = sorted(os.listdir(noise_path))

    def __len__(self):
        return len(self.image_filenames)
    
    def preprocess_npy(self, npy_path, target_length):
        """
        Loads and preprocesses an NPY file.
        
        Args:
            npy_path (str): Path to the NPY file.
            target_length (int): The length to which the data should be resized.
        
        Returns:
            torch.Tensor: The processed NPY data as a PyTorch tensor.
        """
        npy_data = np.load(npy_path)
        if len(npy_data) > target_length:
            npy_data = np.interp(np.linspace(0, len(npy_data) - 1, target_length), 
                                    np.arange(len(npy_data)), npy_data)
        return torch.tensor(npy_data, dtype=torch.float32)

    def __getitem__(self, index):
        # Load and preprocess image
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).resize(self.image_size)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Load and preprocess npy file
        signal_name = self.signal_filenames[index]
        signal_path = os.path.join(self.signal_path, signal_name)
        signal_data = self.preprocess_npy(signal_path, target_length=1024)

        noise_name = self.noise_filenames[index]
        noise_path = os.path.join(self.noise_path, noise_name)
        noise_data = self.preprocess_npy(noise_path, target_length=1024)

        return img, signal_data, noise_data
    
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
import os
import random
import pandas as pd
import numpy as np

from image_loader import CustomImageDataset
from signal_loader import CustomSignalDataset
from noise_loader import CustomNoiseDataset

from torch.utils.data import DataLoader
import torchvision.transforms as T
from itertools import islice
from PIL import Image

import torch
from einops import repeat, rearrange

image_path = './data/noisyImg/'
signal_path = './data/signal/'
noise_path = './data/noise/'
batch_size = 1
image_size = 32

config = {
    'image_path':image_path,
    'signal_path':signal_path,
    'noise_path':noise_path,
    'batch_size': batch_size,
    'image_size': image_size,
}

# img_list = os.walk(config['image_path'])
# print(img_list)
# noise_list = os.walk(config['noise_path'])
# signal_path = os.walk(config['signal_path'])

# transform = T.Compose(T.ToTensor())

image = CustomImageDataset(config['image_path'])
noise = CustomSignalDataset(config['noise_path'])
signal = CustomSignalDataset(config['signal_path'])


data = {'image':image, 'signal':signal, 'noise':noise}

image_dataloader = DataLoader(data['image'], batch_size=config['batch_size'], shuffle=True)
print("image loader: ", image_dataloader)

def preprocess(image, signal, noise):

    # resized_image = image.resize(config['image_size'])
    # image_array = np.array(resized_image)
    normalized_image_array = image / 255.0

    # def norm_array(arr):
    #     return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    # def norm_tensor(signal):
    #     return (signal - torch.min(signal)) / (torch.max(signal) - torch.min(signal))
    
    def norm_complex_tensor(tensor):
        # Separate real and imaginary parts
        real_part = tensor.real
        imag_part = tensor.imag

        # Normalize real part
        real_min = torch.min(real_part)
        real_max = torch.max(real_part)
        real_norm = (real_part - real_min) / (real_max - real_min)

        # Normalize imaginary part
        imag_min = torch.min(imag_part)
        imag_max = torch.max(imag_part)
        imag_norm = (imag_part - imag_min) / (imag_max - imag_min)

        # Reconstruct the complex tensor
        normalized_tensor = torch.complex(real_norm, imag_norm)
        
        return normalized_tensor

    return normalized_image_array, norm_complex_tensor(signal), norm_complex_tensor(noise)

def generate_random_image_masks(images, mask_ratio=0.75):
    """
    Generate random masks for a batch of input images.

    Args:
        images (torch.Tensor): Batch of images with shape [batch_size, height, width, channels].
        mask_ratio (float): Ratio of the image pixels to be masked.

    Returns:
        torch.Tensor: Random masks with the same shape as the input images, including the channel dimension.
    """

    batch_size, H, W, C = images.size()  # Get dimensions from the input batch
    num_pixels = H * W
    num_masked = int(num_pixels * mask_ratio)

    masks = torch.ones((batch_size, H, W, C), dtype=torch.bool)

    for i in range(batch_size):
        for c in range(C):
            masked_indices = random.sample(range(num_pixels), num_masked)
            for idx in masked_indices:
                y = idx // W
                x = idx % W
                masks[i, y, x, c] = False
    
    return masks

def apply_masks(image, masks):
    """
    Apply masks to the input data.

    Args:
        input_data (torch.Tensor): The input data to be masked.
        masks (torch.Tensor): The masks to apply.

    Returns:
        torch.Tensor: The masked input data.
    """

    masked_data = image * (1 - masks.float())
    
    return masked_data

def random_indexes(index_size: int):
    forward_indexes = np.arrange(index_size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T*(1-self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes
    
def random_masking(x, mask_ratio): # got from medium https://medium.com/thedeephub/building-mae-vision-transformer-from-scratch-using-pytorch-masked-autoencoders-are-scalable-2c2e78e0be02
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        B, T, D = x.shape  
        len_keep = int(T * (1 - mask_ratio))
        
        # creating noise of shape (B, T) to latter generate random indices
        noise = torch.rand(B, T, device=x.device)  
        
        # sorting the noise, and then ids_shuffle to keep the original indexe format
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # gathering the first few samples
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, T], device=x.device)
        mask[:, :len_keep] = 0 

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask, ids_restore

class MultiModalDataGenerator:
    def __init__(self,data, config, ):
        self.data = data
        self.config = config
        # generator process
        # load the signal
        # generate the noise
        # merge noise and signal
        # generate images of noise and signal and noisy signal
        # return (noisy_image, noise, clean_image)

        # self.normalized_image_array = self.preprocess()

        self.image_dataloader = DataLoader(self.data['image'], batch_size=config['batch_size'], shuffle=True)
        self.signal_dataloader = DataLoader(self.data['signal'], batch_size=config['batch_size'], shuffle=True)
        self.noise_dataloader = DataLoader(self.data['noise'], batch_size=config['batch_size'], shuffle=True)

    def __getitem__(self, index):

        image_batch = next(iter(self.image_dataloader))
        signal_batch = next(iter(self.signal_dataloader))
        noise_batch = next(iter(self.noise_dataloader))
    
        # image_batch = next(islice(self.image_dataloader, index, index+1))
        # signal_batch = next(islice(self.signal_dataloader, index, index+1))
        # noise_batch = next(islice(self.noise_dataloader, index, index+1))

        image, signal, noise = preprocess(image_batch, signal_batch, noise_batch)
        masks = generate_random_image_masks(image, mask_ratio=0.75)
        print("masks: ", masks.shape)
        masked_image = apply_masks(image, masks)

        print("masked_image: ", masks.shape)
        
        return masked_image


# masked_image = MultiModalDataGenerator(data,config)[1]

# tensor_np = masked_image.squeeze(0).numpy()
# tensor_np = (tensor_np * 255).astype(np.uint8)
# image = Image.fromarray(tensor_np)
# image = image.resize((224, 224), Image.BILINEAR)
# image.save("masked_image.jpg", dpi=(600,600))
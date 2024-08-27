from data_loader2 import DenoMAEDataGenerator
from torch.utils.data import DataLoader

config = {'train_image_path' : './data/train/noisyImg/',
            'train_signal_path' : './data/train/signal/',
            'train_noise_path' : './data/train/noise/',
            'test_image_path' : './data/test/noisyImg/',
            'test_signal_path' : './data/test/signal/',
            'test_noise_path' : './data/test/noise/',
            'batch_size' : 16,
            'image_size' : (32, 32),
            }

# Create dataset and data loader
train_dataset = DenoMAEDataGenerator(image_path=config['train_image_path'], signal_path=config['train_signal_path'],
                            noise_path = config['train_noise_path'], image_size=config['image_size'])

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

test_dataset = DenoMAEDataGenerator(image_path=config['test_image_path'], signal_path=config['test_signal_path'],
                            noise_path = config['test_noise_path'], image_size=config['image_size'])

test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

# # Iterate through the data loader
# for batch_idx, (images, signals, noises) in enumerate(train_dataloader):
#     # images: Tensor of shape (batch_size, 3, image_size[0], image_size[1])
#     # npy_data: Tensor of shape (batch_size, ...)
#     print(f"Batch {batch_idx+1}")
#     print("Images shape:", images.shape)
#     print("Signals data shape:", signals.shape)
#     print("Noises data shape:", noises.shape)

# for batch_idx, (images, signals, noises) in enumerate(test_dataloader):
#     # images: Tensor of shape (batch_size, 3, image_size[0], image_size[1])
#     # npy_data: Tensor of shape (batch_size, ...)
#     print(f"Batch {batch_idx+1}")
#     print("Test Images shape:", images.shape)
#     print("Test Signals data shape:", signals.shape)
#     print("Test Noises data shape:", noises.shape)

for batch_idx, data in enumerate(test_dataloader):
    # images: Tensor of shape (batch_size, 3, image_size[0], image_size[1])
    # npy_data: Tensor of shape (batch_size, ...)
    print(f"Batch {batch_idx+1}")
    print("Concatenated data shape:", data.shape)
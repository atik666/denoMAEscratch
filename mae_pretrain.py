from model import *
import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from data_loader2 import DenoMAEDataGenerator
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=200) # default 2000
    parser.add_argument('--warmup_epoch', type=int, default=5) # default 200
    parser.add_argument('--model_path', type=str, default='./models/vit-t-mae.pt')

    args = parser.parse_args()

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    config = {'train_image_path' : './data/train/noisyImg/',
              'train_noiseless_image_path' : './data/train/noiseLessImg/',
            'train_signal_path' : './data/train/signal/',
            'train_noise_path' : './data/train/noise/',
            'test_image_path' : './data/test/noisyImg/',
            'test_signal_path' : './data/test/signal/',
            'test_noise_path' : './data/test/noise/',
            'batch_size' : 16,
            'image_size' : (32, 32),
            }

    # Create dataset and data loader
    train_dataset = DenoMAEDataGenerator(image_path=config['train_image_path'], noiseLessImg_path=config['train_noiseless_image_path'], signal_path=config['train_signal_path'],
                                noise_path = config['train_noise_path'], image_size=config['image_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    test_dataset = DenoMAEDataGenerator(image_path=config['test_image_path'],noiseLessImg_path=config['train_noiseless_image_path'], signal_path=config['test_signal_path'],
                                noise_path = config['test_noise_path'], image_size=config['image_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)   

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        # for img, _, _ in tqdm(iter(train_dataloader)):
        for modalities, noiseless_img in tqdm(iter(train_dataloader)):
            step_count += 1
            modalities = modalities.to(device) # TODO: img should be noisy image
            predicted_img, mask = model(modalities)
            loss = torch.mean((predicted_img - noiseless_img.to(device)) ** 2 * mask) / args.mask_ratio # TODO: img should be noiseless image
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss:.4f}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            for val_img, _, _ in tqdm(iter(test_dataloader)):
                # val_img, _, _ = torch.stack([test_dataloader[i][0] for i in range(16)])
                val_img = val_img.to(device)
                predicted_val_img, mask = model(val_img)
                predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, args.model_path)
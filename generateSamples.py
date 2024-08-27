import os
import numpy as np
from funcSampleGeneration import generate_constellation_images
import shutil

# Modify parameters below:
samples_per_image = 1024    # Number of samples to produce each constellation images
image_size = (32, 32)
image_num = [2]
cons_scale = [2.5, 2.5]    # scale of constellation image 
mod_type = ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK']
set_type = ['noiseLessImg', 'noisyImg', 'signal', 'noise']
mode = 'train'    # 'train' or 'test'

fold_path = f'./data/{mode}'     # where to store images

# Main script
if __name__ == "__main__":
    # Clean old images if the directory exists
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    
    os.makedirs(fold_path, exist_ok=True)

    for gen_type in set_type:
        os.makedirs(os.path.join(fold_path, gen_type), exist_ok=True)

    for mod in mod_type:
        # generate_constellation_images(mod, snr, samples_per_image, image_format, image_num[0], image_size, cons_scale, gen_type, fd_txt, set_path, label)
        generate_constellation_images(mod, samples_per_image, image_num[0], image_size, cons_scale, set_type, fold_path)
    
    print("Processing complete.")


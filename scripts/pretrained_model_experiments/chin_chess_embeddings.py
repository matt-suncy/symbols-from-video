"""
Script for experimenting with embeddings from variants of CLIP + GAN models of 
XiangQi stock footage video. 
"""

import src.FuseDream as FuseDream
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image

''' 
First state transition at frame 41/42 to frame 79/80

Second state transition at frame 178/179 to frame 257/258
'''

if __name__ == "__main__":
    ### Generation: Click the 'run' button and the final generated image will be shown after the end of the algorithm
    utils.seed_rng(SEED) 

    sentence = SENTENCE

    print('Generating:', sentence)
    if MODEL == "biggan-256":
        G, config = get_G(256) 
    elif MODEL == "biggan-512":
        G, config = get_G(512) 
    else:
        raise Exception('Model not supported')
    generator = FuseDreamBaseGenerator(G, config, 10) 
    z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=INIT_ITERS, num_basis=NUM_BASIS)

    z_cllt_save = torch.cat(z_cllt).cpu().numpy()
    y_cllt_save = torch.cat(y_cllt).cpu().numpy()
    img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=False, augment=True, opt_iters=OPT_ITERS, optimize_y=True)
    ### Set latent_noise = True yields slightly higher AugCLIP score, but slightly lower image quality. We set it to False for dogs.
    score = generator.measureAugCLIP(z, y, sentence, augment=True, num_samples=20)
    print('AugCLIP score:', score)
    import os
    if not os.path.exists('./samples'):
        os.mkdir('./samples')
    save_image(img, 'samples/fusedream_%s_seed_%d_score_%.4f.png'%(sentence, SEED, score))

    from IPython import display
    display.display(display.Image('samples/fusedream_%s_seed_%d_score_%.4f.png'%(sentence, SEED, score)))
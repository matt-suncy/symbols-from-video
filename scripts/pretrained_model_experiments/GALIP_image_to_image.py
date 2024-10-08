'''
Pipeline for image to image with GALIP
Video frame --> CLIP embedding --> Image generator 
'''

# Imports
### 
import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils

from src.GALIP.code.lib.utils import load_model_weights,mkdir_p
from src.GALIP.code.models.GALIP import NetG, CLIP_TXT_ENCODER
### 
# End

if __name__ == "__main__":

    # Device declaration
    device = 'cuda:0' # 'cpu' # 'cuda:0'

    # CLIP settings
    CLIP_text = "ViT-B/32"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.eval()

    # Model settings 
    text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
    netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
    # path = '../../src/GALIP/code/saved_models/pretrained/pre_cc12m.pth'
    # TODO: This path must change, can't be so machine depend
    path = '/home/jovyan/Documents/latplan-temporal-segmentation/src/GALIP/saved_models/pretrained/pre_cc12m.pth'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

    # Generation hyperparams
    batch_size = 8 # This is literally the number of images it's gonna generate
    noise = torch.randn((batch_size, 100)).to(device) # Don't really understand why there needs to be noise

    # Text prompt
    captions = ['An overhead shot of a chessboard']
    
    # CHANGE current directory for 'samples' folder
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_file_path)
    # Output directory
    if not os.path.exists('./samples'):
        os.mkdir('./samples')

    # Image generation pipeline
    with torch.no_grad():
        for i in range(len(captions)):
            caption = captions[i]
            tokenized_text = clip.tokenize([caption]).to(device)
            sent_emb, word_emb = text_encoder(tokenized_text)
            sent_emb = sent_emb.repeat(batch_size,1)
            fake_imgs = netG(noise,sent_emb,eval=True).float()
            name = f'{captions[i].replace(" ", "-")}'
            vutils.save_image(fake_imgs.data, './samples/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)
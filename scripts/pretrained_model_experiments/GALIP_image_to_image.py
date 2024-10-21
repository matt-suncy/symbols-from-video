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
import numpy as np

from src.GALIP.code.lib.utils import load_model_weights,mkdir_p
from src.GALIP.code.models.GALIP import NetG, CLIP_TXT_ENCODER
### 

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

    # Generation hyperparameters
    batch_size = 4 # This is literally the number of images it's gonna generate
    noise = torch.randn((batch_size, 100)).to(device) # Don't really understand why there needs to be noise

    # Text prompt
    captions = ['A kid playing with blocks.']
    
    # CHANGE current directory for 'samples' folder
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_file_path)
    # Output directory
    if not os.path.exists('./samples'):
        os.mkdir('./samples')

    # Load and preprocess the two images, NOTE replace as necessary
    image1_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames/kid_playing_with_blocks_1.mp4/0000000000.jpg'  
    image2_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames/kid_playing_with_blocks_1.mp4/0000000147.jpg' 

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Preprocess images using CLIP's preprocess function
    img1_processed = preprocess(img1).unsqueeze(0).to(device)
    img2_processed = preprocess(img2).unsqueeze(0).to(device)

    # Extract CLIP embeddings for both images
    with torch.no_grad():
        img1_embedding = clip_model.encode_image(img1_processed)
        img2_embedding = clip_model.encode_image(img2_processed)

    # Number of interpolation steps
    num_steps = 10  # Adjust as needed for smoother interpolation

    # Interpolate between the two embeddings
    interpolated_embeddings = []
    for t in np.linspace(0, 1, num_steps):
        # Linear interpolation between embeddings
        embedding_t = (1 - t) * img1_embedding + t * img2_embedding
        # Normalize the interpolated embedding
        embedding_t = embedding_t / embedding_t.norm(dim=-1, keepdim=True)
        interpolated_embeddings.append(embedding_t)

    # Image generation pipeline
    with torch.no_grad():
        for idx, embedding_t in enumerate(interpolated_embeddings):
            # Repeat the embedding to match the batch size
            sent_emb = embedding_t.repeat(batch_size, 1)
            # Generate images using netG
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            # Save the generated images
            name = f'interpolated_{idx}'
            vutils.save_image(
                fake_imgs.data,
                f'./samples/{name}.png',
                nrow=8,
                value_range=(-1, 1),
                normalize=True
            )

    # Image generation pipeline
    # TODO: Modify below to use CLIP to embed images instead of text
    '''
    with torch.no_grad():
        for i in range(len(captions)):
            caption = captions[i]
            tokenized_text = clip.tokenize([caption]).to(device)
            sent_emb, word_emb = text_encoder(tokenized_text)
            sent_emb = sent_emb.repeat(batch_size,1)
            fake_imgs = netG(noise,sent_emb,eval=True).float()
            name = f'{captions[i].replace(" ", "-")}'
            vutils.save_image(fake_imgs.data, './samples/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)
    '''
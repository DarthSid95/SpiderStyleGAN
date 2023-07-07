# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import tensorflow as tf

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--outdirMid', help='Where to save the Mid images', type=str, required=True, metavar='DIR2')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """


    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    impath = outdir+'/Images'
    midpath = outdir+'/MidImages'

    os.makedirs(impath, exist_ok=True)
    os.makedirs(midpath, exist_ok=True)



    ####### ####### ####### 
    ####### SpiderStyleGAN3
    ####### ####### ####### 
    # Lod the input datset from existing pickle.
    #####
    ##### FFHQ Dogs and TIN are old style, 9.6k and 800
    ##### Uki and MetFaces are New Style, 9.6k and 25k. 
    ##### FFHQ SG3 is TIN 9.6k, old.
    ##### AFHQ Cats is old, and probably 9.6k or 800. From NeurIPS Era
    #####
    transform_flag = 'old'
    input_pkl = 'models/network-snapshot-000800.pkl' ### Dogs32(From DGX) Style3-T-- DGX; Style2 -- SM3
    # input_pkl = 'models/network-snapshot-001209.pkl' ### TIN32(From Sirius) C10 -- Altair
    # input_pkl = 'models/network-snapshot-009676.pkl' ### TIN32(From Sirius) Style3-T -- FS; TIN32+FFwts(From Altair) Style2

    # input_pkl = 'models/Dog32-network-snapshot-025000.pkl'
    # input_pkl = 'models/TIN32-network-snapshot-025000.pkl'
    # transform_flag = 'new'

    # input_pkl = 'models/DiffAugment-stylegan2-cifar10.pkl'
    # input_pkl = 'models/network-final.pkl'
    # if (input_pkl is not None) and (rank == 0):
    print(f'Loading Input model from "{input_pkl}"')
    with dnnlib.util.open_url(input_pkl) as f:
        # InputG, InputD, InputGema = legacy.load_network_pkl(f)
        InputG = legacy.load_network_pkl(f)['G_ema'].to(device)
    # with open(input_pkl, 'rb') as file:
    #     G1, D1, InputG = pickle.load(file)

    print("InputG Load")


    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    step = 50
    # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    for i in range(0,len(seeds),step):
        print('Generating image for seed %d (%d/%d) ...' % (seeds[i], seeds[i], len(seeds)))
        

        ### For Truncation trick Experiments
        # z = np.random.RandomState(seeds[i]).randn(1, G.z_dim)

        with tf.device('/CPU'):
            z = tf.random.truncated_normal((1,G.z_dim),mean=0.0,stddev=1.0,dtype=tf.dtypes.float32,seed=seeds[i]).numpy()



        for k in range(1,step):
            print('Generating image for seed %d (%d/%d) ...' % (seeds[i+k], seeds[i+k], len(seeds)))
            with tf.device('/CPU'):
                z_next = tf.random.truncated_normal((1,G.z_dim),mean=0.0,stddev=1.0,dtype=tf.dtypes.float32,seed=seeds[i+k]).numpy()
            # z_next = np.random.RandomState(seeds[i+k]).randn(1, G.z_dim)
            z = np.concatenate((z, z_next), axis = 0)
        
        z = torch.from_numpy(z).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        ####### ####### ####### 
        ####### SpiderStyleGAN3
        ####### ####### #######

        z_img_org = z_img = InputG(z=z, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode).to(device)

        if transform_flag == 'old':
            import torchvision.transforms as T
            transform = T.Resize((16,32))
            # print('grid_z,grid_c',grid_z.size(),grid_c.size()) ##[Num,512], [Num,0] each
            # print('z_img',z_img.size()) ##[Num,3,32,32]
            z_img = torch.mean(input = z_img, dim = 1, keepdim = True)
            # print('z_img',z_img.size()) ##[Num,1,32,32]
            z_img = transform(z_img)
            # print('z_img',z_img.size()) ##[Num,1,16,32]
            z_img = z_img.permute(0, 2, 3, 1)
            # print('z_img',z_img.size()) ##[Num,16,32,1]
            z_img = torch.reshape(z_img, (z_img.size()[0],16*32,1))
            # print('z_img',z_img.size()) ##[Num,512,1]
            z_img = torch.squeeze(z_img, dim=2).to(device)
            # print('z_img',z_img.size(),c.size()) ##[Num,512], [Num,0] each
            # exit(0)
        else:
            import torchvision.transforms as T
            transform2 = T.Resize((23,23))
            z_img = torch.mean(input = z_img, dim = 1, keepdim = True)
            # print('z_img',z_img.size()) ##[4,1,32,32]
            z_img = transform2(z_img)
            # print('z_img',z_img.size()) ##[4,1,16,32]
            z_img = z_img.permute(0, 2, 3, 1)
            # print('z_img',z_img.size()) ##[4,16,32,1]
            z_img = torch.reshape(z_img, (z_img.size()[0],23*23,1))[:,0:512,:]
            # print('z_img',z_img.size()) ##[4,512,1]
            z_img = torch.squeeze(z_img)

        img = G(z_img, label, truncation_psi=truncation_psi, noise_mode=noise_mode)


        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        z_img_org = (z_img_org.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        for j in range(50):
            PIL.Image.fromarray(z_img_org[j].cpu().numpy(), 'RGB').save(f'{midpath}/seed{seeds[i+j]:04d}.png')
            PIL.Image.fromarray(img[j].cpu().numpy(), 'RGB').save(f'{impath}/seed{seeds[i+j]:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

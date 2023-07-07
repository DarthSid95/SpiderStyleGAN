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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from scipy.interpolate import interp1d
from torchvision.utils import make_grid
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

    

    impath = outdir+'/Interpol_Images_Supplementary'
    midpath = outdir+'/Interpol_MidImages_Supplementary'

    os.makedirs(impath, exist_ok=True)
    os.makedirs(midpath, exist_ok=True)

    ####### ####### ####### 
    ####### SpiderStyleGAN3
    ####### ####### ####### 
    # Lod the input datset from existing pickle.
    #####
    ##### FFHQ Dogs and TIN are old style, 9.6k and 800
    ##### Uki and MetFaces are Net Style, 9.6k and 25k. 
    ##### FFHQ SG3 is TIN 9.6k, old.
    ##### AFHQ Cats is old, and probably 9.6k or 800. From NeurIPS Era
    #####
    # transform_flag = 'old'
    # input_pkl = 'models/network-snapshot-000800.pkl' ### Dogs32(From DGX) Style3-T-- DGX; Style2 -- SM3
    # input_pkl = 'models/network-snapshot-001209.pkl' ### TIN32(From Sirius) C10 -- Altair
    # input_pkl = 'models/network-snapshot-009676.pkl' ### TIN32(From Sirius) Style3-T -- FS; TIN32+FFwts(From Altair) Style2

    transform_flag = 'new'
    input_pkl = 'models/Dog32-network-snapshot-025000.pkl'
    # input_pkl = 'models/TIN32-network-snapshot-025000.pkl'



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
    num_interps = 7
    label = torch.zeros([(num_interps+1), G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    print(seeds)
    import torchvision.transforms as T
    transform = T.Resize((16,32))
    transform2 = T.Resize((23,23))

    def img_transform(img,transform,transform_flag):

        if transform_flag == 'old':
            img = torch.mean(input = img, dim = 1, keepdim = True)
            img = transform(img)
            img = img.permute(0, 2, 3, 1)
            img = torch.reshape(img, (img.size()[0],16*32,1))
            img = torch.squeeze(img, dim=2).to(device)
        else:            
            img = torch.mean(input = img, dim = 1, keepdim = True)
            img = transform2(img)
            img = img.permute(0, 2, 3, 1)
            img = torch.reshape(img, (img.size()[0],23*23,1))[:,0:512,:]
            img = torch.squeeze(img)

        return img


    # torch.from_numpy


    for i in range(len(seeds)-3):
        tl_seed = seeds[i]
        tr_seed = seeds[i+1]
        bl_seed = seeds[i+2]
        br_seed = seeds[i+3]

        print('Interpolating on  MidImage for seeds grid [%d-%d ; %d-%d]) ...' % (tl_seed, tr_seed, bl_seed, br_seed))
        z_tl = torch.from_numpy(np.random.RandomState(tl_seed).randn(1, G.z_dim)).to(device)
        z_tr = torch.from_numpy(np.random.RandomState(tr_seed).randn(1, G.z_dim)).to(device)
        z_bl = torch.from_numpy(np.random.RandomState(bl_seed).randn(1, G.z_dim)).to(device)
        z_br = torch.from_numpy(np.random.RandomState(br_seed).randn(1, G.z_dim)).to(device)

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
        z_tl_img = InputG(z=z_tl, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        z_tr_img = InputG(z=z_tr, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        z_bl_img = InputG(z=z_bl, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        z_br_img = InputG(z=z_br, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        ##### z_tl ----------- z_tr
        #####  |-----------------|
        #####  |-----------------|
        #####  |-----------------|
        ##### z_bl ----------- z_br

        #### Col 1: 
        stack = np.vstack([z_tl_img.cpu().numpy(), z_bl_img.cpu().numpy()]) 
        linfit = interp1d([1,num_interps+1], stack, axis=0)
        col1_interp_latentsPart= linfit(list(range(1,num_interps+1)))

        col1_interp_latents = np.vstack([ col1_interp_latentsPart, z_bl_img.cpu().numpy()])

        print(col1_interp_latents.shape)

        #### Col n: 
        stack = np.vstack([z_tr_img.cpu().numpy(), z_br_img.cpu().numpy()]) 
        linfit = interp1d([1,num_interps+1], stack, axis=0)
        coln_interp_latentsPart = linfit(list(range(1,num_interps+1)))

        coln_interp_latents = np.vstack([coln_interp_latentsPart, z_br_img.cpu().numpy()]) 

        print(coln_interp_latents.shape)
        
        for row_id in range(col1_interp_latents.shape[0]):
            cur_start = col1_interp_latents[row_id:row_id+1]
            cur_end = coln_interp_latents[row_id:row_id+1]

            stack = np.vstack([cur_start, cur_end]) 
            linfit = interp1d([1,num_interps+1], stack, axis=0)
            interp_latentsPart = linfit(list(range(1,num_interps+1)))

            interp_latents = np.vstack([ interp_latentsPart, cur_end])

            print(interp_latents.shape) 

            if row_id != 0:
                ### Append
                all_interps = np.concatenate((all_interps,interp_latents),axis = 0)
            else:
                ### Create
                all_interps = interp_latents


        all_z_org = all_z = torch.from_numpy(all_interps)
        # cur_z_org = cur_z = torch.from_numpy(interp_latents[j:j+1])
        all_z = img_transform(all_z,transform,transform_flag)
        # print(all_z.shape,label.shape)
        for ind in range((num_interps+1)**2):
            cur_img = G(all_z[ind:ind+1].cuda(), label[ind:ind+1].cuda(), truncation_psi=truncation_psi, noise_mode=noise_mode).cpu()
            if ind != 0:
                all_img = torch.cat((all_img,cur_img),0)
            else:
                all_img = cur_img
        # all_img = G(all_z[0:num_interps**2], label[0:num_interps**2], truncation_psi=truncation_psi, noise_mode=noise_mode)
        all_img = (all_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        all_z_org = (all_z_org * 127.5 + 128).clamp(0, 255).to(torch.uint8)


        img_grid = make_grid(all_img, nrow = num_interps+1)
        z_grid = make_grid(all_z_org, nrow = num_interps+1)

        # print(img_grid.shape)
        # exit(0)


        PIL.Image.fromarray(z_grid.permute(1,2,0).cpu().numpy(), 'RGB').save(f'{midpath}/seeds{tl_seed:04d}_{tr_seed:04d}_{bl_seed:04d}_{br_seed:04d}.png')
        PIL.Image.fromarray(img_grid.permute(1,2,0).cpu().numpy(), 'RGB').save(f'{impath}/seeds{tl_seed:04d}_{tr_seed:04d}_{bl_seed:04d}_{br_seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

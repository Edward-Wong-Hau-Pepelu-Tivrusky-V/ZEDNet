'''

First Script to Run to Create Samplers for Synthetic Images.

Identify all labelled images (volume or slice labels) and sample the fluorescence based on the distance to membrane.

Output: 
 - Fluorescent samplers that will be used to texture synthetic images
 - Metadata from labels to inform automatic sythetic image generation

'''

import numpy as np
from tifffile import imread, imwrite
import pickle
from tqdm import tqdm

import sys
import os
import argparse
import glob
import re

parser = argparse.ArgumentParser(description='Sample Real Fluorecence for Synthetic Images.')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--data-folder', type=str, required=True,
                    help='Subfolder name inside data-root containing the images (e.g., Fluo-C3DH-H157-train/).')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()

assert args.data_root[-1] == '/'
assert args.data_folder[-1] == '/'

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
if args.data_root not in sys.path:
    (sys.path).append(args.data_root)

from synthetic_generator import fluorescent_sampler
from utils import load_config

if args.global_config is None:
    global_params = load_config(f'config/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']

disable_labels = global_params['SAMPLER']['DISABLE_LABELS']
use_silver_truth = global_params['SAMPLER']['SILVER_TRUTH']
normalise = global_params['SAMPLER']['NORMALISE']
min_dist = global_params['SAMPLER']['MIN_DIST']
max_dist = global_params['SAMPLER']['MAX_DIST']
dx = global_params['SAMPLER']['DX']
skip = global_params['SAMPLER']['SKIP']

# print(f'Region Width: {dx:.3f}' + r'\mu m')
# print(f'Range: ({min_dist:.3f}, {max_dist:.3f})')
# print(f'Skip {skip}')

data_path = args.data_root + args.data_folder

## Main Code

# get labels
if use_silver_truth:
    label_type = 'ST'
else:
    label_type = 'GT'

# find all labelled images
man_seg_files = sorted(glob.glob(f'{data_path}*_{label_type}/SEG/cellpose_seg*.tif'))
N = len(man_seg_files)

os.makedirs(f'data_generator/sampled_data/', exist_ok=True)

for f in tqdm(man_seg_files, desc='Sampling fluorescence'):

    filename = os.path.basename(f)
    sample_id = filename.replace('.tif', '')
    
    # load mask
    mask = imread(f)
    image_filename = filename.replace('cellpose_seg_', '')
    seq_folder = os.path.basename(os.path.dirname(os.path.dirname(f))).replace('_GT', '')
    
    image_path = os.path.join(data_path, seq_folder, image_filename)

    image = imread(image_path)

    if normalise:
        image = (image - image.mean()) / image.std()

    save_dir = f'data_generator/sampled_data/data_{args.dataset_id}/'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{sample_id}.pkl')

    sampler = fluorescent_sampler(
        image,
        (mask > 0),
        labelled_mask=mask,
        sampling=sampling,
        dx=dx,
        min_dist=min_dist,
        max_dist=max_dist,
        save_path=save_path,
        skip=skip,
        disable_labels=disable_labels,
    )

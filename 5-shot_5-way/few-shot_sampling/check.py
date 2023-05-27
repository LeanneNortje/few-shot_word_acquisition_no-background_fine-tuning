#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

# import os
import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from tqdm import tqdm
from pathlib import Path
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# import time
# import json
# import torchvision.models as imagemodels
# import torch.utils.model_zoo as model_zoo
import numpy as np
# from image_models import *
# import statistics
# from align import align_semiglobal, score_semiglobal

# from torchvision.io import read_image
# from torchvision.models import *

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"
    
train_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
train_neg_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
train_images = set()
train_audio = set()

for id in train_id_lookup:
    pos = set()
    neg = set()
    
    for name in train_id_lookup[id]['images']: 
        pos.update(train_id_lookup[id]['images'][name])
        train_images.update(train_id_lookup[id]['images'][name])

    for n in train_neg_id_lookup[id]['images']:

        neg.update(train_neg_id_lookup[id]['images'][n])
        train_images.update(train_neg_id_lookup[id]['images'][n])

    print(f'Image overlap: {pos.intersection(neg)}')

    pos = set()
    neg = set()
    for name in train_id_lookup[id]['audio']: 
        pos.update(train_id_lookup[id]['audio'][name])
        train_audio.update(train_id_lookup[id]['audio'][name])

    for n in train_neg_id_lookup[id]['audio']:
        neg.update(train_neg_id_lookup[id]['audio'][n])
        train_audio.update(train_neg_id_lookup[id]['audio'][n])

    print(f'Audio overlap: {pos.intersection(neg)}')


val_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['lookup'].item()
val_neg_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
val_images = set()
val_audio = set()

for id in val_id_lookup:
    pos = set()
    neg = set()
    for name in val_id_lookup[id]['images']: 
        pos.update(val_id_lookup[id]['images'][name])
        val_images.update(val_id_lookup[id]['images'][name])

    for n in val_neg_id_lookup[id]['images']:
        neg.update(val_neg_id_lookup[id]['images'][n])
        val_images.update(val_neg_id_lookup[id]['images'][n])

    print(f'Image overlap: {pos.intersection(neg)}')

    pos = set()
    neg = set()
    for name in val_id_lookup[id]['audio']: 
        pos.update(val_id_lookup[id]['audio'][name])
        val_audio.update(val_id_lookup[id]['audio'][name])

    for n in val_neg_id_lookup[id]['audio']:
        neg.update(val_neg_id_lookup[id]['audio'][n])
        val_audio.update(val_neg_id_lookup[id]['audio'][n])

    print(f'Audio overlap: {pos.intersection(neg)}')

print(f'Image training+validation overlap: {train_images.intersection(val_images)}')
print(f'Audio training+validation overlap: {train_audio.intersection(val_audio)}')

ss_save_fn = '../support_set/support_set.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
ss_images = set()
ss_wavs = set()

for wav_name in support_set:
    wav, im, spkr, start, stop, word = support_set[wav_name]
    ss_images.add(im)
    ss_wavs.add(wav)

print(f'Image support set+training overlap: {ss_images.intersection(train_images)}')
print(f'Audio support set+training overlap: {ss_wavs.intersection(train_audio)}')
print(f'Image support set+validation overlap: {ss_images.intersection(val_images)}')
print(f'Audio support set+validation overlap: {ss_wavs.intersection(val_audio)}')


test_save_fn = '../data/test_episodes.npz'
test_episodes = np.load(test_save_fn, allow_pickle=True)['episodes'].item()
test_images = set()
test_wavs = set()

for key in test_episodes:
    if key == 'matching_set':
        for word in test_episodes[key]:
            for im in test_episodes[key][word]:
                test_images.add(im)
    else:
        for word in test_episodes[key]['queries']:
            test_wavs.add(test_episodes[key]['queries'][word][0])   


print(f'Image test epiodes+support set overlap: {test_images.intersection(ss_images)}')
print(f'Audio test epiodes+support set overlap: {test_wavs.intersection(ss_wavs)}')
print(f'Image test epiodes+training overlap: {test_images.intersection(train_images)}')
print(f'Audio test epiodes+training overlap: {test_wavs.intersection(train_audio)}')
print(f'Image test epiodes+validation overlap: {test_images.intersection(val_images)}')
print(f'Audio test epiodes+validation overlap: {test_wavs.intersection(val_audio)}')
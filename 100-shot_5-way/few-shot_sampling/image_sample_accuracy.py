#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import re

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for word in key:
    print(f'{key[word]:<3}: {word}')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

image2wav = {}

for fn in Path('../../Datasets/spokencoco/SpokenCOCO').rglob('*.json'):
    with open(fn, 'r') as f:
        data = json.load(f)

    data = data['data']

    for entry in tqdm(data):
        image = Path(entry['image']).stem
        for caption in entry['captions']:
            
            wav = Path(caption['wav']).stem
            # if wav not in image2wav:
            #     image2wav[wav] = []
            image2wav[wav] = image

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(Path('../../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        im = image2wav[wav]
        if label in vocab or (label == 'hydrant' and prev == 'fire' and wav == prev_wav):
            if im not in alignments: alignments[im] = {}
            if label == 'hydrant' and prev == 'fire': 
                label = prev + " " + label
                start = prev_start
            if label not in alignments[im]: alignments[im][label] = (int(float(start)*50), int(float(stop)*50))
        prev = label
        prev_wav = wav
        prev_start = start

train_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
train_neg_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
train_image_acc = 0
train_image_total = 0
train_audio_acc = 0
train_audio_total = 0
train_audio_per_keyword_acc = {}

for id in train_id_lookup:
    if key[id] not in train_audio_per_keyword_acc: train_audio_per_keyword_acc[key[id]] = {"acc": 0, "total": 0}
    
    for name in train_id_lookup[id]['images']: 
        
        if name not in alignments: continue
        for word in alignments[name]:
            if key[id] == word: 
                train_audio_acc += 1
                train_audio_per_keyword_acc[key[id]]["acc"] += 1
            train_audio_total += 1  
            train_audio_per_keyword_acc[key[id]]["total"] += 1

print(f'Training accuracies:')
for word in train_audio_per_keyword_acc:
    a = train_audio_per_keyword_acc[word]['acc']
    t = train_audio_per_keyword_acc[word]['total']
    print(f'{word}: {a}/{t} = {a/t} = {100*a/t:.2f}%')
print(f'Overall: {train_audio_acc}/{train_audio_total} = {train_audio_acc/train_audio_total} = {100*train_audio_acc/train_audio_total:.2f}%')


val_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['lookup'].item()
val_neg_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
val_image_acc = 0
val_image_total = 0
val_audio_acc = 0
val_audio_total = 0
val_audio_per_keyword_acc = {}

for id in val_id_lookup:
    if key[id] not in val_audio_per_keyword_acc: val_audio_per_keyword_acc[key[id]] = {"acc": 0, "total": 0}

    for name in val_id_lookup[id]['images']: 
        if name not in alignments: continue
        for word in alignments[name]:
            if key[id] == word: 
                val_audio_acc += 1
                val_audio_per_keyword_acc[key[id]]["acc"] += 1
            val_audio_total += 1  
            val_audio_per_keyword_acc[key[id]]["total"] += 1

print(f'Validation accuracies:')
for word in val_audio_per_keyword_acc:
    a = val_audio_per_keyword_acc[word]['acc']
    t = val_audio_per_keyword_acc[word]['total']
    print(f'{word}: {a}/{t} = {a/t} = {100*a/t:.2f}%')
print(f'Overall: {val_audio_acc}/{val_audio_total} = {val_audio_acc/val_audio_total} = {100*val_audio_acc/val_audio_total:.2f}%')
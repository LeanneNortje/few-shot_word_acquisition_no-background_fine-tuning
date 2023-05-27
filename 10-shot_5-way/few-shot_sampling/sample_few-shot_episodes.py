#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import re
import numpy as np
from tqdm import tqdm

K = 5
num_episodes = 1000

files = "../../Datasets/spokencoco/SpokenCOCO"
val_fn = Path(files) / 'SpokenCOCO_val.json'
val = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

base = []
with open('../data/base_keywords.txt', 'r') as f:
    for keyword in f:
        base.append(' '.join(keyword.split()))
base = base[0:10]
print(base)

with open(val_fn, 'r') as f: data = json.load(f)
data = data['data']

for entry in data: 
    im = entry['image']
    for caption in entry['captions']:
        for word in vocab:
            if re.search(word, caption['text'].lower()) is not None:
                if word not in val: val[word] = []
                val[word].append((im, caption['wav'], caption['speaker']))

test_episodes = {}

matching_set = {}

##################################
# Test matching set 
##################################

for entry in data:
    im = entry['image']
    if im not in matching_set: matching_set[im] = set()
    for caption in entry['captions']:
        
        for word in vocab:
            if re.search(word, caption['text'].lower()) is not None:
                # if im not in matching_set: matching_set[im] = set()
                matching_set[im].add(word)
        # used_images.add(im)
test_episodes['matching_set'] = matching_set
print(len(matching_set))

##################################
# Test queries  
##################################

for word in vocab:

    instances = np.random.choice(np.arange(0, len(val[word])), num_episodes)        
    for episode_num in tqdm(range(num_episodes)):

        if episode_num not in test_episodes: test_episodes[episode_num] = {'queries': {}}
        entry = val[word][instances[episode_num]]
        test_episodes[episode_num]['queries'][word] = (entry[1], entry[2])

test_save_fn = '../data/test_episodes'
np.savez_compressed(
    Path(test_save_fn).absolute(), 
    episodes=test_episodes
    )
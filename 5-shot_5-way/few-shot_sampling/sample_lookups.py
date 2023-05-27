#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import re

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

files = "../../Datasets/spokencoco/SpokenCOCO"
train_fn = Path(files) / 'SpokenCOCO_train.json'
with open(train_fn, 'r') as f:
    data = json.load(f)

data = data['data']
neg_imgs = set()
neg_wavs = set()

for entry in tqdm(data):
    image = entry['image']
    for caption in entry['captions']:
        # for word in base:
        #     if re.search(word, caption['text'].lower()) is not None:
        c = False
        for v in vocab:
            if re.search(v, caption['text'].lower()) is not None:
                c = True
        if c is False: 
            neg_imgs.add(image)
            neg_wavs.add(caption['wav'])

val_neg_imgs = np.random.choice(list(neg_imgs), 7000, replace=False)
train_neg_imgs = [entry for entry in list(neg_imgs) if entry not in val_neg_imgs]
val_neg_wavs = np.random.choice(list(neg_wavs), 7000, replace=False)
train_neg_wavs = [entry for entry in list(neg_wavs) if entry not in val_neg_wavs]

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['key'].item()
id_to_word_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for l in key:
    print(f'{key[l]}: {l}')

audio_samples = np.load(Path("../data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()
image_samples = np.load(Path("../data/sampled_img_data.npz"), allow_pickle=True)['data'].item()

ss_save_fn = '../support_set/support_set_5.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()

s = {}
for wav_name in support_set:
    _, _, _, _, _, word = support_set[wav_name]
    if key[word] not in s: s[key[word]] = []
    s[key[word]].append(support_set[wav_name])
print(len(s))

val_number = 100
aud_train = []
aud_val = []

# Speakers
speakers= {} 
for id in audio_samples:
    for wav in audio_samples[id]:
        spkr = Path(wav).stem.split('-')[0]
        if spkr not in speakers: speakers[spkr] = {}
        if id not in speakers[spkr]: speakers[spkr][id] = []
        speakers[spkr][id].append(str(wav))

id_count = {}
for spkr in speakers:
    if spkr not in id_count: id_count[spkr] = {}
    for id in speakers[spkr]:
        if id not in id_count: id_count[spkr][id] = 0
        id_count[spkr][id] += 1

possible_val_speakers = set()
for spkr in id_count:
    values = set([id_count[spkr][id] for id in id_count[spkr]])
    if values == {1}: possible_val_speakers.add(spkr)

val_speakers = set()
val_dict = {}
for spkr in speakers:
    if spkr in possible_val_speakers:
        for id in speakers[spkr]:
            if id not in val_dict: val_dict[id] = []
            if len(val_dict[id]) < val_number: val_dict[id].extend(speakers[spkr][id])
for id in val_dict:
    aud_val.extend(val_dict[id])

for id in audio_samples:
    options = [str(wav) for wav in audio_samples[id] if str(wav) not in aud_val]
    aud_train.extend(options)
 
val_speakers = set()
train_speakers = set()
for wav in aud_val: 
    val_speakers.add(Path(wav).stem.split('-')[0])
    
for wav in aud_train: 
    train_speakers.add(Path(wav).stem.split('-')[0])

print(len(train_speakers.intersection(val_speakers)), len(aud_train), len(aud_val))

train = []
val = []
for id in image_samples:
    have_to_add = [str(img) for img in image_samples[id] if str(img) in val and str(img) not in train]
    options = [str(img) for img in image_samples[id] if str(img) not in val and str(img) not in train]
    v = np.random.choice(options, val_number-len(have_to_add), replace=False)
    val.extend(v)
    val.extend(have_to_add)
    # t = list(set(audio_samples[id]) - set(v))
    options = [str(img) for img in image_samples[id] if str(img) not in val]
    train.extend(options)

# Training 

pos_lookup = {}

for id in s:
    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav, image, spkr, start, stop, word in s[id]:
        
        wav_name = Path(wav).stem
        if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
        pos_lookup[id]['audio'][wav_name].append(wav)

        image_name = Path(image).stem
        if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
        pos_lookup[id]['images'][image_name].append(str(image))

for id in audio_samples:

    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav in audio_samples[id]:
        wav_name = wav.stem
        wav = str(wav)
        if wav in aud_train:
            if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
            pos_lookup[id]['audio'][wav_name].append(wav)

neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio training negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in all_ids: #tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['audio'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for w in list(set(temp)):
                neg_lookup[id]['audio'][neg_id].append(w)


for id in image_samples:
    if id not in pos_lookup: continue #pos_lookup[id] = {"audio": {}, "images":{}, "neg_images": {}}
    for image in image_samples[id]:
        image_name = image.stem
        image = str(image)
        if image in train: 
            if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
            pos_lookup[id]['images'][image_name].append(str(image))


print("*", len(train_neg_imgs))
for id in tqdm(sorted(pos_lookup), desc='Sampling image training negatives'):
    images_with_id = list(set([im for im in pos_lookup[id]['images']]))
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
        
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in images_with_id]
        if len(temp) > 0:
            neg_lookup[id]['images'][neg_id] = temp 
            # neg_lookup[id]['images'][neg_id].extend(list(set(train_neg_imgs)-set(temp)))
            train_neg_imgs = list(set(train_neg_imgs) - set(temp))
print(len(train_neg_imgs))

for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))

np.savez_compressed(
    Path("../data/train_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup,
    base_negs=train_neg_imgs
    )

# Validation

pos_lookup = {}

for id in audio_samples:

    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav in audio_samples[id]:
        wav_name = wav.stem
        wav = str(wav)
        if wav in aud_val:
            if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
            pos_lookup[id]['audio'][wav_name].append(wav)

neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio validation negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        # if len(temp) > 0:
        neg_lookup[id]['audio'][neg_id] = temp
        # neg_lookup[id]['audio'][neg_id].extend(list(set(val_neg_wavs) - set(temp)))
        # for w in list(set(val_neg_wavs) - set(temp)):
        for w in list(set(temp)):
            neg_lookup[id]['audio'][neg_id].append(w)

for id in image_samples:
    if id not in pos_lookup: continue #pos_lookup[id] = {"audio": {}, "images":{}, "neg_images": {}}
    for image in image_samples[id]:
        image_name = image.stem
        image = str(image)
        if image in val: 
            if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
            pos_lookup[id]['images'][image_name].append(str(image))  

for id in tqdm(sorted(pos_lookup), desc='Sampling image validation negatives'):
    images_with_id = list(set([im for im in pos_lookup[id]['images']]))
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
        
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in images_with_id]
        # if len(temp) > 0:
        neg_lookup[id]['images'][neg_id] = temp 
        val_neg_imgs = list(set(val_neg_imgs) - set(temp))
        # neg_lookup[id]['images'][neg_id].extend(list(set(val_neg_imgs)-set(temp)))

for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))


np.savez_compressed(
    Path("../data/val_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup,
    base_negs=val_neg_imgs
)
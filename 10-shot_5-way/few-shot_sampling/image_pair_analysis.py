#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
import json
import re 

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for word in key:
    print(f'{key[word]:<3}: {word}')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

transcriptions = {}
prev = ''
prev_wav = ''
prev_start = 0

# for json_fn in Path('../../Datasets/spokencoco/SpokenCOCO').rglob('*train*.json'):
#     with open(json_fn, 'r') as f:
#         data = json.load(f)['data']
    
#     for entry in data:
#         name = Path(entry['image']).stem
#         for cap in entry['captions']:
#             text = cap['text'].lower()
#             if name not in transcriptions: transcriptions[name] = []
#             # transcriptions[Path(cap['wav']).stem] = text
#             for word in vocab:
#                 if re.search(word, text) is not None: transcriptions[name].append(word)

categories = {}
for json_fn in Path('../../Datasets/spokencoco/panoptic_annotations_trainval2017').rglob('*.json'):
    if re.search('__MACOSX', str(json_fn)) is not None: continue
    with open(json_fn, 'r') as f:
        data = json.load(f)

    # for entry in data['images']:
    #     print(entry)
    #     break

    for entry in data['categories']:
        for word in vocab:
            if re.search(word, entry['name']) is not None: categories[entry['id']] = entry['name']
    # break
for id in categories:
    print(id, categories[id])

image_annotations = {}
for json_fn in Path('../../Datasets/spokencoco/panoptic_annotations_trainval2017').rglob('*.json'):
    if re.search('__MACOSX', str(json_fn)) is not None: continue
    with open(json_fn, 'r') as f:
        data = json.load(f)

    for entry in data['annotations']:
        if entry['image_id'] not in image_annotations: image_annotations[entry['image_id']] = set()
        for seg in entry['segments_info']:
            if seg['category_id'] in categories: image_annotations[entry['image_id']].add(categories[seg['category_id']])
    
train_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
val_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['lookup'].item()
train_neg_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
train_audio_per_keyword_acc = {}
results = {}

for id in train_id_lookup:

    pred_word = key[id]

    for name in train_id_lookup[id]['images']: 
        # if name not in transcriptions: continue
        image_name = int(name.split('_')[-1])
        if name not in results: results[name] = {'pred': [], 'gt': set()}
        results[name]['pred'].append(pred_word)
        results[name]['gt'].update(image_annotations[image_name])

for id in val_id_lookup:

    pred_word = key[id]

    for name in val_id_lookup[id]['images']: 
        # if name not in transcriptions: continue
        image_name = int(name.split('_')[-1])
        if name not in results: results[name] = {'pred': [], 'gt': set()}
        results[name]['pred'].append(pred_word)
        results[name]['gt'].update(image_annotations[image_name])

for i, name in enumerate(results):
    
    for p_w in results[name]['pred']: 
        if p_w not in train_audio_per_keyword_acc: train_audio_per_keyword_acc[p_w] = {"tp": 0, "fn": 0, "fp": 0, 'counts': 0}
        
        if p_w in results[name]['gt']: train_audio_per_keyword_acc[p_w]['tp'] += 1
        else: train_audio_per_keyword_acc[p_w]['fp'] += 1
        train_audio_per_keyword_acc[p_w]['counts'] += 1

    for g_w in results[name]['gt']: 
        if g_w not in results[name]['pred']: train_audio_per_keyword_acc[p_w]['fn'] += 1

t_tp = 0
t_fp = 0
t_fn = 0
print(f'Training accuracies:')
for word in train_audio_per_keyword_acc:
    tp = train_audio_per_keyword_acc[word]['tp']
    t_tp += tp
    fp = train_audio_per_keyword_acc[word]['fp']
    t_fp += fp
    fn = train_audio_per_keyword_acc[word]['fn']
    t_fn += fn
    pres = tp / (tp + fp)
    recall = tp / (tp + fn)
    c = train_audio_per_keyword_acc[word]['counts']
    print(f'{word:<10}\t Counts: {c}\\t Precision: {100*pres:.2f}%')#\t Recall: {100*recall:.2f}%')
pres = t_tp / (t_tp + t_fp)
recall = t_tp / (t_tp + t_fn)
a = 'Overall'
print(f'{a:<10}\t Precision: {100*pres:.2f}%')#\t Recall: {100*recall:.2f}%')
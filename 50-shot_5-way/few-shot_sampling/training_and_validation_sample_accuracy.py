#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for word in key:
    print(f'{key[word]:<3}: {word}')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(Path('../../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        if label in vocab or (label == 'hydrant' and prev == 'fire' and wav == prev_wav):
            if wav not in alignments: alignments[wav] = {}
            if label == 'hydrant' and prev == 'fire': 
                label = prev + " " + label
                start = prev_start
            if label not in alignments[wav]: alignments[wav][label] = (int(float(start)*50), int(float(stop)*50))
        prev = label
        prev_wav = wav
        prev_start = start
    
train_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
train_neg_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
train_audio_per_keyword_acc = {}
results = {}

for id in train_id_lookup:

    pred_word = key[id]

    for name in train_id_lookup[id]['audio']: 
        if name not in alignments: continue
        if name not in results: results[name] = {'pred': [], 'gt': []}
        results[name]['pred'].append(pred_word)
        results[name]['gt'].extend(list(alignments[name].keys()))

for i, name in enumerate(results):
    
    for p_w in results[name]['pred']: 
        if p_w not in train_audio_per_keyword_acc: train_audio_per_keyword_acc[p_w] = {"tp": 0, "fn": 0, "fp": 0}
        
        if p_w in results[name]['gt']: train_audio_per_keyword_acc[p_w]['tp'] += 1
        else: train_audio_per_keyword_acc[p_w]['fp'] += 1

    for g_w in results[name]['gt']: 
        if g_w not in results[name]['pred']: train_audio_per_keyword_acc[p_w]['fn'] += 1


# for id in train_id_lookup:
#     if key[id] not in train_audio_per_keyword_acc: train_audio_per_keyword_acc[key[id]] = {"tp": 0, "fn": 0, "fp": 0}

#     pred_word = key[id]

#     for name in train_id_lookup[id]['audio']: 
#         if name not in alignments: continue
        
#         gt_words = list(alignments[name].keys())

#         if pred_word in gt_words: train_audio_per_keyword_acc[key[id]]['tp'] += 1
#         else: train_audio_per_keyword_acc[key[id]]['fp'] += 1

    
#         # for word in alignments[name]:
#         #     if key[id] == word: 
# #                 train_audio_acc += 1
# #                 train_audio_per_keyword_acc[key[id]]["acc"] += 1
# #             train_audio_total += 1  
# #             train_audio_per_keyword_acc[key[id]]["total"] += 1

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
    print(f'{word:<10}\t Precision: {100*pres:.2f}%\t Recall: {100*recall:.2f}%')
pres = t_tp / (t_tp + t_fp)
recall = t_tp / (t_tp + t_fn)
a = 'Overall'
print(f'{a:<10}\t Precision: {100*pres:.2f}%\t Recall: {100*recall:.2f}%')
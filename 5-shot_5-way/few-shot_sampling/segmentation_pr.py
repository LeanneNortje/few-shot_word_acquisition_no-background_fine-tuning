#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from align import align_semiglobal, score_semiglobal

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

audio_segments_dir = Path('../../QbERT/segments')
# train_save_fn = '../data/train_for_preprocessing.npz'
ss_save_fn = '../support_set/support_set_5.npz'
image_base = Path('../../Datasets/spokencoco')
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
files = "../../Datasets/spokencoco/SpokenCOCO"
train_fn = Path(files) / 'SpokenCOCO_train.json'
train = {}
pam = np.load("pam.npy")

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

label_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
key = {}
for id in label_key:
    key[label_key[id]] = id
    print(f'{label_key[id]:<3}: {id}')

class_count = {}
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
            if label not in alignments[wav]: 
                alignments[wav][label] = (int(float(start)*100), int(float(stop)*100))
                if label not in class_count: class_count[label] = 0
                class_count[label] += 1
        prev = label
        prev_wav = wav
        prev_start = start

print(class_count)

s_audio = {}
s_wavs = []

for wav_name in tqdm(support_set):
    wav, img, spkr, start, end, word = support_set[wav_name]
    fn = Path(wav).relative_to(*Path(wav).parts[:1]).with_suffix('.npz')
    fn = audio_segments_dir / fn
    query = np.load(fn)
    x = query["codes"][query["boundaries"][:-1]]

    x0, = np.where(query["boundaries"] <= int(start))
    x0 = x0[-1]
    xn, = np.where(query["boundaries"] >= int(end))
    xn = xn[0]
    x = x[x0-1:xn+1]

    if word not in s_audio: s_audio[word] = []
    s_audio[word].append(x)
    s_wavs.append(Path(wav).stem)

with open(train_fn, 'r') as f:
    data = json.load(f)

data = data['data']
query_scores = {}
record = {}

for q_word in s_audio:
    id = key[q_word]
    for entry in tqdm(data, desc=f'{q_word}({id})'):

        for caption in entry['captions']:
            wav = caption['wav']
            wav_name = Path(wav).stem
            if wav_name in s_wavs: continue

            if id not in query_scores: query_scores[id] = {'values': [], 'wavs': []}

            fn = Path(wav).relative_to(*Path(wav).parts[:1]).with_suffix('.npz')
            fn = audio_segments_dir / fn
            test = np.load(fn)
            y = test["codes"][test["boundaries"][:-1]]

            max_score = -np.inf
            for x in s_audio[q_word]:
                path, p, q, score = align_semiglobal(x, y, pam, 3)
                indexes, = np.where(np.array(p) != -1)
                if len(indexes) != 0:
                    start, end = indexes[1], indexes[-1]
                    norm_score = score / (end - start)

                    if norm_score > max_score:
                        max_score = norm_score
                        
            query_scores[id]['values'].append(max_score)
            query_scores[id]['wavs'].append(wav)    
    
        # if len(query_scores[id]['values']) >= 5000: break

for id in query_scores:
    
    indices = np.argsort(query_scores[id]['values'])[::-1]
    print(len(indices), class_count[label_key[id]])
    i = 0
    while i < class_count[label_key[id]]:

        wav = Path(query_scores[id]['wavs'][indices[i]])
        wav_name = wav.stem

        if wav not in record: record[wav] = []
        record[wav].append(id)
        i += 1



threshold = 0.5 / 0.01
pr = {}
wavs = {}
count = 0
for wav in tqdm(record):
    for id in record[wav]:

        if id not in pr: pr[id] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        wav = Path(wav)
        wav_name = wav.stem

        if wav_name not in alignments: continue
        count += 1
        wavs[wav_name] = wav

        if label_key[id] in alignments[wav_name]: 
            pr[id]['tp'] += 1
        else: pr[id]['fp'] += 1


for wav_name in alignments:

    for word in alignments[wav_name]:
        id = key[word]
        for wav in record:
            name = Path(wav).stem
            if wav_name == name:
                if id not in record[wav]: pr[id]['fn'] += 1
            else: pr[id]['tn'] += 1

print(pr)
t_tp = 0
t_fp = 0
t_fn = 0
t_tn = 0
for id in pr:
    tp = pr[id]['tp']
    fp = pr[id]['fp']
    fn = pr[id]['fn']
    tn = pr[id]['tn']
    pres = tp / (tp+fp)
    rec = tp / (tp+fn)
    print(f'{label_key[id]}: {(2 * pres * rec)/(pres + rec)}')# Recall: {tp/(tp+fn)}')
    t_tp += tp
    t_fp += fp
    t_fn += fn
    t_tn += tn
pres = t_tp / (t_tp+t_fp)
recall = t_tp / (t_tp+t_fn)
print(f'Overall: {(2 * pres * rec)/(pres + rec)}')# Recall: {t_tp/(t_tp+t_fn)}')
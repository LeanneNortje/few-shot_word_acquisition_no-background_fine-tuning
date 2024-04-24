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
import librosa
import scipy.signal
import scipy

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

label_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
key = {}
for id in label_key:
    key[label_key[id]] = id
    print(f'{label_key[id]:<3}: {id}')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

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
distances = {}
for fn in tqdm(list(Path('dtw').rglob('*.npz'))):
    name = fn.stem
    d = np.load(Path(f'dtw/{name}.npz'), allow_pickle=True)['distances'].item()
    k = list(d.keys())[0]
    distances[name] = d[k]
    # if len(distances) == 10000: break
print(len(distances))
    
sampled = np.load(Path("../data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()
ss_save_fn = '../support_set/support_set_5.npz'
files = "../../Datasets/spokencoco/SpokenCOCO"
audio_segments_dir = Path(files)
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
s_wavs = {}

for wav_name in tqdm(support_set):
    wav, img, spkr, start, end, word = support_set[wav_name]
    name = Path(wav.split('.')[0]+'_'+word)
    fn = (Path('../support_set') / name).with_suffix('.wav')
    if word not in s_wavs: s_wavs[word] = []
    s_wavs[word].append(wav)

train_fn = Path(files) / 'SpokenCOCO_train.json'
with open(train_fn, 'r') as f:
    data = json.load(f)

data = data['data']
query_scores = {}
record = {}

for q_word in vocab:
    id = key[q_word]
    for entry in tqdm(data, desc=f'{q_word}({id})'):

        for caption in entry['captions']:
            wav = caption['wav']
            wav_name = Path(wav).stem
            if wav_name in s_wavs or wav_name not in distances: continue

            if id not in query_scores: query_scores[id] = {'values': [], 'wavs': [], 'segments': []}

            max_sore = -np.inf

            for q_fn in s_wavs[q_word]:
                p = str(Path('../support_set') / Path(q_fn).parent / Path(Path(q_fn).stem + '_' + q_word + '.wav'))
                w0, wT, norm_score = distances[wav_name][p]

                if norm_score > max_sore: 
                    max_sore = norm_score

            query_scores[id]['values'].append(max_sore)
            query_scores[id]['wavs'].append(wav)
 

newly_labeled = {}

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
    print(label_key[id], pres, rec)
    print(f'{label_key[id]}: {(2 * pres * rec)/(pres + rec)}')# Recall: {tp/(tp+fn)}')
    t_tp += tp
    t_fp += fp
    t_fn += fn
    t_tn += tn
pres = t_tp / (t_tp+t_fp)
recall = t_tp / (t_tp+t_fn)
print(f'Overall: {(2 * pres * rec)/(pres + rec)}')# Recall: {t_tp/(t_tp+t_fn)}')
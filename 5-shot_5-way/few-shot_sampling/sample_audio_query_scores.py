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

key = {}
id_to_word_key = {}
for i, l in enumerate(vocab):
    key[l] = i
    id_to_word_key[i] = l
    print(f'{i}: {l}')

np.savez_compressed(
    Path('../data/label_key'),
    key=key,
    id_to_word_key=id_to_word_key
)

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
                        if wav not in record: record[wav] = {}
                        record[wav][id] = (path, start, end, p, q, indexes)
            query_scores[id]['values'].append(max_score)
            query_scores[id]['wavs'].append(wav)    
    
        # if len(query_scores[id]['values']) == 1900: break

for id in query_scores:
    print(len(query_scores[id]['values']), len(query_scores[id]['wavs']))

save_dir = Path('segment_examples')
audio_dir = Path("../../Datasets/spokencoco/SpokenCOCO")
top_N = 600
newly_labeled = {}

for id in query_scores:
    indices = np.argsort(query_scores[id]['values'])[::-1]
    i = 0
    while i < top_N:
        wav = Path(query_scores[id]['wavs'][indices[i]])
        wav_name = wav.stem
        fn = save_dir / Path(id_to_word_key[id]) / wav_name
        fn.parent.mkdir(parents=True, exist_ok=True)

        segment_fn = wav.relative_to(*wav.parts[:1]).with_suffix('.npz')
        segment_fn = audio_segments_dir / segment_fn
        test = np.load(segment_fn)
        path, start, end, p, q, indexes = record[str(wav)][id]
        _, b0 = path[start - 1]
        _, bT = path[end]
        w0, wT = 0.02 * test["boundaries"][b0 - 1], 0.02 * test["boundaries"][bT]
        offset = int(w0 * 16000)
        frames = int(np.abs(wT - w0) * 16000)
        aud, sr = torchaudio.load(audio_dir / wav, frame_offset=offset, num_frames=frames)
        
        if frames == aud.size(1):
        
            if wav_name == 'm3t1oiftx18474-3TR2532VIPUCJEHB0694KVVG1QFJ6A_266069_594194': print(aud.size(), frames, offset)
            torchaudio.save(fn.with_suffix('.wav'), aud, sr)

            if id not in newly_labeled: newly_labeled[id] = []
            newly_labeled[id].append(wav)
            i += 1

for id in newly_labeled:
    print(id, len(newly_labeled[id]))

np.savez_compressed(
    Path("../data/sampled_audio_data"), 
    data=newly_labeled
)
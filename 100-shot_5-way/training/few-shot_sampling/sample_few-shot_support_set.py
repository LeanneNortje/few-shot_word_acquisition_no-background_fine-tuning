#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from tqdm import tqdm
from pathlib import Path
import json
import re
import numpy as np
import torchaudio

K = 100
num_episodes = 1000

def load_fn(fn, word_points, vocab):
    count = {}
    with open(fn, 'r') as f:
        data = json.load(f)

    data = data['data']

    for entry in tqdm(data):
        image = entry['image']

        for caption in entry['captions']:
            
            for word in vocab:
                if re.search(word, caption['text'].lower()) is not None:
                    if word not in word_points:
                        word_points[word] = []
                    word_points[word].append((image, caption['wav'], caption['speaker']))
                    if word not in count: count[word] = 0
                    count[word] += 1

    print('Word counts')
    for word in count:
        print(f'{word}: {count[word]}')
                    

files = "../../Datasets/spokencoco/SpokenCOCO"
train_fn = Path(files) / 'SpokenCOCO_train.json'
train = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

print(vocab)
load_fn(train_fn, train, vocab) 

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

support_set = {}

##################################
# Support set 
##################################
audio_dir = Path('/home/leannenortje/Datasets/spokencoco/SpokenCOCO')
save_dir = Path('../support_set')

for word in vocab:
    filtered = [(i, w, s) for i, w, s in train[word] if Path(w).stem in alignments and Path(w).stem not in support_set]
    instances = np.random.choice(np.arange(0, len(filtered)), K, replace=False)
    
    for im, wav, spkr in [filtered[i] for i in instances]:
        name = Path(wav).stem
        support_set[name] = (wav, im, spkr, alignments[name][word][0], alignments[name][word][1], word)

        dur = int((float(alignments[name][word][1])*0.02 - float(alignments[name][word][0])*0.02)*16000)
        offset = int(float(alignments[name][word][0])*16000*0.02)
        aud, sr = torchaudio.load(audio_dir / wav, frame_offset=offset, num_frames=dur)
        out_path = save_dir / wav
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, aud, sr)

np.savez_compressed(
    save_dir / Path('support_set'), 
    support_set=support_set
    )
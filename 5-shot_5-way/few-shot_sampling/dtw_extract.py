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
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
from os import path
sys.path.append(path.join("..", "..", "speech_dtw"))
from speech_dtw import _dtw

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
    
sampled = np.load(Path("../data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()
ss_save_fn = '../support_set/support_set_5.npz'
files = "../../Datasets/spokencoco/SpokenCOCO"
audio_segments_dir = Path(files)
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
s_audio = {}
s_wavs = []

for wav_name in tqdm(support_set):
    wav, img, spkr, start, end, word = support_set[wav_name]
    name = Path(wav.split('.')[0]+'_'+word)
    fn = (Path('../support_set') / name).with_suffix('.wav')
    if word not in s_audio: s_audio[word] = []
    s_audio[word].append(fn)
    s_wavs.append(Path(wav).stem)

train_fn = Path(files) / 'SpokenCOCO_train.json'
with open(train_fn, 'r') as f:
    data = json.load(f)

data = data['data']
query_scores = {}
record = {}

def preemphasis(x,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    return scipy.signal.lfilter([1, -coeff], [1], x)

preemph_coef = 0.97
sample_rate = 16000
window_size = 0.025
window_stride = 0.01
window_type = "hamming" 
num_mel_bins = 40
fmin = 20
n_fft = int(sample_rate * window_size)   
win_length = int(sample_rate * window_size)
hop_length = int(sample_rate * window_stride) 

def load_mel(fn):
    y, sr = librosa.load(fn, sample_rate)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=scipy_windows.get(window_type, scipy_windows['hamming']))
    spec = np.abs(stft)**2
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
    melspec = np.dot(mel_basis, spec)
    logspec = librosa.power_to_db(melspec, ref=np.max)
    return logspec
    # logspec = np.transpose(logspec, (1, 0))
    # return np.ascontiguousarray(logspec)

def dtw_func(q, s):
    D, wp = librosa.sequence.dtw(q, s, subseq=True, metric='euclidean')
    w0 = wp[-1, 0]
    wT = wp[0, 0]

    if len(wp) != 0:
        score = -np.min(D[-1, :] / wp.shape[0])
        return w0, wT, score

for entry in tqdm(data):
    for caption in entry['captions']:
        outputs = {}
        wav = caption['wav']
        if str(wav) not in outputs: outputs[str(wav)] = {}
        wav_name = Path(wav).stem
        if Path(f'dtw/{wav_name}.npz').is_file(): continue
        if wav_name in s_wavs: continue

        fn = (audio_segments_dir / Path(wav)).with_suffix('.wav')
        y = load_mel(fn)#librosa.load(fn)

        for word in s_audio:
            for q_fn in s_audio[word]:

                x = load_mel(q_fn)
                w0, wT, norm_score = dtw_func(x, y)
                outputs[str(wav)][str(q_fn)] = w0, wT, norm_score

        np.savez_compressed(
            Path(f'dtw/{wav_name}'), 
            distances=outputs
            )
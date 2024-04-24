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

K = 5
num_episodes = 1000


save_dir = Path('../support_set')
support_set = np.load(save_dir / Path('support_set.npz'), allow_pickle=True)['support_set'].item()

for name in support_set:
    wav, im, spkr, start, stop, word = support_set[name]
    print(word, name, wav)
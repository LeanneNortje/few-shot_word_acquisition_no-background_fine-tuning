#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
from .util import *
    
def compute_matchmap_similarity_matrix_loss(
    image_outputs, english_output, english_nframes, negatives, positives, attention, contrastive_loss, 
    margin, simtype, alphas, rank):
    
    # loss = 0
    s_anchor = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, attention, simtype)

    neg_aud_s = []
    neg_im_s = []
    base_neg_im_s = []
    for neg_dict in negatives:
        s = compute_matchmap_similarity_matrix_IA(image_outputs, None, neg_dict["english_output"], neg_dict["english_nframes"], attention, simtype)
        neg_aud_s.append(s)
        s = compute_matchmap_similarity_matrix_IA(neg_dict['image'], None, english_output, english_nframes, attention, simtype)
        neg_im_s.append(s)

    neg_aud_s = torch.cat(neg_aud_s, dim=1)
    neg_im_s = torch.cat(neg_im_s, dim=1)

    pos_aud_s = []
    pos_im_s = []
    for pos_dict in positives:
        s = compute_matchmap_similarity_matrix_IA(image_outputs, None, pos_dict["english_output"], pos_dict["english_nframes"], attention, simtype)
        pos_aud_s.append(s)
        s = compute_matchmap_similarity_matrix_IA(pos_dict['image'], None, english_output, english_nframes, attention, simtype)
        pos_im_s.append(s)

    pos_aud_s = torch.cat(pos_aud_s, dim=1)
    pos_im_s = torch.cat(pos_im_s, dim=1)

    # anch = torch.cat([s_anchor, s_anchor], dim=0)
    # positives = torch.cat([pos_aud_s, pos_im_s], dim=0)
    # negatives = torch.cat([neg_aud_s, neg_im_s], dim=0)
    # base_negatives = torch.cat([base_neg_im_s, base_neg_im_s], dim=0)
    # loss = contrastive_loss(anch, positives, negatives, base_negatives) 

    loss = contrastive_loss(s_anchor, pos_aud_s, pos_im_s, neg_aud_s, neg_im_s, None)  

    return loss

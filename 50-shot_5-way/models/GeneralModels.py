#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoringAttentionModule(nn.Module):
    def __init__(self, args):
        super(ScoringAttentionModule, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.audio_encoder = nn.Sequential(
            # nn.LayerNorm(512),
            nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Linear(256, 128),
            # nn.ReLU()
            nn.Linear(128, 1)
        )
        # self.image_encoder = nn.LSTM(49, 44, batch_first=True)
        self.image_encoder = nn.Sequential(
            # # nn.LayerNorm(49),
            # nn.Linear(49, 49),
            # # nn.ReLU(),
            # # nn.LayerNorm(49),
            # nn.Linear(49, 49),
            # # nn.ReLU()
        )
        # self.similarity = nn.Sequential(
        #     # nn.LayerNorm(49),
        #     nn.Linear(49, 49),
        #     nn.ReLU(),
        #     # nn.LayerNorm(49),
        #     nn.Linear(49, 1),
        #     nn.ReLU()
        # )
        self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sig = torch.nn.Sigmoid()

    def normalise(self, x, fr=None):
        # if fr is None:
        #     minimum, _ = x.min(dim=-1)
        #     minimum, _ = minimum.min(dim=-1)
        #     x = x - minimum
        #     maximum, _ = x.max(dim=-1)
        #     maximum, _ = maximum.max(dim=-1)
        #     x = x / maximum
        #     # x = torch.nn.functional.normalize(x)
        # else:
        #     minimum, _ = x[:, :, 0:fr].min(dim=-1)
        #     minimum, _ = minimum.min(dim=-1)
        #     x = x - minimum
        #     maximum, _ = x[:, :, 0:fr].max(dim=-1)
        #     maximum, _ = maximum.max(dim=-1)
        #     x = x / maximum
        #     # x = torch.nn.functional.normalize(x)
        return x

    def forward(self, image_embedding, audio_embeddings, audio_nframes):
        
        image_embedding = self.normalise(image_embedding.transpose(1, 2))
        audio_embeddings = self.normalise(audio_embeddings, audio_nframes)
        
        aud_em = self.audio_encoder(audio_embeddings)
        image_embedding = self.image_encoder(image_embedding)

        att = torch.bmm(aud_em.transpose(1, 2), image_embedding)# / (torch.norm(aud_em, dim=1) * torch.norm(image_embedding, dim=1))
        s, _ = att.max(dim=-1)

        # s = att.mean(dim=-1).unsqueeze(1)

        # # print(att.size())
        # # s, _ = att.max(dim=-1)
        # s = self.similarity(att).unsqueeze(-1)

        # # im_context = self.image_encoder(image_embedding).squeeze(2)
        # # aud_context = self.audio_encoder(audio_embeddings).squeeze(2)
        return s #self.sig(s)

    def score(self, image_embedding, audio_embeddings, audio_nframes):
        
        scores = [] #torch.zeros((audio_embeddings.size(0), image_embedding.size(0)), device=audio_embeddings.device)
        for i in range(image_embedding.size(0)):
            im = self.normalise(image_embedding[i, :, :].unsqueeze(0).transpose(1, 2))
            aud_em = self.normalise(audio_embeddings[i, :, :].unsqueeze(0), audio_nframes[i])
            aud_em = self.audio_encoder(aud_em) #self.audio_encoder(a).squeeze(2)
            im = self.image_encoder(im)
            att = torch.bmm(aud_em.transpose(1, 2), im)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            # s, _ = att.max(dim=-1)
            # s = self.similarity(att).unsqueeze(-1)
            s, _ = att.max(dim=-1)
            # s = att.mean(dim=-1).unsqueeze(1)
            scores.append(s)
        scores = torch.cat(scores, dim=0)
        return scores #self.sig(scores)

    def one_to_many_score(self, image_embedding, audio_embeddings, audio_nframes):
        
        audio_embeddings = self.normalise(audio_embeddings, audio_nframes)
        aud_em = self.audio_encoder(audio_embeddings)#.squeeze(2)

        # image_embedding = self.normalise(image_embedding.transpose(1, 2))
        # image_embedding = self.image_encoder(image_embedding)

        scores = [] #torch.zeros((audio_embeddings.size(0), image_embedding.size(0)), device=audio_embeddings.device)
        for i in range(image_embedding.size(0)):
            im = self.normalise(image_embedding[i, :, :].unsqueeze(0).transpose(1, 2))
            im = self.image_encoder(im)
            att = torch.bmm(aud_em.transpose(1, 2), im)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            # s, _ = att.max(dim=-1)
            # s = self.similarity(att).unsqueeze(-1)
            s, _ = att.max(dim=-1)
            # s = att.mean(dim=-1).unsqueeze(1)
            scores.append(s)
        scores = torch.cat(scores, dim=1)
        return scores #self.sig(scores)
        
class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, anchor, positives_1, positives_2, negatives_1, negatives_2, base_negatives):
        N = anchor.size(0)
        sim = [anchor, positives_1, positives_2, negatives_1, negatives_2]
        if base_negatives is not None: sim.append(base_negatives)
        sim = torch.cat(sim, dim=1)
        labels = []
        labels.append(100*torch.ones((N, anchor.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_1.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_2.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_1.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_2.size(1)), device=anchor.device))
        if base_negatives is not None: labels.append(0*torch.ones((N, base_negatives.size(1)), device=anchor.device))
        labels = torch.cat(labels, dim=1)
        loss = self.criterion(sim, labels)

        # 
        # loss = 0
        # for p in range(positives.size(1)):
        #     sim = [self.cos(positives[:, p, :], anchor.squeeze(1)).unsqueeze(1)]
        #     labels = [torch.ones((anchor.size(0), 1), device=anchor.device)]
        #     for n in range(negatives.size(1)): 
        #         sim.append(self.cos(negatives[:, n, :], anchor.squeeze(1)).unsqueeze(1))
        #         labels.append(-1 * torch.ones((anchor.size(0), 1), device=anchor.device))
        #         # loss += (self.cos(negatives[:, n, :], anchor.squeeze(1)) - self.cos(positives[:, p, :], anchor.squeeze(1)) + 2.0).clamp(min=0).mean()
        #         # print(self.cos(negatives[:, n, :], anchor.squeeze(1)), self.cos(positives[:, p, :], anchor.squeeze(1)))
        #     sim = torch.cat(sim, dim=1)
        #     labels = torch.cat(labels, dim=1)
        #     loss += self.criterion(sim, labels) 

        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Gate_Aspect_model(nn.Module):
    def __init__(self, args) -> None:
        super(CNN_Gate_Aspect_model, self).__init__()
        self.args = args

        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        A = args['aspect_num']

        Co = args['kernel_num']
        Ks = args['kernel_sizes']
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args['embedding'], requires_grad=False)

        self.aspect_embed = nn.Embedding(A, args['aspect_embed_dim'])
        self.aspect_embed.weight = nn.Parameter(args['aspect_embedding'], requires_grad=False)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(args['aspect_embed_dim'], Co, K, padding=K-2) for K in [3]])

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(len(Ks)*Co, C)
        # self.fc_aspect = nn.Linear(args['aspect_embed_dim'], Co)
        self.fc_aspect = nn.Linear(Co, Co)

    def forward(self, feature, aspect):
        feature = self.embed(feature) # (N, L, D)
        aspect_v = self.aspect_embed(aspect) # (N, L', D)
        # aspect_v = aspect_v.sum(1)/aspect_v.size(1) # (N, D)

        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks))
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1) # (N, D)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1] # [(N, Co, L), ...] *len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2] # [(N, Co, L), ...] *len(Ks)
        x = [i*j for i, j in zip(x, y)]

        x0 = [F.max_pool1d(i, i.size(2)).unsqueeze(2) for i in x]
        x0 = [i.view(i.size(0), -1) for i in x0] # [(N, Co)...]*len(Ks)

        x0 = torch.cat(x0, 1)
        logit = self.fc(x0) # (N, C)
        return logit, x, y

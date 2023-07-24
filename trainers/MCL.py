
from torch import nn
import torch
import numpy as np

class MCL(nn.Module):
    def __init__(self):
        super(MCL, self).__init__()
        self.margin = 0.5
    def get_pos_neg(self,reconstructed_log_feats_head,label):
        pos=reconstructed_log_feats_head[label]
        neg_samples=[]
        for _ in range(self.neg_sampling_num):
            item = np.random.choice(len(self.feats))
            while item in label or item in neg_samples:
                item = np.random.choice(len(self.feats))
            neg_samples.append(item)
        neg=self.feats[neg_samples]
        return pos,neg

    def forward(self, log_feats_head,reconstructed_log_feats_head,label):
        epsilon = 1e-5
        loss = list()
        n = log_feats_head.shape[0] 
        sim_mat = torch.matmul(log_feats_head, reconstructed_log_feats_head.t()) 

        label_matrix = label.unsqueeze(0)==label.unsqueeze(1)

        pos_pair = label_matrix
        neg_pair = ~label_matrix
        pos_sim_mat = sim_mat*pos_pair
        pos_sim_mat = torch.masked_select(sim_mat*pos_pair, sim_mat*pos_pair < 1 - epsilon)
        neg_sim_mat = torch.masked_select(sim_mat*neg_pair, sim_mat*neg_pair > self.margin)
 
        pos_sim_mat = -pos_sim_mat
        pos_sim_mat[pos_sim_mat!=0]+=1
        pos_loss = torch.sum(pos_sim_mat)

        neg_loss = torch.sum(neg_sim_mat)
        loss.append(pos_loss + neg_loss)
        loss = sum(loss) / n
        return loss

import numpy as np
import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
 
class SASRec(torch.nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.dev = args.device
        self.hidden_units = args.dcair_hidden_units
        self.max_len = args.dcair_max_len
        self.dropout_rate = args.dcair_dropout
        self.num_heads =  args.dcair_num_heads
        self.num_blocks = args.dcair_num_blocks_sas
        self.context_window = args.context_window
        self.dataset = args.dataset_code
        self.item_emb = torch.nn.Embedding(self.item_num+1, self.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.max_len, self.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_units,
                                                            self.num_heads,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def circulant(self,n, window):
        circulant_t = torch.zeros(n,n, device=self.dev)
        offsets = [-i for i in range(0,window)]
        for offset in offsets:
            circulant_t.diagonal(offset=offset).copy_(torch.ones(n-abs(offset), device=self.dev))
        circulant_t = ~circulant_t.bool()
        return circulant_t

    def forward(self, timeline_mask, seqs, attention_mode):
        seqs *= ~timeline_mask.unsqueeze(-1) 
        tl = seqs.shape[1] 
        if attention_mode == 'individual':
            attention_mask = torch.diagonal(torch.ones((tl, tl), device=self.dev))
            attention_mask = ~torch.diagflat(attention_mask).bool()
            attention_mask = attention_mask.to(self.dev)
        elif attention_mode == 'new_window':
            attention_mask = self.circulant(tl, self.context_window)
        elif attention_mode == 'window':
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        elif attention_mode == 'aggregate':
            attention_mask = (torch.zeros((tl, tl), device=self.dev)).bool()
        else:
            attention_mask = ~torch.diag(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
            
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, weights = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) 
        return log_feats



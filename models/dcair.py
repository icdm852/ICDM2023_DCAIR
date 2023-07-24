from .base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import pickle
import os
import math


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
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
  
        outputs += inputs
        return outputs

class DCAIR(BaseModel):

    def __init__(self,SASRec, args):
        super().__init__(args)
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.dev = args.device
        self.hidden_units = args.dcair_hidden_units
        self.max_len = args.dcair_max_len
        self.dropout_rate = args.dcair_dropout
        self.num_heads =  args.dcair_num_heads
        self.num_blocks = args.dcair_num_blocks
        self.num_blocks_aggr = args.dcair_num_blocks_aggr
        self.dataset = args.dataset_code
        P5_path = 'P5_our/'
        pretrained_path = P5_path+'data/pre_trained_bert_data/'+self.dataset+'/'
        pretrained_file = open(pretrained_path+self.dataset+'_pretrained_P5_item_id_vecs.pkl','rb')
        pretrained_file_data = pickle.load(pretrained_file)
        weights = torch.FloatTensor(pretrained_file_data)
        pretrained_file.close()
        self.pretrained_item_embedding = torch.nn.Embedding.from_pretrained(weights)

        self.pretrained_emb_dim = weights.shape[-1]
        self.attribute_encoder = SASRec(args)
        self.context_encoder = SASRec(args)

        self.fc = torch.nn.Linear(self.hidden_units, self.item_num+1)
        self.reconstruct_item_layer = torch.nn.Sequential(
                                                            torch.nn.Linear(self.hidden_units*2, self.hidden_units),
                                                            torch.nn.Sigmoid(),
                                                            torch.nn.Linear(self.hidden_units, self.hidden_units),
                                                            torch.nn.Sigmoid()
                                                            )
        self.context_aggegator = SASRec(args)
        self.seq_para_gate = nn.Parameter(torch.zeros(1,self.hidden_units))
        self.ctxt_para_gate = nn.Parameter(torch.zeros(1,self.hidden_units))
        self.attr_para_gate = nn.Parameter(torch.zeros(1,self.hidden_units))
        self.cstr_para_gate = nn.Parameter(torch.zeros(1,self.hidden_units))
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        self.hidden_units = self.hidden_units
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

    @classmethod
    def code(cls):
        return 'dcair'

    def log2feats(self, timeline_mask,seqs):
        seqs = seqs * (~timeline_mask.unsqueeze(-1)) 
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, weights = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))
        log_feats = self.last_layernorm(seqs) 
        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs, seq_context): 
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb(log_seqs)
        seqs = seqs+self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
  
        batch_size = log_seqs.shape[0]
        sequence_len = log_seqs.shape[1]
        
        log_feats = self.log2feats(timeline_mask,seqs) 
        log_seqs_ctx = seq_context

        positions_ctx = np.tile(np.array(range(log_seqs_ctx.shape[1])), [log_seqs_ctx.shape[0], 1])
        log_seqs_ctx = torch.LongTensor(log_seqs_ctx).to(self.dev)

        seqs_ctx = self.item_emb(log_seqs_ctx)
        seqs_ctx += self.pos_emb(torch.LongTensor(positions_ctx).to(self.dev))
        seqs_ctx = self.emb_dropout(seqs_ctx)

        log_feats_context = self.context_encoder(timeline_mask,seqs_ctx, attention_mode = 'new_window')
        ctx_mask = torch.tril(torch.ones((log_seqs_ctx.shape[-1], log_seqs_ctx.shape[-1]), dtype=torch.bool, device=self.dev))
        log_seqs_ctx_new = log_seqs_ctx.repeat(1, log_seqs_ctx.shape[-1])
        log_seqs_ctx_new = log_seqs_ctx_new.reshape(-1,log_seqs_ctx.shape[-1], log_seqs_ctx.shape[-1])
        log_seqs_ctx_new = log_seqs_ctx_new * ctx_mask
        log_seqs_ctx_new = log_seqs_ctx_new.reshape(-1, log_seqs_ctx_new.shape[-1])

        pretrained_feats_context = self.pretrained_item_embedding(log_seqs_ctx_new)
        zero = torch.zeros_like(log_seqs_ctx_new).to(self.dev)
        one  = torch.ones_like(log_seqs_ctx_new).to(self.dev)
        context_timeline_mask= torch.where(log_seqs_ctx_new > 0, zero, one).bool()

        context_positions = np.tile(np.array(range(log_seqs_ctx_new.shape[1])), [log_seqs_ctx_new.shape[0], 1])
        pretrained_feats_context += self.pos_emb(torch.LongTensor(context_positions).to(self.dev))
        pretrained_feats_context = self.emb_dropout(pretrained_feats_context)

        pretrained_feats_context = self.context_aggegator(context_timeline_mask,pretrained_feats_context, attention_mode = 'aggregate')
        pretrained_feats_context = pretrained_feats_context.reshape(batch_size,sequence_len,-1,self.hidden_units)
        pretrained_feats_context = pretrained_feats_context.mean(dim=2)
        seqs_attr = seqs
        log_feats_attribute = self.attribute_encoder(timeline_mask,seqs_attr, attention_mode = 'individual')
        pretrained_feats_attribute = self.pretrained_item_embedding(log_seqs)
        reconstructed_log_feats = self.reconstruct_item_layer(torch.cat((log_feats_attribute,pretrained_feats_context),dim=-1)) 
        log_feats_new  = self.seq_para_gate*log_feats + self.ctxt_para_gate*log_feats_context + self.attr_para_gate*log_feats_attribute + self.cstr_para_gate*reconstructed_log_feats
        log_pred = self.fc(log_feats_new)
        return log_pred, log_feats, log_feats_attribute, log_feats_context, pretrained_feats_attribute, pretrained_feats_context,reconstructed_log_feats

    def predict(self, log_seqs, item_indices, seq_context): 
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        log_feats = self.log2feats(timeline_mask,seqs) 
        log_feats = log_feats[:, -1, :]

        context_timeline_mask = torch.BoolTensor(seq_context == 0).to(self.dev)  
        positions_ctx = np.tile(np.array(range(seq_context.shape[1])), [seq_context.shape[0], 1])
        seq_context = torch.LongTensor(seq_context).to(self.dev)

        seqs_ctx = self.item_emb(seq_context)
        seqs_ctx += self.pos_emb(torch.LongTensor(positions_ctx).to(self.dev))
        seqs_ctx = self.emb_dropout(seqs_ctx)

        log_feats_context = self.context_encoder(timeline_mask,seqs_ctx, attention_mode = 'new_window')
        log_feats_context = log_feats_context[:, -1, :] 

        pretrained_feats_context = self.pretrained_item_embedding(seq_context) 
        context_positions = np.tile(np.array(range(seq_context.shape[1])), [seq_context.shape[0], 1])
        pretrained_feats_context += self.pos_emb(torch.LongTensor(context_positions).to(self.dev))
        pretrained_feats_context = self.emb_dropout(pretrained_feats_context)

        pretrained_feats_context = self.context_aggegator(context_timeline_mask,pretrained_feats_context, attention_mode = 'aggregate')   
        pretrained_feats_context = pretrained_feats_context.mean(dim=1)
        seqs_attr = seqs
        log_feats_attribute = self.attribute_encoder(timeline_mask,seqs_attr, attention_mode = 'individual') 
        log_feats_attribute = log_feats_attribute[:, -1, :] 

        reconstructed_log_feats = self.reconstruct_item_layer(torch.cat((log_feats_attribute,pretrained_feats_context),dim=-1)) 
        log_feats_new  = self.seq_para_gate*log_feats + self.ctxt_para_gate*log_feats_context + self.attr_para_gate*log_feats_attribute + self.cstr_para_gate*reconstructed_log_feats
        log_pred = self.fc(log_feats_new)
        return log_pred
   
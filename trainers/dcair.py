from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch.nn as nn
import numpy as np
import torch
from .MCL import MCL

class SASRecTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.bce_criterion = nn.BCEWithLogitsLoss() 
        self.device = self.device
        self.num_epochs = args.num_epochs
        self.MCL = MCL()

    @classmethod
    def code(cls):
        return 'dcair'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def orthogonal_loss(self, x, y):
        x_norm = torch.norm(x.clone(), 2, -1, keepdim=True)
        y_norm = torch.norm(y.clone(), 2, -1, keepdim=True)
        xy = torch.mm(x, y.t())
        xy = torch.norm(xy.clone(), 2, -1, keepdim=True) 
        xy = torch.pow(xy,2)
        loss = torch.mean(xy/(x_norm*y_norm))
        return loss
    
    def calculate_loss(self, batch):
        seq, pos, neg, context_mask, head_mask, seq_context = batch
        seq, pos, neg, context_mask, head_mask, seq_context = np.array(seq), np.array(pos), np.array(neg), np.array(context_mask), np.array(head_mask), np.array(seq_context)
        
        pos = torch.LongTensor(pos).to(self.device)
        y_pred, log_feats, log_feats_attribute, log_feats_context,pretrained_feats_attribute, pretrained_feats_context, reconstructed_log_feats = self.model( seq, pos, neg,seq_context)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        pos = pos.view(-1)
        bceloss = self.ce(y_pred,pos) 

        log_feats_head = log_feats.reshape(-1,log_feats.shape[-1]) 
        head_mask = torch.LongTensor(head_mask).to(self.device)
        head_mask = head_mask.view(-1)
        head_mask = head_mask==1
        log_feats_head= log_feats_head[head_mask]
        reconstructed_log_feats_head = reconstructed_log_feats.reshape(-1,reconstructed_log_feats.shape[-1]) 
        construted_label = torch.LongTensor(seq).to(self.device).view(-1)[head_mask]
        reconstructed_log_feats_head = reconstructed_log_feats_head[head_mask]
        construted_loss = self.MCL(log_feats_head,reconstructed_log_feats_head,construted_label)
        
        context_mask = torch.LongTensor(context_mask).to(self.device).view(-1)
        context_mask = context_mask!=0
        seq_label_context = torch.LongTensor(seq).to(self.device).view(-1)[context_mask]
        log_feats_context = log_feats_context.reshape(-1,log_feats_context.shape[-1]) 
        pretrained_feats_context = pretrained_feats_context.reshape(-1,pretrained_feats_context.shape[-1]) 
        log_feats_context_new = log_feats_context[context_mask]  
        pretrained_feats_context = pretrained_feats_context[context_mask] 
        context_loss = self.MCL(log_feats_context_new,pretrained_feats_context,seq_label_context)
        
        all_items_mask = torch.LongTensor(seq).to(self.device).view(-1)
        all_items_mask  = all_items_mask!=0
        seq_lable = torch.LongTensor(seq).to(self.device).view(-1)[all_items_mask]
        log_feats_attribute = log_feats_attribute.reshape(-1,log_feats_attribute.shape[-1]) 
        pretrained_feats_attribute = pretrained_feats_attribute.reshape(-1,pretrained_feats_attribute.shape[-1]) 
        log_feats_attribute = log_feats_attribute[all_items_mask]
        pretrained_feats_attribute = pretrained_feats_attribute[all_items_mask]  
        attribute_loss = self.MCL(log_feats_attribute,pretrained_feats_attribute,seq_lable)
        loss = bceloss + construted_loss / (construted_loss / bceloss).detach() + attribute_loss / (attribute_loss / bceloss).detach()+ context_loss / (context_loss / bceloss).detach() 
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels, seq_context = batch
        seqs = np.array(seqs)
        candidates = np.array(candidates)
        seq_context = np.array(seq_context)
        scores = self.model.predict(seqs, candidates, seq_context)
        candidates = torch.LongTensor(candidates).to(self.device)
        labels  = labels.to(scores.device)
        scores = scores.gather(1, candidates)
        metrics_out = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics_out 
    
   

    
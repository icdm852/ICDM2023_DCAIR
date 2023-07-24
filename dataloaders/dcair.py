from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils
from collections import Counter
import random 
import numpy as np

GLOBAL_SEED = 1
from utils import fix_random_seed_as
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

class DCAIRDataloader(AbstractDataloader):

    def __init__(self, args, dataset):
        
        super().__init__(args, dataset)
        set_seed(1)
        fix_random_seed_as(1)
        args.num_items = len(self.smap)
        args.num_users = len(self.umap)
        self.K = args.context_window
        
        self.device = args.device
        self.max_len = args.dcair_max_len

        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder,mode = 'test')
        val_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder,mode = 'val')                                                

        self.test_negative_samples = test_negative_sampler.get_negative_samples()
        self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.train_len_sum = [len(v) for k,v in self.train.items()]

        self.train_sum = [i for k,v in  self.train.items() for i in v]
        self.val_sum = [i for k,v in  self.val.items() for i in v]
        self.test_sum = [i for k,v in  self.test.items() for i in v]
        

        self.train_sum_dic = dict(Counter(self.train_sum))
        self.total_data_sum_dic = dict(Counter(self.train_sum+ self.val_sum + self.test_sum))
        self.data_loader_num = args.data_loader_num
        args.tail_num = len(self.total_data_sum_dic) - args.head_num

        self.head_class_id = [i[0] for i in Counter(self.train_sum).most_common(args.head_num)]

        self.data_item2bilabel_dic ={}
        self.data_item2bilabel_list =[]
        for i in self.total_data_sum_dic:
            if i in self.head_class_id:
                self.data_item2bilabel_dic[i] = 1
                self.data_item2bilabel_list.append(1)
            else:
                self.data_item2bilabel_dic[i] = 0
                self.data_item2bilabel_list.append(0)

        self.data_item2bilabel_list = [-1] + self.data_item2bilabel_list

    @classmethod
    def code(cls):
        return 'dcair'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader,val_loader_head,val_loader_tail = self._get_val_loader()
        test_loader,test_loader_head,test_loader_tail = self._get_test_loader()
        
        val_loader_list = val_loader,val_loader_head,val_loader_tail
        test_loader_list = test_loader,test_loader_head,test_loader_tail
        return train_loader, val_loader_list, test_loader_list 
    
    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = DCAIRTrainDataset(self.train, self.max_len, self.item_count, self.head_class_id, self.K)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')
    
    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset,dataset_head,dataset_tail = self._get_eval_dataset(mode)
        dataloader_head = data_utils.DataLoader(dataset_head, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        dataloader_tail = data_utils.DataLoader(dataset_tail, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        return dataloader,dataloader_head,dataloader_tail

    def _get_eval_dataset(self, mode):
        if mode=='test':
            self.test = {k:v for k,v in self.test.items() if v[0] in self.train_sum_dic}
            answers_head = {k:v for k,v in self.test.items() if v[0] in self.head_class_id}
            answers_tail = {k:v for k,v in self.test.items() if v[0] not in self.head_class_id}
            answers = self.test
            dataset_head = DCAIREvalDataset(self.train, answers_head, self.max_len, self.test_negative_samples, self.K)
            dataset_tail = DCAIREvalDataset(self.train, answers_tail, self.max_len, self.test_negative_samples, self.K)
            dataset = DCAIREvalDataset(self.train, answers, self.max_len, self.test_negative_samples, self.K)
        else:
         
            self.val = {k:v for k,v in self.val.items() if v[0] in self.train_sum_dic}
            answers_head = {k:v for k,v in self.val.items() if v[0] in self.head_class_id}
            answers_tail = {k:v for k,v in self.val.items() if v[0] not in self.head_class_id}
            answers = self.val

            dataset_head = DCAIREvalDataset(self.train, answers_head, self.max_len, self.val_negative_samples, self.K)
            dataset_tail = DCAIREvalDataset(self.train, answers_tail, self.max_len, self.val_negative_samples, self.K)
            dataset = DCAIREvalDataset(self.train, answers, self.max_len, self.val_negative_samples, self.K)
        return dataset,dataset_head,dataset_tail


class DCAIRTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, num_items, head_class_id, K):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.num_items = num_items
        self.K = K
        self.head_class_id = head_class_id 

    def __len__(self):
        return len(self.users)
    def random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t
    def __getitem__(self, index):
        user = self.users[index]
        item_seq = self._getseq(user)
        if int(len(item_seq))<= 1:
            index=random.randint(0,self.__len__()-1)
            return self.__getitem__(index)
        seq = np.zeros([self.max_len], dtype=np.int32)
        seq_context = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        head_mask = np.zeros([self.max_len], dtype=np.int32) 

        nxt = item_seq[-1]
        idx = self.max_len - 1
        ts = set(item_seq)
    
        for item_idx, cur_item in enumerate(reversed(item_seq[:-1])):
            
            seq[idx] = cur_item
            pos[idx] = nxt
            if cur_item in self.head_class_id:
                head_mask[idx] = 1
            if nxt != 0: neg[idx] = self.random_neq(1, self.num_items + 1, ts)
            nxt = cur_item
            idx -= 1
            if idx == -1: break
        seq_context = np.copy(seq)
        seq_context = np.insert(seq_context, 0,[0])
        seq_context = np.delete(seq_context,[-1])
        context_mask = (seq_context>0).astype(int)
        return (seq, pos, neg, context_mask,head_mask, seq_context)

    def _getseq(self, user):
        return self.u2seq[user]

class DCAIREvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples, K):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples
        self.K = K

    def __len__(self):
        return len(self.users)
    def __getitem__(self, index):
        user = self.users[index]  
        if user not in self.u2answer:
            index=random.randint(1,len(self.users)-1)
            return self.__getitem__(index)
        item_seq = self.u2seq[user]
        seq = np.zeros([self.max_len], dtype=np.int32)

        idx = self.max_len - 1
        for i in reversed(item_seq):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(item_seq)
        rated.add(0)
        answer = self.u2answer[user] 
        item_idx = [answer[0]] 
        negs = self.negative_samples[user]
        item_idx = item_idx + negs 
        item_idx = np.array(item_idx)
        labels = [1] * 1 + [0] * (len(item_idx)-1)
        labels = np.array(labels)
        labels = torch.LongTensor(labels)
        seq_context = np.copy(seq)
        seq_context = np.insert(seq_context, 0,[0])
        seq_context = np.delete(seq_context,[-1])

        return seq, item_idx, labels, seq_context



from templates import set_template
from datasets import DATASETS
from dataloaders.__init__ import DATALOADERS
from models.__init__ import MODELS
from trainers.__init__ import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='beauty', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=10, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500, 
                    help='Size of val and test set')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random', 'all_in'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random', 'all_in'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=1000)
parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='31')
parser.add_argument('--use_cuda',action='store_false')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=25, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='Recall@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=0)

# SASRec #
parser.add_argument('--sas_max_len', type=int, default=50, help='Length of sequence for bert')
parser.add_argument('--sas_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--sas_hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--sas_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--sas_num_heads', type=int, default=1, help='Number of heads for multi-attention')
parser.add_argument('--sas_dropout', type=float, default=0.5, help='Dropout probability to use throughout the model')

# DCAIR #
parser.add_argument('--dcair_max_len', type=int, default=50, help='Length of sequence for bert')
parser.add_argument('--dcair_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--dcair_hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--dcair_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--dcair_num_heads', type=int, default=1, help='Number of heads for multi-attention')
parser.add_argument('--dcair_dropout', type=float, default=0.5, help='Dropout probability to use throughout the model')
parser.add_argument('--dcair_num_blocks_aggr', default=2, type=int)
parser.add_argument('--dcair_num_blocks_constr', default=2, type=int)
parser.add_argument('--dcair_num_blocks_sas', default=2, type=int)
parser.add_argument('--context_window', default=5, type=int,help='the window size of pretrained context')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')


################
# MASR
################
parser.add_argument('--head_num',type=int,default=1000,help='we divide the head and tail class based on the numbers of head items and tail items, the ratio is 80:20')
parser.add_argument('--head_num_code', type=str, default='num_80') 
parser.add_argument('--data_loader_num',type=int,default=32, help='take 1/k data as new data')
parser.add_argument('--batch',type=int,default=64, help='the batch num start use memory bank')

################
args = parser.parse_args()
set_template(args)

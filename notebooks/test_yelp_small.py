#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
from packaging import version
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from src.param import parse_args
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from src.trainer_base import TrainerBase

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


# In[2]:


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
args = DotDict()

args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = "yelp"
args.valid = "yelp"
args.test = "yelp"
args.batch_size = 16
args.optim = 'adamw' 
args.warmup_ratio = 0.02
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.losses = 'rating,sequential,explanation,review,traditional'
args.backbone = 't5-small' # small or base
args.output = 'snap/yelp-small'
args.epoch = 10
args.local_rank = 0

args.comment = ''
args.train_topk = -1
args.valid_topk = -1
args.dropout = 0.1

args.tokenizer = 'p5'
args.max_text_length = 512
args.do_lower_case = False
args.word_mask_rate = 0.15
args.gen_max_length = 64

args.weight_decay = 0.01
args.adam_eps = 1e-6
args.gradient_accumulation_steps = 1

'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

'''
Whole word embedding
'''
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss') # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 1 #args.local_rank
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

comments = []
dsets = []
if 'yelp' in args.train:
    dsets.append('yelp')
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)

if args.local_rank in [0, -1]:
    print(args)


# In[3]:


def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone
    
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)
    
    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model


# In[4]:


config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone
    
tokenizer = create_tokenizer(args)

model_class = P5Pretraining
model = create_model(model_class, config)

model = model.cuda()

if 'p5' in args.tokenizer:
    model.resize_token_embeddings(tokenizer.vocab_size)
    
model.tokenizer = tokenizer


# #### Load Model

# In[5]:


args.load = "../snap/yelp-small.pth"

# Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)

ckpt_path = args.load
load_checkpoint(ckpt_path)

from src.all_yelp_templates import all_tasks as task_templates


# #### Check Test Split

# In[6]:


data_splits = load_pickle('../data/yelp/rating_splits_augmented.pkl')
test_review_data = data_splits['test']


# In[7]:


print("len(test_review_data):", len(test_review_data))


# In[8]:


print("test_review_data[0]:", test_review_data[0])


# In[9]:


data_maps = load_json(os.path.join('../data', 'yelp', 'datamaps.json'))
print(len(data_maps['user2id'])) # number of users
print(len(data_maps['item2id'])) # number of items


# ### Test P5

# In[10]:


from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all


# #### Evaluation - Rating

# In[11]:


test_task_list = {'rating': ['1-10'] # or '1-6'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if p in [str(i/10.0) for i in list(range(10, 50))]]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))


# In[12]:


test_task_list = {'rating': ['1-6'] # or '1-10'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if p in [str(i/10.0) for i in list(range(10, 50))]]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))


# #### Evaluation - Sequential

# In[13]:


test_task_list = {'sequential': ['2-13'] # or '2-3'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                max_length=50, 
                num_beams=20,
                no_repeat_ngram_size=0, 
                num_return_sequences=20,
                early_stopping=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict
    
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)


# In[14]:


test_task_list = {'sequential': ['2-3'] # or '2-13'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                max_length=50, 
                num_beams=20,
                no_repeat_ngram_size=0, 
                num_return_sequences=20,
                early_stopping=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict
    
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)


# #### Evaluation - Explanation

# In[15]:


test_task_list = {'explanation': ['3-10'] # or '3-7' or '3-2'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=10,
                num_beams=12,
                num_return_sequences=1,
                num_beam_groups=3
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))


# In[16]:


test_task_list = {'explanation': ['3-7'] # or '3-10' or '3-2'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=10,
                num_beams=12,
                num_return_sequences=1,
                num_beam_groups=3
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))


# In[17]:


test_task_list = {'explanation': ['3-2']
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=8
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))


# #### Evaluation - Review

# In[18]:


test_task_list = {'review': ['4-3'] # or '4-2'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i > 50:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))


# In[19]:


test_task_list = {'review': ['4-2'] # or '4-3'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i > 50:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))


# #### Evaluation - Traditional

# In[20]:


test_task_list = {'traditional': ['5-8']  # or '5-5'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                max_length=50, 
                num_beams=20,
                no_repeat_ngram_size=0, 
                num_return_sequences=20,
                early_stopping=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict
    
evaluate_all(ui_scores, gt, 1)
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)


# In[21]:


test_task_list = {'traditional': ['5-5']  # or '5-8'
}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                max_length=50, 
                num_beams=20,
                no_repeat_ngram_size=0, 
                num_return_sequences=20,
                early_stopping=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict
    
evaluate_all(ui_scores, gt, 1)
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)


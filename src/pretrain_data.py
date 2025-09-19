from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

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

    
class P5_Amazon_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        if self.mode == 'train':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['train']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['train']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['train']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'val':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['val']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['val']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'test':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
            self.zeroshot_exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', 'beauty', 'zeroshot_exp_splits.pkl')) # change to dataset to be transferred (e.g., 'beauty')
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('/scratch/jpa2742/P5/data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('/scratch/jpa2742/P5/data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('/scratch/jpa2742/P5/data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'user_id2name.pkl'))
        
        self.meta_data = []
        for meta in parse(os.path.join('/scratch/jpa2742/P5/data', split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                # The first group of sequential prompts (directly predict next item): 2-1 to 2-6 and 2-13
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of sequential prompts (predict next item from a candidate list): 2-7 to 2-10
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                # The third group of sequential prompts (predict yes or no for each user-item pair): 2-11 to 2-12
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                # The first group of direct recommendation prompts (predict yes or no for each user-item pair): 5-1 to 5-4
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of direct recommendation prompts (choose one item from 100 candidates): 5-5 to 5-8
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'zeroshot':
                if sum([0 < int(ind.split('-')[1]) <= 7 for ind in self.task_list[key]]):
                    self.total_length += len(self.zeroshot_exp_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
            else:
                raise NotImplementedError
    
    # use Gaussian sampling to augment rating scores
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'rating':            
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'
            
            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-2':
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], title) 
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], int(rating_datum['overall']))
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], overall_candidates[overall_idx])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-4':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-5':
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-6':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-7':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-8':
                rand_prob = random.random()
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, int(rating_datum['overall']), title)
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(user_desc, overall_candidates[overall_idx], title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-9':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-10':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            else:
                raise NotImplementedError
            
        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1] # sample a history sequence from the full user purchase history
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] == '2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-5':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-6':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-7' or task_template['id'] == '2-9':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-8' or task_template['id'] == '2-10':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 99
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-11':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-12':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
        
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            
            if task_template['id'] == '3-1':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-2':
                source_text = task_template['source'].format(exp_datum['summary'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']]) 
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-3':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-4':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-5':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['summary'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-6':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-7':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-8':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-9':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-10':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-11':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-12':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            else:
                raise NotImplementedError
                
        elif task_name == 'review': 
            review_datum = self.review_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['review'][task_candidates[task_idx]]
            assert task_template['task'] == 'review'
            
            if task_template['id'] == '4-1':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(review_datum['summary'])
            elif task_template['id'] == '4-2':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-3':
                if 'reviewerName' in review_datum:
                    user_desc = review_datum['reviewerName']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(review_datum['summary'])
            elif task_template['id'] == '4-4':
                if 'reviewerName' in review_datum:
                    user_desc = review_datum['reviewerName']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            else:
                raise NotImplementedError
            
        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1) # random choose the target index for target_candidates
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-5' or task_template['id'] == '5-6':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '5-7' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
                
        elif task_name == 'zeroshot' and self.mode == 'test':
            zeroshot_exp_datum = self.zeroshot_exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['zeroshot'][task_candidates[task_idx]]
            assert task_template['task'] == 'zeroshot'
            
            if task_template['id'] == 'Z-1':
                source_text = task_template['source'].format(self.user2id[zeroshot_exp_datum['reviewerID']], zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['brand'], zeroshot_exp_datum['price'])
                if int(zeroshot_exp_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == 'Z-2':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(zeroshot_exp_datum['item_title'], zeroshot_exp_datum['brand'], \
                                                             zeroshot_exp_datum['price'], user_desc)
                target_text = task_template['target'].format(int(zeroshot_exp_datum['overall']))
            elif task_template['id'] == 'Z-3':
                source_text = task_template['source'].format(self.user2id[zeroshot_exp_datum['reviewerID']], zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                target_text = task_template['target'].format(int(zeroshot_exp_datum['overall']))
            elif task_template['id'] == 'Z-4':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                if int(zeroshot_exp_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == 'Z-5':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['brand'], zeroshot_exp_datum['price'])
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            elif task_template['id'] == 'Z-6':
                source_text = task_template['source'].format(zeroshot_exp_datum['feature'], self.user2id[zeroshot_exp_datum['reviewerID']], \
                                                             int(zeroshot_exp_datum['overall']), zeroshot_exp_datum['item_title'], \
                                                             zeroshot_exp_datum['price'], zeroshot_exp_datum['brand'])
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            elif task_template['id'] == 'Z-7':
                if 'reviewerName' in zeroshot_exp_datum:
                    user_desc = zeroshot_exp_datum['reviewerName']
                else:
                    user_desc = zeroshot_exp_datum['reviewerID']
                source_text = task_template['source'].format(zeroshot_exp_datum['item_title'], user_desc)
                target_text = task_template['target'].format(zeroshot_exp_datum['explanation'])
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry

    
class P5_Yelp_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='yelp', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        if self.mode == 'train':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['train']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['train']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['train']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'val':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['val']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['val']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'test':
            self.review_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('/scratch/jpa2742/P5/data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('/scratch/jpa2742/P5/data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('/scratch/jpa2742/P5/data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'user_id2name.pkl'))
            
        self.meta_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'meta_data.pkl'))
        self.user_data = load_pickle(os.path.join('/scratch/jpa2742/P5/data', split, 'user_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
        self.user_meta_dict = {}
        for j, user_meta_item in enumerate(self.user_data):
            self.user_meta_dict[user_meta_item['user_id']] = j
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            else:
                raise NotImplementedError
    
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'rating':            
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'
            
            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-2':
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], title) 
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], int(rating_datum['overall']))
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], overall_candidates[overall_idx])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-4':
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']])
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-5':
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[rating_datum['reviewerID']], self.item2id[rating_datum['asin']], title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-6':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[rating_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-7':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            elif task_template['id'] == '1-8':
                rand_prob = random.random()
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, int(rating_datum['overall']), title)
                    target_text = task_template['target'].format('yes')
                else:
                    overall_candidates = [_ for _ in range(0+1, 5+1) if _ != int(rating_datum['overall'])]
                    overall_idx = random.randint(0, len(overall_candidates)-1) # random choose the overall index for overall_candidates
                    source_text = task_template['source'].format(user_desc, overall_candidates[overall_idx], title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-9':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                if int(rating_datum['overall']) >= 4:
                    target_text = task_template['target'].format('like')
                else:
                    target_text = task_template['target'].format('dislike')
            elif task_template['id'] == '1-10':
                if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
            else:
                raise NotImplementedError
            
        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            history_limit = 20
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            if len(purchase_history) > history_limit:
                purchase_history = purchase_history[-history_limit:]
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] == '2-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-5':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-6':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-7' or task_template['id'] == '2-9':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-8' or task_template['id'] == '2-10':
                if self.mode in ['train', 'val']:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = random.randint(79, 99)
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                elif self.mode == 'test':
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                else:
                    raise NotImplementedError
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history), ' , '.join(candidate_samples))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history), ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '2-11':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-12':
                symbol_prob = random.random()
                if symbol_prob > 0.5:
                    symbol = ' , '
                else:
                    symbol = ' -> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.join(purchase_history), candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
        
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            
            if task_template['id'] == '3-1':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-2':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-3':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-4':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-5':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-6':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
            elif task_template['id'] == '3-7':
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-8':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                if 'name' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['name']
                else:
                    title = 'unknown name'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title)
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-9':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            elif task_template['id'] == '3-10':
                if 'name' in self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[exp_datum['reviewerID']]]['name']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
            else:
                raise NotImplementedError
                
        elif task_name == 'review':
            review_datum = self.review_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['review'][task_candidates[task_idx]]
            assert task_template['task'] == 'review'
            
            if task_template['id'] == '4-1':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-2':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            elif task_template['id'] == '4-3':
                if 'name' in self.user_data[self.user_meta_dict[review_datum['reviewerID']]]:
                    user_desc = self.user_data[self.user_meta_dict[review_datum['reviewerID']]]['name']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
            else:
                raise NotImplementedError
            
        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1) # random choose the target index for target_candidates
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(candidate_samples[0], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-5' or task_template['id'] == '5-6':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99 # random.randint(19, 99)
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            elif task_template['id'] == '5-7' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99 # random.randint(19, 99)
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                target_text = task_template['target'].format(target_item)
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] # [0] for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry
    

class P5_ML1M_Dataset(Dataset):
    def __init__(self, all_tasks_for_ml1m, task_list, tokenizer, args, sample_numbers, mode='train', split='ml1m', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks_for_ml1m # This will be all_ml1m_templates.all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split_name = split # e.g., 'ml1m_p5_processed'
        self.rating_augment = rating_augment # ML1M doesn't have explicit ratings that need augmentation like text-based ones
        self.sample_type = sample_type # For negative sampling in sequential/traditional tasks
        
        data_root = os.path.join('/scratch/jpa2742/P5/data', self.split_name)
        print(f'Data sources root: {data_root}')
        self.mode = mode

        # Load data based on mode
        if self.mode == 'train':
            self.rating_data = load_pickle(os.path.join(data_root, 'review_splits.pkl'))['train']
            # ML1M doesn't have explicit explanations, so exp_data might be empty or not used for rating task
            self.exp_data = load_pickle(os.path.join(data_root, 'exp_splits.pkl'))['train'] 
        elif self.mode == 'val':
            self.rating_data = load_pickle(os.path.join(data_root, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join(data_root, 'exp_splits.pkl'))['val']
        elif self.mode == 'test':
            self.rating_data = load_pickle(os.path.join(data_root, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join(data_root, 'exp_splits.pkl'))['test']
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented for ML1M dataset")
            
        # Sequential data (UserID ItemID1 ItemID2 ...)
        self.sequential_data = ReadLineFromFile(os.path.join(data_root, 'sequential_data.txt'))
        
        # For negative sampling if sequential/traditional tasks are used
        item_count = defaultdict(int)
        self.user_items_for_neg_sampling = defaultdict(list) # Stores mapped item IDs per mapped user ID
        for line in self.sequential_data:
            parts = line.strip().split(' ', 1)
            user = parts[0]
            if len(parts) > 1:
                items = parts[1].split(' ')
                items = [item_id for item_id in items if item_id] # Filter out empty strings if any
                self.user_items_for_neg_sampling[user].extend(items)
                for item in items:
                    item_count[item] += 1
            
        self.all_item_ids_for_neg_sampling = list(item_count.keys())
        if not self.all_item_ids_for_neg_sampling: # Handle case where sequential_data might be empty
             print("Warning: No items found for negative sampling. all_item_ids_for_neg_sampling is empty.")
             self.probability_for_neg_sampling = []
        else:
            counts_for_prob = list(item_count.values())
            sum_value = np.sum(counts_for_prob)
            self.probability_for_neg_sampling = [value / sum_value for value in counts_for_prob] if sum_value > 0 else []

        if self.mode == 'test':
            self.negative_samples_file_content = ReadLineFromFile(os.path.join(data_root, 'negative_samples.txt'))
            self.test_negative_samples = {}
            for line in self.negative_samples_file_content:
                parts = line.strip().split(' ', 1)
                self.test_negative_samples[parts[0]] = parts[1].split(' ') if len(parts) > 1 else []
            
        datamaps = load_json(os.path.join(data_root, 'datamaps.json'))
        self.user2id_raw = datamaps['user2id'] # raw_id -> mapped_id
        self.item2id_raw = datamaps['item2id'] # raw_id -> mapped_id
        self.id2user_mapped = datamaps['id2user'] # mapped_id -> raw_id
        self.id2item_mapped = datamaps['id2item'] # mapped_id -> raw_id
        
        # This maps P5's mapped user ID to the original raw UserID string (which can serve as a 'name' or 'desc')
        self.user_id_mapped_to_desc = load_pickle(os.path.join(data_root, 'user_id2name.pkl'))
            
        print('Computing datum_info for P5_ML1M_Dataset...')
        self.total_length = 0
        self.datum_info = [] # Stores tuples of (global_idx, task_group_key, original_data_idx_within_group, optional_task_specific_idx)
        self.compute_datum_info()
        print(f"Total number of samples for mode '{self.mode}': {self.total_length}")

    def compute_datum_info(self):
        curr = 0
        for task_group_key in list(self.task_list.keys()): # e.g., 'rating', 'sequential'
            if task_group_key == 'rating':
                num_rating_data = len(self.rating_data)
                if num_rating_data == 0: continue
                self.total_length += num_rating_data * self.sample_numbers[task_group_key]
                for i in range(num_rating_data * self.sample_numbers[task_group_key]):
                    original_data_idx = i // self.sample_numbers[task_group_key]
                    self.datum_info.append((curr + i, task_group_key, original_data_idx))
                curr = self.total_length
            # Add other task groups if you plan to use them for ML1M with P5
            # For now, focusing on rating prediction.
            # Minimal handling for other task types if they are in task_list but not primary.
            elif task_group_key == 'sequential':
                num_seq_data = len(self.sequential_data)
                if num_seq_data == 0: continue
                # sample_numbers['sequential'] is a tuple, e.g., (for_direct_pred, for_candidate_list, for_yes_no)
                # For simplicity, we'll just use the first sample number if defined, or 1.
                sampling_multiplier = self.sample_numbers[task_group_key][0] if isinstance(self.sample_numbers[task_group_key], tuple) else self.sample_numbers[task_group_key]
                self.total_length += num_seq_data * sampling_multiplier
                for i in range(num_seq_data * sampling_multiplier):
                    original_data_idx = i // sampling_multiplier
                    self.datum_info.append((curr + i, task_group_key, original_data_idx))
                curr = self.total_length

            elif task_group_key == 'explanation': # ML1M has no explanations
                num_exp_data = len(self.exp_data) # Should be 0 for ML1M
                if num_exp_data == 0: continue 
                # This part likely won't run if exp_data is empty
                self.total_length += num_exp_data * self.sample_numbers[task_group_key]
                for i in range(num_exp_data * self.sample_numbers[task_group_key]):
                    original_data_idx = i // self.sample_numbers[task_group_key]
                    self.datum_info.append((curr + i, task_group_key, original_data_idx))
                curr = self.total_length
            
            elif task_group_key == 'review': # ML1M has no textual reviews to summarize or predict rating from
                # 'rating_data' is used for "review" task when it implies rating prediction from text.
                # Here, we'll assume 'review' task refers to a different aspect if it's in task_list.
                # If rating_data is used for this, it's redundant with the 'rating' task group.
                # For now, let's assume it's minimal or not the focus.
                num_review_data_for_task = len(self.rating_data) # Or some other source if defined
                if num_review_data_for_task == 0 : continue
                self.total_length += num_review_data_for_task * self.sample_numbers[task_group_key]
                for i in range(num_review_data_for_task * self.sample_numbers[task_group_key]):
                    original_data_idx = i // self.sample_numbers[task_group_key]
                    self.datum_info.append((curr + i, task_group_key, original_data_idx)) # Use rating_data for this for now
                curr = self.total_length

            elif task_group_key == 'traditional':
                # Traditional often means direct yes/no recommendation or ranking from candidates
                num_users_for_trad = len(self.id2user_mapped) # Number of unique mapped users
                if num_users_for_trad == 0: continue
                # sample_numbers['traditional'] is a tuple, e.g., (for_yes_no, for_candidate_ranking)
                sampling_multiplier_trad = self.sample_numbers[task_group_key][0] if isinstance(self.sample_numbers[task_group_key], tuple) else self.sample_numbers[task_group_key]
                self.total_length += num_users_for_trad * sampling_multiplier_trad
                for i in range(num_users_for_trad * sampling_multiplier_trad):
                    original_user_idx = i // sampling_multiplier_trad # This will be an index into list(self.id2user_mapped.keys())
                    self.datum_info.append((curr + i, task_group_key, original_user_idx))
                curr = self.total_length
            else:
                print(f"Warning: Task group '{task_group_key}' not specifically handled for ML1M datum_info computation.")
    
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0

        datum_info_entry = self.datum_info[idx]
        task_group_key = datum_info_entry[1]
        original_data_idx = datum_info_entry[2]
        
        source_text = ""
        target_text = ""

        if task_group_key == 'rating':            
            rating_datum = self.rating_data[original_data_idx] # review_splits.pkl content
            
            # Choose a random prompt from the 'rating' task_list for ML1M
            task_template_key = random.choice(self.task_list[task_group_key])
            task_template = self.all_tasks['rating'][task_template_key]

            mapped_user_id = rating_datum['reviewerID'] # Already mapped ID
            mapped_item_id = rating_datum['asin']       # Already mapped ID
            star_rating = rating_datum['overall']
            item_title = rating_datum.get('title', "an unnamed movie") # Use .get for safety
            user_desc = self.user_id_mapped_to_desc.get(mapped_user_id, mapped_user_id) # Raw UserID as desc

            if task_template['id'] == '1-1':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format(str(float(star_rating))) # P5 expects string target
            elif task_template['id'] == '1-2':
                source_text = task_template['source'].format(mapped_user_id, item_title) 
                target_text = task_template['target'].format(str(float(star_rating)))
            elif task_template['id'] == '1-3': # Yes/No for a given rating
                rand_prob = random.random()
                star_to_check = int(star_rating)
                if rand_prob > 0.5: # True case
                    target_ans = 'yes'
                else: # False case: pick a different rating
                    possible_ratings = [r for r in [1,2,3,4,5] if r != int(star_rating)]
                    if not possible_ratings: possible_ratings = [1] # fallback
                    star_to_check = random.choice(possible_ratings)
                    target_ans = 'no'
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id, star_to_check)
                target_text = task_template['target'].format(target_ans)
            elif task_template['id'] == '1-4': # Like/Dislike
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format('like' if star_rating >= 4.0 else 'dislike')
            elif task_template['id'] == '1-5':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id, item_title)
                target_text = task_template['target'].format(str(float(star_rating)))
            elif task_template['id'] == '1-6':
                source_text = task_template['source'].format(user_desc, mapped_item_id)
                target_text = task_template['target'].format(str(float(star_rating)))
            elif task_template['id'] == '1-7':
                source_text = task_template['source'].format(user_desc, item_title)
                target_text = task_template['target'].format(str(float(star_rating)))
            else: # Fallback or if more templates are added
                source_text = self.all_tasks['rating']['1-1']['source'].format(mapped_user_id, mapped_item_id)
                target_text = self.all_tasks['rating']['1-1']['target'].format(str(float(star_rating)))
                
        elif task_group_key == 'diversity':
            if task_template['id'] == '6-1':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format(str(float(star_rating))) # P5 expects string target
            elif task_template['id'] == '6-2':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format('yes' if star_rating >= 6.0 else 'no')
            elif task_template['id'] == '6-3':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format(str(float(star_rating)))
            elif task_template['id'] == '6-4':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format('yes' if star_rating >= 6.0 else 'no')
            elif task_template['id'] == '6-5':
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id, item_title)
                target_text = task_template['target'].format(str(float(star_rating)))
            elif task_template['id'] == '6-6':
                source_text = task_template['source'].format(user_desc, mapped_item_id)
                target_text = task_template['target'].format(str(float(star_rating)))
            else: # Fallback or if more templates are added
                source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
                target_text = task_template['target'].format('yes' if star_rating >= 6.0 else 'no')

        # --- Add other task group handling if used for ML1M ---
        # For now, primarily focusing on rating. Below are minimal placeholders.
        elif task_group_key == 'sequential':
            if not self.sequential_data or original_data_idx >= len(self.sequential_data): # Boundary check
                # Fallback to a simple rating task if sequential data is problematic
                return self[random.randint(0, len(self.datum_info) -1 )] # Retry with another random sample

            seq_line = self.sequential_data[original_data_idx]
            parts = seq_line.split(' ')
            mapped_user_id = parts[0]
            item_sequence = parts[1:]
            
            if len(item_sequence) < 2: # Need at least one history item and one target
                 # Fallback if sequence too short
                return self[random.randint(0, len(self.datum_info) -1 )]


            # Simplified sequential task: predict last item from second to last
            history_items = item_sequence[:-1]
            target_item_id = item_sequence[-1]
            
            # Use the ML1M specific sequential prompt
            task_template = self.all_tasks['sequential']['2-1-ml1m']
            source_text = task_template['source'].format(mapped_user_id, " , ".join(history_items[-5:])) # Use last 5 items as history
            target_text = task_template['target'].format(target_item_id)

        elif task_group_key in ['explanation', 'review', 'traditional']:
            # Since ML1M lacks rich data for these, we'll use a simple rating prompt as a fallback
            # to ensure the dataloader always returns valid text data.
            # This is a practical way to make the model train if these task keys are in args.losses
            # but you primarily care about rating.
            rating_datum = random.choice(self.rating_data) # Pick a random rating entry
            task_template = self.all_tasks['rating']['1-1'] # Use a basic rating prompt

            mapped_user_id = rating_datum['reviewerID']
            mapped_item_id = rating_datum['asin']
            star_rating = rating_datum['overall']
            
            source_text = task_template['source'].format(mapped_user_id, mapped_item_id)
            target_text = task_template['target'].format(str(float(star_rating)))
        else:
            raise NotImplementedError(f"Task group {task_group_key} not implemented for ML1M.")

        # Tokenization
        input_ids = self.tokenizer.encode(
                source_text, padding='max_length', truncation=True, max_length=self.args.max_text_length)
        
        # Define the local variable that holds the tokenized text
        tokenized_text_local = self.tokenizer.tokenize(source_text) # Use a distinct local name
        
        # Pass this local variable to the method
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text_local) 
        
        target_ids = self.tokenizer.encode(
                target_text, padding='max_length', truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids) # Or sum(attention_mask)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids) # Or sum(attention_mask_for_target)

        out_dict['source_text'] = source_text
        # out_dict['tokenized_text'] = tokenized_text_for_wwi # Optional, for debugging
        out_dict['target_text'] = target_text
        out_dict['task'] = task_group_key # Store the main task group
        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text_list_arg): # Parameter name changed for clarity
        ww_ids = []
        curr = 0
        # Use the passed argument name consistently
        if not tokenized_text_list_arg: 
            return [0] * self.args.max_text_length # Use 0 for padding whole word IDs

        # Use the passed argument name consistently
        for i in range(len(tokenized_text_list_arg)): 
            if tokenized_text_list_arg[i].startswith(' ') or i == 0: 
                curr += 1
            # Ensure whole word IDs do not exceed the embedding table size (e.g., 511 for size 512)
            ww_ids.append(min(curr, 511)) 
        
        # Pad to max_text_length, assuming the last whole_word_id (for EOS) should be 0
        final_ww_ids = [0] * self.args.max_text_length 
        
        copy_len = min(len(ww_ids), self.args.max_text_length - 1) # Leave space for the EOS token's 0
        final_ww_ids[:copy_len] = ww_ids[:copy_len]
        # The last position (for EOS) will remain 0 due to initialization, which is typical for P5.
                
        return final_ww_ids

    def collate_fn(self, batch):
        # This collate_fn is mostly standard, adapted from P5's existing ones.
        batch_entry = {}
        B = len(batch)
        args = self.args

        # Determine max sequence lengths for padding in this batch
        # Note: tokenizer.encode already pads to args.max_text_length, so S_W_L and T_W_L will be fixed.
        S_W_L = self.args.max_text_length
        T_W_L = self.args.gen_max_length

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.zeros(B, S_W_L, dtype=torch.long) # Pad with 0 for whole_word_ids
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        loss_weights = torch.ones(B, dtype=torch.float)
        tasks = []
        source_texts = []
        target_texts = []

        for i, entry in enumerate(batch):
            input_ids[i, :] = entry['input_ids'] # Assuming entry['input_ids'] is already padded/truncated
            whole_word_ids[i, :] = entry['whole_word_ids'] # Assuming this is also correctly sized
            target_ids[i, :] = entry['target_ids']

            tasks.append(entry['task'])
            source_texts.append(entry['source_text'])
            target_texts.append(entry['target_text'])
            loss_weights[i] = entry['loss_weight']

        # For T5, labels not equal to pad_token_id are considered, others are set to -100
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        batch_entry['task'] = tasks
        batch_entry['source_text'] = source_texts
        batch_entry['target_text'] = target_texts
        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['loss_weights'] = loss_weights
        return batch_entry


# Modify the main get_loader function to include ML1M
def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False,
               all_tasks_templates_override=None): # Added for flexibility

    if 't5' in args.backbone:
        # Use the P5Tokenizer from the src directory
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case
        )
    else:
        # Fallback or raise error if backbone not T5 based
        raise ValueError(f"Unsupported backbone: {args.backbone} for P5Tokenizer")

    if split == 'ml1m': 
        import all_ml1m_templates # Corrected import
        task_templates = all_ml1m_templates.all_tasks
        dataset = P5_ML1M_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split, 
            rating_augment=False
        )
    elif split == 'yelp':
        import all_yelp_templates # Corrected import
        task_templates = all_yelp_templates.all_tasks
        # Assuming P5_Yelp_Dataset is defined in this file or imported correctly
        # For this example, using a placeholder if not defined
        try:
            dataset = P5_Yelp_Dataset( 
                task_templates, task_list, tokenizer, args, sample_numbers,
                mode=mode, split=split, rating_augment=args.rating_augment
            )
        except NameError: # If P5_Yelp_Dataset isn't defined in this scope
             dataset = P5_Yelp_Dataset_Placeholder(
                task_templates, task_list, tokenizer, args, sample_numbers,
                mode=mode, split=split, rating_augment=args.rating_augment
            )
    elif split in ['beauty', 'toys', 'sports']: 
        import all_amazon_templates # Corrected import
        task_templates = all_amazon_templates.all_tasks
        try:
            dataset = P5_Amazon_Dataset( 
                task_templates, task_list, tokenizer, args, sample_numbers,
                mode=mode, split=split, rating_augment=args.rating_augment
            )
        except NameError: # If P5_Amazon_Dataset isn't defined in this scope
             dataset = P5_Amazon_Dataset_Placeholder(
                task_templates, task_list, tokenizer, args, sample_numbers,
                mode=mode, split=split, rating_augment=args.rating_augment
            )
    else:
        raise ValueError(f"Unknown dataset split: {split}")

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else: 
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=False, 
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader

# Placeholder for Yelp/Amazon datasets if you copy this file and they are not defined
# If they are, these placeholders are not needed.
class P5_Yelp_Dataset_Placeholder(Dataset): 
    def __init__(self, *args, **kwargs): print("Note: Using Placeholder for P5_Yelp_Dataset in pretrain_data.py"); self.data = []
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx] if self.data else {} # Ensure it returns a dict
    def collate_fn(self, batch): return {} if not batch else batch[0] 

class P5_Amazon_Dataset_Placeholder(Dataset): 
    def __init__(self, *args, **kwargs): print("Note: Using Placeholder for P5_Amazon_Dataset in pretrain_data.py"); self.data = []
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx] if self.data else {} # Ensure it returns a dict
    def collate_fn(self, batch): return {} if not batch else batch[0]

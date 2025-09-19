# P5-main/notebooks/test_lab_experiment_leave_one_out.py

import sys
import os
import collections
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
import torch.optim as optim # For fine-tuning optimizer
import torch.backends.cudnn as cudnn
import math 
import pandas as pd 

# --- Adjust Python Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..')) 

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

# --- P5 Imports ---
try:
    from src.param import parse_args
    from src.utils import LossMeter, load_state_dict
    from transformers import T5Tokenizer, T5TokenizerFast, T5Config, AdamW, get_linear_schedule_with_warmup
    from src.tokenization import P5Tokenizer, P5TokenizerFast
    from src.pretrain_model import P5Pretraining
    from src.pretrain_data import get_loader # We'll use a custom simple dataloader for fine-tuning
    from src.all_ml1m_templates import all_tasks as ml1m_task_templates # For ML1M prompts
    from evaluate.utils import root_mean_square_error, mean_absolute_error
except ImportError as e:
    print(f"Error importing P5 modules: {e}")
    print("Please ensure that this script is run from a location where 'src' and 'evaluate' are accessible,")
    print("or that P5-main and P5-main/src are in your PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    exit(1)

import pickle
import json
from torch.utils.data import Dataset, DataLoader

def load_pickle(filename):
    """Loads a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    """Saves data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    """Reads lines from a file, stripping newline characters."""
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

# --- DotDict for args ---
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

# --- Custom Dataset for Lab Experiment Fine-tuning and Testing ---
class LabDataDataset(Dataset):
    def __init__(self, data, tokenizer, args, prompt_id='6-1'):
        self.data = data # List of {'reviewerID': mapped_user_id, 'asin': mapped_item_id, 'overall': rating, 'title': title}
        self.tokenizer = tokenizer
        self.args = args
        self.prompt_id = prompt_id
        self.all_prompts = ml1m_task_templates # Using ML1M templates

        if 'rating' not in self.all_prompts or self.prompt_id not in self.all_prompts['rating']:
            raise ValueError(f"Prompt ID {self.prompt_id} not found in rating templates.")
        self.task_template = self.all_prompts['rating'][self.prompt_id]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.data[idx]
        out_dict = {}
        
        mapped_user_id = item_data['reviewerID']
        mapped_item_id = item_data['asin']
        star_rating = item_data['overall']
        item_title = item_data.get('title', "a movie")
        # user_desc can be fetched if user_id_mapped_to_desc is loaded and passed
        # For simplicity, using mapped_user_id if user_desc not critical for the chosen prompt
        user_desc = item_data.get('user_desc', mapped_user_id)


        # Format source text based on the selected prompt
        if self.task_template['id'] == '6-1':
            source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        elif self.task_template['id'] == '6-2':
            source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        elif self.task_template['id'] == '6-3':
            source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        elif self.task_template['id'] == '6-4':
             source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        elif self.task_template['id'] == '6-5':
            source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        elif self.task_template['id'] == '6-6':
             source_text = self.task_template['source'].format(mapped_user_id, mapped_item_id)
        else: # Fallback or add more prompt handlers
            print(f"Warning: Prompt ID {self.task_template['id']} specific formatting not explicitly handled in LabDataDataset, using 1-1.")
            source_text = ml1m_task_templates['rating']['1-1']['source'].format(mapped_user_id, mapped_item_id)

        target_text = self.task_template['target'].format(str(float(star_rating)))

        input_ids = self.tokenizer.encode(
            source_text, padding='max_length', truncation=True, max_length=self.args.max_text_length
        )
        
        # Simplified whole_word_ids for this specific dataset.
        # P5's JointEncoder requires it. A more robust implementation would align with main P5 preprocessing.
        tokenized_text_local = self.tokenizer.tokenize(source_text)
        curr = 0
        ww_ids = []
        if not tokenized_text_local:
            ww_ids = [0] * self.args.max_text_length
        else:
            for i in range(len(tokenized_text_local)):
                if tokenized_text_local[i].startswith(' ') or i == 0:
                    curr += 1
                ww_ids.append(min(curr, 511))
            
            final_ww_ids = [0] * self.args.max_text_length
            copy_len = min(len(ww_ids), self.args.max_text_length - 1)
            final_ww_ids[:copy_len] = ww_ids[:copy_len]
            ww_ids = final_ww_ids


        target_ids = self.tokenizer.encode(
            target_text, padding='max_length', truncation=True, max_length=self.args.gen_max_length
        )

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(ww_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['source_text'] = source_text # For logging
        out_dict['target_text'] = target_text # For logging
        out_dict['raw_user_id'] = item_data.get('raw_user_id_for_csv', mapped_user_id) # For CSV
        out_dict['raw_item_id'] = item_data.get('raw_item_id_for_csv', mapped_item_id) # For CSV
        
        return out_dict

def lab_collate_fn(batch, tokenizer, args):
    batch_entry = {}
    B = len(batch)
    S_W_L = args.max_text_length
    T_W_L = args.gen_max_length

    input_ids = torch.ones(B, S_W_L, dtype=torch.long) * tokenizer.pad_token_id
    whole_word_ids = torch.zeros(B, S_W_L, dtype=torch.long)
    target_ids = torch.ones(B, T_W_L, dtype=torch.long) * tokenizer.pad_token_id
    
    source_texts = []
    target_texts = []
    raw_user_ids_for_csv = []
    raw_item_ids_for_csv = []


    for i, entry in enumerate(batch):
        input_ids[i, :] = entry['input_ids']
        whole_word_ids[i, :] = entry['whole_word_ids']
        target_ids[i, :] = entry['target_ids']
        source_texts.append(entry['source_text'])
        target_texts.append(entry['target_text'])
        raw_user_ids_for_csv.append(entry['raw_user_id'])
        raw_item_ids_for_csv.append(entry['raw_item_id'])

    target_ids[target_ids == tokenizer.pad_token_id] = -100 # For T5 loss

    batch_entry['input_ids'] = input_ids
    batch_entry['whole_word_ids'] = whole_word_ids
    batch_entry['target_ids'] = target_ids
    batch_entry['source_text'] = source_texts
    batch_entry['target_text'] = target_texts
    batch_entry['raw_user_id_for_csv'] = raw_user_ids_for_csv
    batch_entry['raw_item_id_for_csv'] = raw_item_ids_for_csv
    return batch_entry


def main():
    args = DotDict()

    # --- Basic P5 Arguments (match your pre-training if possible) ---
    args.backbone = 't5-small' 
    args.dropout = 0.1
    args.losses = 'rating' # Focus on rating
    args.tokenizer = 'p5'
    args.max_text_length = 512 
    args.do_lower_case = False 
    args.gen_max_length = 64
    args.whole_word_embed = True
    args.seed = 42
    
    # --- Paths ---
    args.ml1m_processed_data_dir = os.path.join(project_root, "data", "ml1m")
    args.pretrained_p5_checkpoint_path = os.path.join(project_root, "snap", "Epoch10.pth") #  PATH TO YOUR PRETRAINED ML1M MODEL
    args.lab_experiment_data_path = os.path.join(project_root, "data", "ml1m", "review_splits.pkl") # PATH TO YOUR NEW LAB DATA (preprocessed)
    args.output_dir = os.path.join(project_root, "64_diversity_output")
    args.predictions_csv_path = os.path.join(args.output_dir, "64_diversity_predictions_6.csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Fine-tuning Hyperparameters ---
    args.ft_epochs = 5 # Number of epochs for fine-tuning on leave-one-out train set
    args.ft_lr = 5e-5  # Learning rate for fine-tuning
    args.ft_batch_size = 4 # Small batch size for fine-tuning small datasets
    args.ft_eval_prompt_id = '6-6' # Prompt to use for prediction of the left-out item

    # --- GPU Setup ---
    if torch.cuda.is_available():
        gpu_to_use = 0 
        args.gpu = gpu_to_use
        current_device = torch.device(f'cuda:{gpu_to_use}')
        print(f"Using GPU: {gpu_to_use}")
    else:
        args.gpu = -1 
        current_device = torch.device('cpu')
        print("CUDA not available, using CPU.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # --- Load Tokenizer and Config ---
    print("Loading tokenizer and config...")
    p5_config = T5Config.from_pretrained(args.backbone)
    p5_config.dropout_rate = args.dropout # Ensure consistency
    p5_config.losses = args.losses # Pass losses to config as P5 model expects it

    p5_tokenizer = P5Tokenizer.from_pretrained(
        args.backbone,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )
    
    # --- Load Lab Experiment Data ---
    # This data should be a list of dictionaries:
    # [{'reviewerID': 'mapped_user_1', 'asin': 'mapped_item_A', 'overall': 5.0, 'title': 'Movie A', 'raw_user_id_for_csv': 'original_user_id_1', 'raw_item_id_for_csv': 'original_item_id_A'}, ...]
    # 'reviewerID' and 'asin' should be IDs ALREADY MAPPED according to your ML1M datamaps.json
    if not os.path.exists(args.lab_experiment_data_path):
        print(f"ERROR: Lab experiment data not found at {args.lab_experiment_data_path}")
        print("Please prepare your lab data in the specified format and path.")
        # Create a dummy file for demonstration if it doesn't exist
        print("Creating a dummy lab_ratings_mapped.pkl for demonstration purposes...")
        dummy_lab_data_dir = os.path.dirname(args.lab_experiment_data_path)
        if not os.path.exists(dummy_lab_data_dir): os.makedirs(dummy_lab_data_dir)
        # Load datamaps to get some valid mapped IDs for dummy data
        datamaps_path = os.path.join(args.ml1m_processed_data_dir, 'datamaps.json')
        if os.path.exists(datamaps_path):
            datamaps = load_json(datamaps_path)
            sample_users = list(datamaps['id2user'].keys())[:5] # Take first 5 mapped users
            sample_items = list(datamaps['id2item'].keys())[:10] # Take first 10 mapped items
            dummy_data = []
            for u_idx, user_id in enumerate(sample_users):
                raw_user_id = datamaps['id2user'][user_id]
                for i_idx in range(min(len(sample_items), 3)): # Each user rates 3 items
                    item_id = sample_items[(u_idx + i_idx) % len(sample_items)] # Cycle through items
                    raw_item_id = datamaps['id2item'][item_id]
                    dummy_data.append({
                        'reviewerID': user_id, 'asin': item_id, 
                        'overall': float(random.randint(1,5)), 'title': f"Dummy Movie for {item_id}",
                        'raw_user_id_for_csv': raw_user_id, 'raw_item_id_for_csv': raw_item_id
                    })
            save_pickle(dummy_data, args.lab_experiment_data_path)
            print(f"Dummy data saved with {len(dummy_data)} entries.")
        else:
            print(f"Could not create dummy lab data: {datamaps_path} not found.")
            exit(1)

    #print(f"Loading lab experiment data from: {args.lab_experiment_data_path}")
    #lab_data_all = load_pickle(args.lab_experiment_data_path)

    # Group data by user
    #user_ratings_lab = defaultdict(list)
    #for record in lab_data_all:
    #    user_ratings_lab[record['reviewerID']].append(record)
    
    print(f"Loading lab experiment data from: {args.lab_experiment_data_path}")
    # lab_data_all is expected to be a dictionary like {'train': [...], 'val': [...], 'test': [...]}
    loaded_data_splits = load_pickle(args.lab_experiment_data_path) 

    # For Leave-One-Out, you typically use all available data for a user.
    # The data_preprocess_lab.py script saves all lab experiment data into the 'test' key of review_splits.pkl.
    lab_data_list_of_records = loaded_data_splits.get('test', []) 

    if not lab_data_list_of_records and ('train' in loaded_data_splits or 'val' in loaded_data_splits):
        print(f"Warning: No data found in 'test' key of {args.lab_experiment_data_path}. Checking 'train' and 'val' keys if they exist and are non-empty.")
        # This is a fallback if the data was structured differently than expected by data_preprocess_lab.py
        # For LOO, you usually want all data for the specific users.
        if loaded_data_splits.get('train'):
            lab_data_list_of_records.extend(loaded_data_splits['train'])
        if loaded_data_splits.get('val'):
            lab_data_list_of_records.extend(loaded_data_splits['val'])
    
    if not isinstance(lab_data_list_of_records, list):
        print(f"Error: Expected a list of records from the loaded pickle file (e.g., from key 'test'), but got {type(lab_data_list_of_records)}")
        print("Please ensure 'args.lab_experiment_data_path' points to a pickle file containing a dictionary with a key (e.g., 'test') that holds a list of rating records.")
        exit(1)
    
    if not lab_data_list_of_records:
        print(f"Warning: No records found in the loaded lab data from {args.lab_experiment_data_path} after checking relevant keys. The script might not produce results.")
        # The script will proceed, but likely print "No leave-one-out results were generated." later.

    # Group data by user
    user_ratings_lab = defaultdict(list)
    for record in lab_data_list_of_records: # Now 'record' will be a dictionary
        if not isinstance(record, dict):
            print(f"Skipping an item in lab_data_list_of_records because it's not a dictionary: {type(record)}")
            continue
        if 'reviewerID' not in record:
            print(f"Skipping record due to missing 'reviewerID': {record}")
            continue
        user_ratings_lab[record['reviewerID']].append(record)
    
    print(f"Loaded lab data for {len(user_ratings_lab)} users.")

    all_leave_one_out_results = []

    # --- Leave-One-Out Cross-Validation ---
    for user_id, ratings in tqdm(user_ratings_lab.items(), desc="Processing Users (Leave-One-Out)"):
        if len(ratings) < 2: # Need at least one item for training and one for testing
            print(f"User {user_id} has less than 2 ratings, skipping for leave-one-out.")
            continue

        for i in range(len(ratings)): # i is the index of the item to leave out
            # Create a fresh copy of the model for each LOO fine-tuning run
            #p5_model_loo = P5Pretraining.from_pretrained(args.backbone, config=p5_config)
            
            # Resize to match tokenizer
            #p5_model_loo.resize_token_embeddings(p5_tokenizer.vocab_size)  # = 32128
            #p5_model_loo = p5_model_loo.to(current_device)
            
            # Now load checkpoint
            #state_dict = load_state_dict(args.pretrained_p5_checkpoint_path, str(current_device))
            #p5_model_loo.load_state_dict(state_dict, strict=False)
            
            #if 'p5' in args.tokenizer:
            #     p5_model_loo.resize_token_embeddings(p5_tokenizer.vocab_size)
            #p5_model_loo = p5_model_loo.to(current_device)
            
            p5_model_loo = P5Pretraining(config=p5_config)
            
            # Step 2: Tokenizer ??? ?? resize
            p5_model_loo.resize_token_embeddings(p5_tokenizer.vocab_size)  # tokenizer.vocab_size == 32128
            
            # Step 3: GPU? ??
            p5_model_loo = p5_model_loo.to(current_device)
            
            # Step 4: ???? state_dict ??
            if os.path.exists(args.pretrained_p5_checkpoint_path):
                state_dict = load_state_dict(args.pretrained_p5_checkpoint_path, str(current_device))
                missing, unexpected = p5_model_loo.load_state_dict(state_dict, strict=False)
                print(f"Checkpoint loaded. Missing: {missing}, Unexpected: {unexpected}")
            else:
                print(f"Checkpoint not found at {args.pretrained_p5_checkpoint_path}")
            

            p5_model_loo.train() # Set to train mode for fine-tuning

            # Prepare fine-tuning data (all items except the i-th)
            loo_train_data = ratings[:i] + ratings[i+1:]
            loo_test_item = ratings[i]

            if not loo_train_data:
                print(f"User {user_id} has only one rating after trying to leave one out. Skipping this specific LOO iteration.")
                continue
            
            # Create DataLoader for fine-tuning
            ft_dataset = LabDataDataset(loo_train_data, p5_tokenizer, args, prompt_id=args.ft_eval_prompt_id)
            ft_loader = DataLoader(
                ft_dataset, 
                batch_size=args.ft_batch_size, 
                shuffle=True, 
                collate_fn=lambda b: lab_collate_fn(b, p5_tokenizer, args)
            )

            # Optimizer for fine-tuning
            optimizer = AdamW(p5_model_loo.parameters(), lr=args.ft_lr)
            
            # Fine-tuning loop
            for epoch in range(args.ft_epochs):
                epoch_loss = 0
                for batch_idx, batch_data in enumerate(ft_loader):
                    input_ids_ft = batch_data['input_ids'].to(current_device)
                    whole_word_ids_ft = batch_data['whole_word_ids'].to(current_device)
                    lm_labels_ft = batch_data['target_ids'].to(current_device)
                    
                    optimizer.zero_grad()
                    outputs = p5_model_loo(
                    input_ids=input_ids_ft,
                    whole_word_ids=whole_word_ids_ft, # This is an argument for JointEncoder, not P5.forward
                    labels=lm_labels_ft,
                    return_dict=True
                    # Do NOT pass reduce_loss=True here, as we will handle reduction manually
                    )
                    
                    # output['loss'] is what comes from P5.forward method.
                    # If reduce_loss=False was effectively used in P5.forward,
                    # output['loss'] is a 1D tensor of losses for each token (where label != -100).
                    unreduced_loss = outputs.loss 
                    
                    if unreduced_loss is not None:
                        if unreduced_loss.dim() > 0:
                            scalar_loss = unreduced_loss.mean()
                        else: # It's already a scalar
                            scalar_loss = unreduced_loss
                    
                        if torch.isnan(scalar_loss) or torch.isinf(scalar_loss):
                            print(f"Warning: NaN or Inf loss detected: {scalar_loss}. Skipping backward pass for this batch.")
                        else:
                            scalar_loss.backward() # Call backward on the now guaranteed scalar loss
                            optimizer.step()
                            epoch_loss += scalar_loss.item()

            # Prediction for the left-out item
            p5_model_loo.eval()
            test_dataset_loo = LabDataDataset([loo_test_item], p5_tokenizer, args, prompt_id=args.ft_eval_prompt_id)
            # No need for a full dataloader for a single item, can directly use __getitem__
            single_test_sample = test_dataset_loo[0] 
            
            input_ids_test = single_test_sample['input_ids'].unsqueeze(0).to(current_device) # Add batch dimension

            with torch.no_grad():
                generated_outputs = p5_model_loo.generate(
                    input_ids_test,
                    max_length=args.gen_max_length,
                    num_beams=1 
                )
                predicted_text = p5_tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
            
            actual_rating_text = single_test_sample['target_text']
            predicted_rating_float = -1.0
            try:
                predicted_rating_float = float(predicted_text)
            except ValueError:
                # print(f"Warning: Could not parse prediction '{predicted_text}' for user {user_id}, item {loo_test_item['asin']}")
                pass
            
            all_leave_one_out_results.append({
                'user_id_mapped': user_id, # Mapped ID used in P5
                'item_id_mapped': loo_test_item['asin'], # Mapped ID used in P5
                'raw_user_id': loo_test_item.get('raw_user_id_for_csv', user_id),
                'raw_item_id': loo_test_item.get('raw_item_id_for_csv', loo_test_item['asin']),
                'item_title': loo_test_item.get('title', "N/A"),
                'actual_rating': float(actual_rating_text),
                'predicted_rating_text': predicted_text,
                'predicted_rating_float': predicted_rating_float,
                'prompt_used_for_prediction': args.ft_eval_prompt_id,
                'source_prompt_text': single_test_sample['source_text']
            })
            
            # Clean up to free memory if running many LOO iterations
            del p5_model_loo, optimizer, ft_loader, ft_dataset, test_dataset_loo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # --- Save results to CSV ---
    if all_leave_one_out_results:
        results_df = pd.DataFrame(all_leave_one_out_results)
        results_df.to_csv(args.predictions_csv_path, index=False)
        print(f"\nSaved leave-one-out prediction details to: {args.predictions_csv_path}")

        # --- Calculate Overall Metrics ---
        valid_predictions = results_df[results_df['predicted_rating_float'] != -1.0]
        if not valid_predictions.empty:
            rating_pairs = list(zip(valid_predictions['actual_rating'].tolist(), valid_predictions['predicted_rating_float'].tolist()))
            min_r, max_r = 0.0, 10.0
            
            overall_rmse = root_mean_square_error(rating_pairs, max_r, min_r)
            overall_mae = mean_absolute_error(rating_pairs, max_r, min_r)
            
            print(f"\n--- Overall Leave-One-Out Metrics (on {len(valid_predictions)} valid predictions) ---")
            print(f"RMSE: {overall_rmse:.4f}")
            print(f"MAE:  {overall_mae:.4f}")
        else:
            print("No valid numeric predictions were made to calculate overall RMSE/MAE.")
    else:
        print("No leave-one-out results were generated.")

    print("\n--- Lab Experiment Leave-One-Out Script Finished ---")


if __name__ == "__main__":
    main()

# P5-main/notebooks/test_ml1m_small.py

import sys
import os
import collections
import random
from pathlib import Path
import logging # Not strictly used in this script, but often in P5
import shutil # Not strictly used in this script
import time
from packaging import version
from collections import defaultdict
from tqdm import tqdm 
import numpy as np
import gzip
import torch
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP # Not for single GPU eval
# import torch.distributed as dist # Not for single GPU eval
import torch.backends.cudnn as cudnn
import math 
import pandas as pd # For CSV output

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
    # from src.dist_utils import reduce_dict # Not needed for single GPU eval
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast
    from src.pretrain_model import P5Pretraining
    from src.trainer_base import TrainerBase # Not directly used, but good for context
    from src.pretrain_data import get_loader 
    from src.all_ml1m_templates import all_tasks as ml1m_task_templates 
    from evaluate.utils import root_mean_square_error, mean_absolute_error
    # from evaluate.metrics4rec import evaluate_all # For ranking, not used in this focused version
except ImportError as e:
    print(f"Error importing P5 modules: {e}")
    print("Please ensure that this script is run from a location where 'src' and 'evaluate' are accessible,")
    print("or that P5-main and P5-main/src are in your PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    exit(1)

import pickle
import json

# --- Utility Functions ---
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
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

def main():
    args = DotDict()

    args.distributed = False
    args.multiGPU = False 
    args.fp16 = True     
    args.train = "ml1m"  # Placeholder for get_loader if it uses args.train contextually
    args.valid = "ml1m"  # Placeholder
    args.test = "ml1m"   # This will be used by get_loader via args.current_test_split_name
    args.batch_size = 16 
    args.optim = 'adamw' 
    args.warmup_ratio = 0.05
    args.lr = 1e-3
    args.num_workers = 0 
    args.clip_grad_norm = 1.0
    args.losses = 'rating' 
    args.backbone = 't5-small' 
    args.output = os.path.join(project_root, "snap") 
    args.epoch = 10 
    args.local_rank = 0 

    args.comment = ''
    args.dropout = 0.1
    args.tokenizer = 'p5'
    args.max_text_length = 512 
    args.do_lower_case = False 
    args.gen_max_length = 64   # Max length for generated rating text

    args.seed = 42
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.whole_word_embed = True 

    cudnn.benchmark = True
    args.world_size = 1 

    if torch.cuda.is_available():
        gpu_to_use = 0 
        args.gpu = gpu_to_use
        args.rank = gpu_to_use 
        torch.cuda.set_device(f'cuda:{gpu_to_use}')
        current_device = torch.device(f'cuda:{gpu_to_use}')
        print(f"Using GPU: {gpu_to_use}")
    else:
        args.gpu = -1 
        args.rank = -1
        current_device = torch.device('cpu')
        print("CUDA not available, using CPU. This will be very slow.")

    print(f"Testing with arguments: {args.__dict__}")

    # --- Model and Tokenizer Creation ---
    def create_config_eval(args_dict):
        from transformers import T5Config
        if 't5' in args_dict.backbone:
            config_class = T5Config
        else:
            print(f"Warning: Backbone {args_dict.backbone} not T5. Config creation might fail.")
            return None
        config = config_class.from_pretrained(args_dict.backbone)
        config.dropout_rate = args_dict.dropout
        config.dropout = args_dict.dropout 
        config.attention_dropout = args_dict.dropout
        config.activation_dropout = args_dict.dropout
        config.losses = args_dict.losses
        return config

    def create_tokenizer_eval(args_dict):
        tokenizer_name = args_dict.backbone
        tokenizer = P5Tokenizer.from_pretrained(
            tokenizer_name,
            max_length=args_dict.max_text_length,
            do_lower_case=args_dict.do_lower_case,
        )
        print(f"Using P5Tokenizer with backbone: {tokenizer_name}")
        return tokenizer

    def create_model_eval(model_class, config_obj, args_dict):
        print(f'Building Model based on: {args_dict.backbone}')
        model_name = args_dict.backbone
        model = model_class.from_pretrained(model_name, config=config_obj)
        return model

    p5_config = create_config_eval(args)
    p5_tokenizer = create_tokenizer_eval(args)
    p5_model = create_model_eval(P5Pretraining, p5_config, args) # Using P5Pretraining as it has generate_step

    p5_model = p5_model.to(current_device)

    if 'p5' in args.tokenizer:
        p5_model.resize_token_embeddings(p5_tokenizer.vocab_size)
        print(f"Resized model token embeddings to: {p5_tokenizer.vocab_size}")
        
    p5_model.tokenizer = p5_tokenizer
    p5_model.eval()
    print("Model and Tokenizer created.")

    # --- Load Trained Model Checkpoint ---
    #args.load_path = os.path.join(args.output, "BEST_EVAL_LOSS.pth") 
    args.load_path = os.path.join(args.output, "Epoch10.pth") # Or a specific epoch

    if not os.path.exists(args.load_path):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"WARNING: Model checkpoint not found at {args.load_path}")
        print(f"Please update 'args.load_path' with the correct path to your trained ML1M model.")
        print(f"Using a randomly initialized model for demonstration purposes only.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        from pprint import pprint
        def load_checkpoint_eval(model_to_load, ckpt_path, device_str='cpu'):
            state_dict = load_state_dict(ckpt_path, device_str) 
            results = model_to_load.load_state_dict(state_dict, strict=False)
            print(f'Model loaded from: {ckpt_path}')
            pprint(results)
        load_checkpoint_eval(p5_model, args.load_path, str(current_device))
    print("Model loading step complete.")

    # --- Prepare Dataloader for ML1M Test Set ---
    args.current_test_split_name = 'ml1m' # This should match the folder name in P5-main/data/
    
    # Define the rating prompts you want to use for this evaluation run
    # Ensure these prompt IDs exist in your all_ml1m_templates.py
    prompts_to_test = ['1-1', '1-2', '1-5', '1-6', '1-7', '1-10'] # Example: test with a couple of different rating prompts
    
    # This list will store all prediction details for CSV export
    all_predictions_for_csv = []


    for rating_prompt_id in prompts_to_test:
        print(f"\nEvaluating with rating prompt: {rating_prompt_id}")
        test_task_list_rating = {'rating': [rating_prompt_id]}
        test_sample_numbers_rating = {'rating': 1, 'sequential': 0, 'explanation': 0, 'review': 0, 'traditional': 0}

        ml1m_test_loader_rating = get_loader(
            args,
            test_task_list_rating,
            test_sample_numbers_rating,
            split=args.current_test_split_name, 
            mode='test', 
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed 
        )
        print(f"DataLoader for rating prediction (prompt {rating_prompt_id}) created. Batches: {len(ml1m_test_loader_rating)}")

        # --- Evaluation - Rating Prediction (RMSE, MAE) ---
        print(f"Starting Rating Prediction Evaluation for prompt {rating_prompt_id} (RMSE, MAE)...")
        gt_ratings_float = []
        pred_ratings_float = []
        
        for i, batch in tqdm(enumerate(ml1m_test_loader_rating), total=len(ml1m_test_loader_rating), desc=f"Eval Prompt {rating_prompt_id}"):
            input_ids_batch = batch['input_ids'].to(current_device)
            # Store original input info for CSV - we need user_id and item_id from the batch source if possible
            # The batch['source_text'] contains the full prompt.
            # The batch itself doesn't directly have raw user/item IDs unless P5_ML1M_Dataset is modified to pass them.
            # For now, we'll save the source text.
            source_texts_batch = batch['source_text']
            target_texts_batch = batch['target_text'] # Ground truth rating strings

            with torch.no_grad():
                outputs = p5_model.generate(
                    input_ids_batch,
                    max_length=args.gen_max_length, 
                    num_beams=1 
                )
                results_text_batch = p5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for src_txt, target_str, pred_str in zip(source_texts_batch, target_texts_batch, results_text_batch):
                    try:
                        gt_val = float(target_str)
                        pred_val_parsed = -1.0 # Default if conversion fails
                        try:
                            pred_val_parsed = float(pred_str)
                        except ValueError:
                            # print(f"Warning: Could not convert prediction '{pred_str}' to float for source: {src_txt}")
                            pass # Keep pred_val_parsed as -1.0 or some indicator
                        
                        gt_ratings_float.append(gt_val)
                        pred_ratings_float.append(pred_val_parsed) # Store parsed or default
                        
                        # Store details for CSV
                        # Attempt to parse user_id and item_id from source_text (this is brittle)
                        # A better way would be to have P5_ML1M_Dataset return these raw/mapped IDs in the batch
                        user_id_in_prompt = "unknown"
                        item_id_in_prompt = "unknown"
                        try:
                            if "user_" in src_txt and "movie_" in src_txt: # Basic parsing for "user_{id} ... movie_{id}"
                                user_id_in_prompt = src_txt.split("user_")[1].split(" ")[0]
                                item_id_in_prompt = src_txt.split("movie_")[1].split(" ")[0].replace("?","").strip()
                        except:
                            pass

                        all_predictions_for_csv.append({
                            'prompt_id': rating_prompt_id,
                            'user_id_in_prompt': user_id_in_prompt,
                            'item_id_in_prompt': item_id_in_prompt,
                            'source_text_prompt': src_txt,
                            'ground_truth_rating': gt_val,
                            'predicted_rating_text': pred_str,
                            'predicted_rating_float': pred_val_parsed
                        })
                    except ValueError:
                        # print(f"Warning: Could not convert target '{target_str}' to float. Skipping for metrics.")
                        # Still log to CSV if needed
                        all_predictions_for_csv.append({
                            'prompt_id': rating_prompt_id,
                            'user_id_in_prompt': "unknown",
                            'item_id_in_prompt': "unknown",
                            'source_text_prompt': src_txt,
                            'ground_truth_rating_text': target_str,
                            'predicted_rating_text': pred_str,
                            'predicted_rating_float': -1.0 # Error indicator
                        })
                        continue 
        
        if gt_ratings_float and pred_ratings_float:
            # Filter out predictions where conversion failed (-1.0) for metric calculation
            valid_metric_pairs = [(gt, p) for gt, p in zip(gt_ratings_float, pred_ratings_float) if p != -1.0]

            if valid_metric_pairs:
                rating_pairs_for_metrics = list(zip(*valid_metric_pairs)) # Separate gt and pred
                gt_for_metrics = rating_pairs_for_metrics[0]
                pred_for_metrics = rating_pairs_for_metrics[1]
                
                # Create pairs for the metric functions
                eval_pairs = list(zip(gt_for_metrics, pred_for_metrics))

                min_rating, max_rating = 0.0, 10.0
                
                rmse = root_mean_square_error(eval_pairs, max_rating, min_rating)
                mae = mean_absolute_error(eval_pairs, max_rating, min_rating)
                
                print(f"\n--- Rating Prediction Results for Prompt {rating_prompt_id} (on {len(eval_pairs)} valid samples) ---")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE:  {mae:.4f}")
            else:
                print(f"\nNo valid rating pairs (numeric predictions) collected for prompt {rating_prompt_id} for RMSE/MAE calculation.")
        else:
            print(f"\nNo rating predictions or targets collected for prompt {rating_prompt_id}.")
        print(f"Rating prediction evaluation for prompt {rating_prompt_id} finished.")

    # --- Save all predictions to CSV ---
    if all_predictions_for_csv:
        output_csv_path = "./rating_predictions_details.csv"
        predictions_df = pd.DataFrame(all_predictions_for_csv)
        predictions_df.to_csv(output_csv_path, index=False)
        print(f"\nSaved detailed predictions to: {output_csv_path}")
    else:
        print("\nNo predictions were logged to save to CSV.")

    # --- Placeholder for other evaluations ---
    print("\nPlaceholder for other task evaluations (Sequential, Explanation, etc.)...")

    print("\n--- Evaluation Script Finished ---")

if __name__ == "__main__":
    main()
# Import necessary libraries
import argparse
import copy
import numpy as np
import torch
import recbole.evaluator.collector
from recbole.quick_start import run_recbole

# This line is for compatibility with older numpy versions used in some environments.
# In modern numpy, np.float is deprecated in favor of float.
np.float = float 

# --- Monkey-patching RecBole's Collector ---
# This modification is to ensure all necessary data (like item IDs) is available
# for custom metrics, such as 'averagepopularity', during evaluation.
new_Collector = recbole.evaluator.collector.Collector

def get_data_struct_new(self):
    """
    A modified version of the Collector's get_data_struct method.
    This version ensures that tensors are moved to the CPU before deepcopying,
    which can prevent device-related issues. It also retains the 'rec.items' key,
    which is needed for the average popularity metric.
    """
    # Move all tensors in the data structure to CPU
    for key in self.data_struct._data_dict:
        if isinstance(self.data_struct._data_dict[key], torch.Tensor):
            self.data_struct._data_dict[key] = self.data_struct._data_dict[key].cpu()
    
    # Create a deep copy of the data structure to return
    returned_struct = copy.deepcopy(self.data_struct)
    
    # Clean up some keys from the original structure to prepare for the next batch
    # NOTE: We intentionally DO NOT delete "rec.items" to make it available for metrics.
    for key in ["rec.topk", "rec.meanrank", "rec.score", "data.label"]:
        if key in self.data_struct:
            del self.data_struct[key]
            
    return returned_struct

# Apply the monkey-patch
new_Collector.get_data_struct = get_data_struct_new
# --- End of Monkey-patching ---


# --- Argument Parsing ---
# Set up command-line arguments to easily configure the experiment.
parser = argparse.ArgumentParser(description="Run baseline BPR model.")
parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID to use.')
parser.add_argument('-d', '--dataset_id', type=int, default=0, help='Index of the dataset to use (0: amazon, 1: yelp, 2: tmall).')
args = parser.parse_args()

# --- Dataset Configuration ---
# A list of available dataset names. The --dataset_id argument selects one.
datasets = ['amazon-books-23', 'yelp-2021', 'tmall-buy-merged']
selected_dataset = datasets[args.dataset_id]

# --- RecBole Configuration ---
# A dictionary holding all the parameters for the RecBole run.
parameter_dict = {
    'data_path': './dataset/',               # Path to the dataset directory.
    'gpu_id' : args.gpu_id,                  # GPU to use.
    'train_batch_size' : 8192,
    'eval_batch_size' : 8192,
    'load_col' : {'inter': ['user_id', 'item_id']}, # Columns to load.
    # Pre-filtering to ensure users and items have at least 10 interactions.
    'user_inter_num_interval' : '[10,inf)',
    'item_inter_num_interval' : '[10,inf)',
    # Evaluation metrics. 'averagepopularity' is a custom metric.
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'map', 'averagepopularity'],
    'eval_args' : {
        'split': {'RS': [0.8, 0.1, 0.1]}, # 80% train, 10% valid, 10% test split.
        'order': 'RO',                    # Chronological order.
        'group_by': 'none',               # Evaluate on all users together.
        'mode': {'valid': 'full', 'test': 'full'} # Full ranking evaluation.
    },
    'epochs': 50000,                      # Maximum number of epochs.
    'eval_step': 5,                       # Evaluate every 5 epochs.
    'stopping_step': 10,                  # Early stopping patience: stop if no improvement after 10 evaluations.
}

# --- Run Experiment ---
# Execute the RecBole pipeline with the specified model, dataset, and configuration.
print(f"Running BPR model on dataset: {selected_dataset} with GPU: {args.gpu_id}")
run_recbole(model='BPR', dataset=selected_dataset, config_dict=parameter_dict)
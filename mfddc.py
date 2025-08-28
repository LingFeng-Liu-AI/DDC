# Import necessary libraries
import os
import torch
import torch.nn as nn
import numpy as np
import json
from recbole.trainer import Trainer
import torch.nn.functional as F
from logging import getLogger
from tqdm import tqdm
import argparse
import copy
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, init_seed, set_color
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
import recbole.evaluator.collector

# Compatibility for older numpy versions
np.float = float

# --- Monkey-patching RecBole's Collector (same as in mf.py) ---
# This ensures the 'averagepopularity' metric can access necessary item data.
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
        else:
            self.data_struct._data_dict[key] = self.data_struct._data_dict[key] 
    # Create a deep copy of the data structure to return
    returned_struct = copy.deepcopy(self.data_struct)
    
    # Clean up some keys from the original structure to prepare for the next batch
    # NOTE: We intentionally DO NOT delete "rec.items" to make it available for metrics.
    for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
        if key in self.data_struct:
            del self.data_struct[key]
            
    return returned_struct
new_Collector.get_data_struct = get_data_struct_new
# --- End of Monkey-patching ---


# --- ANONYMOUS PATH CONFIGURATION ---
# NOTE TO REVIEWERS: Please replace the placeholder paths below with the actual paths
# to your pre-trained model file and the generated popularity direction file.
DEFAULT_MODEL_FILE = [
    './saved/path_to_your_amazon_model.pth',   # amazon
    './saved/path_to_your_yelp_model.pth',     # yelp
    './saved/path_from_step_1.pth',            # tmall (UPDATE THIS)
]
DEFAULT_REP_DIRECTION_FILE = [
    './e_pop_saved/amazon/your_amazon_direction.json', # amazon
    './e_pop_saved/yelp/your_yelp_direction.json',     # yelp
    './e_pop_saved/tmall/path_from_step_2.json',       # tmall (UPDATE THIS, or use the pre-calculated one)
]


class MFDebiasFinetune(GeneralRecommender):
    """
    A model that finetunes a pre-trained general recommender (e.g., BPR, LightGCN)
    to mitigate popularity bias using the DDC (Data-free Debiasing framework).
    It modifies user embeddings by adding components related to popularity and
    personalized preferences.
    """
    def __init__(self, config, dataset, pretrained_model, popularity_direction_norm):
        super(MFDebiasFinetune, self).__init__(config, dataset)
        self.logger = getLogger()
        
        # Load the pre-trained model and freeze its parameters
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()
        
        # Store the normalized popularity direction vector
        self.popularity_direction_norm = popularity_direction_norm.to(self.device)

        # --- Trainable Parameters ---
        # alpha: User-specific sensitivity to the global popularity direction.
        self.alphas = nn.Parameter(torch.FloatTensor(self.n_users).uniform_(-0.5, 0.5))
        # beta: User-specific sensitivity to their personalized preference direction.
        self.use_epre = config['use_epre']
        if self.use_epre:
            self.betas = nn.Parameter(torch.FloatTensor(self.n_users).uniform_(-0.5, 0.5))
        
        # --- Pre-computation for Efficiency ---
        self.logger.info("Pre-calculating original embeddings and scores...")
        with torch.no_grad():
            # Get original user and item embeddings from the pre-trained model
            self.user_embeds_orig = self.pretrained_model.user_embedding.weight.to(self.device)
            self.item_embeds_orig = self.pretrained_model.item_embedding.weight.to(self.device)
            
            # Pre-calculate original dot-product scores (y_uio) between all users and items
            self.y_uio_all = torch.matmul(self.user_embeds_orig.cpu(), self.item_embeds_orig.t().cpu())
            
            # If enabled, construct the personalized preference direction (e_pre) for each user
            if self.use_epre:
                self.logger.info("Constructing personalized preference direction (e_pre)...")
                self.e_pre_all_users = self._construct_e_pre(dataset, config).to(self.device)
        self.logger.info("Pre-calculation finished.")

        # --- Loss and Configuration ---
        self.mf_loss = BPRLoss()
        self.prediction_mode = config["prediction_mode"] # Controls which components are used for prediction
        self.loss_combination = config['loss_combination'] # Controls which components are used in the loss
        self.reg_weight = config["reg_weight"] # L2 regularization weight for alphas and betas
        
        # Caches for full-ranking evaluation
        self.restore_user_e = None
        self.restore_item_e = None

    def _construct_e_pre(self, dataset, config):
        """
        Constructs the personalized preference direction (e_pre) for each user.
        e_pre is the normalized mean of embeddings of items a user has interacted with,
        often filtered to include only the most relevant items (e.g., top-k preferred).
        """
        # Get configuration for e_pre construction
        sort_mode = config['epre_sort_mode']
        select_mode = config['epre_select_mode']
        agg_mode = config['epre_agg_mode']
        top_k_ratio = config['epre_topk']

        uid_tensor = dataset.inter_feat[dataset.uid_field].to(self.device)
        iid_tensor = dataset.inter_feat[dataset.iid_field].to(self.device)
        all_e_pre = []

        self.logger.info("Building e_pre for all users...")
        for user_id in tqdm(range(self.n_users), desc="e_pre construction"):
            # Find all items the user has interacted with
            interacted_items = iid_tensor[uid_tensor == user_id]
            if interacted_items.size(0) == 0:
                all_e_pre.append(torch.zeros(config["embedding_size"], device=self.device))
                continue
            
            # Sort interacted items based on a certain criterion (e.g., original model score)
            with torch.no_grad():
                if sort_mode == 'y_uio':
                    # Sort by the pre-calculated original scores
                    sort_values = self.y_uio_all[user_id, interacted_items.cpu()].to(self.device)
                else:
                    raise ValueError(f"Unknown e_pre sort mode: {sort_mode}")
            
            sorted_indices = torch.argsort(sort_values, descending=True)
            
            # Select a subset of items (e.g., top 30%)
            num_to_select = max(1, int(len(interacted_items) * top_k_ratio))
            if select_mode == 'top':
                selected_indices = sorted_indices[:num_to_select]
            else:
                raise ValueError(f"Unknown e_pre select mode: {select_mode}")
            
            selected_items = interacted_items[selected_indices]
            selected_item_embeds = self.item_embeds_orig[selected_items]
            
            # Aggregate the embeddings of selected items (e.g., by taking the mean)
            normalized_embeds = F.normalize(selected_item_embeds, p=2, dim=1)
            if agg_mode == 'mean':
                e_pre_user = torch.mean(normalized_embeds, dim=0)
            else:
                raise ValueError(f"Unknown e_pre aggregation mode: {agg_mode}")

            # Normalize the final e_pre vector for the user
            all_e_pre.append(F.normalize(e_pre_user, p=2, dim=0))
            
        return torch.stack(all_e_pre, dim=0)

    def _get_modified_user_embeddings(self, users, for_pos_str, for_neg_str):
        """
        Computes the modified user embeddings for positive and negative items in a BPR pair.
        The modification depends on the `loss_combination` config.
        e_u' = e_u_orig + alpha * norm * e_pop + beta * norm * e_pre
        """
        user_embeds_orig_batch = self.user_embeds_orig[users]
        user_norms = torch.norm(user_embeds_orig_batch, p=2, dim=1, keepdim=True)

        # Calculate the popularity term (alpha * e_pop)
        alpha_term = self.alphas[users].unsqueeze(1) * user_norms * self.popularity_direction_norm
        
        # Calculate the preference term (beta * e_pre)
        beta_term = 0.0
        if self.use_epre:
            beta_term = self.betas[users].unsqueeze(1) * user_norms * self.e_pre_all_users[users]

        # Helper to select terms based on the loss_combination string (e.g., 'a', 'b', 'ab')
        def get_term(term_char):
            if term_char == 'a': return alpha_term
            if term_char == 'b': return beta_term
            return 0.0

        # Construct embeddings for the positive item interaction
        u_embeds_pos = user_embeds_orig_batch
        for term in for_pos_str:
            u_embeds_pos = u_embeds_pos + get_term(term)

        # Construct embeddings for the negative item interaction
        u_embeds_neg = user_embeds_orig_batch
        for term in for_neg_str:
            u_embeds_neg = u_embeds_neg + get_term(term)

        return u_embeds_pos, u_embeds_neg

    def forward(self):
        """
        Computes the final user embeddings for prediction/evaluation.
        The `prediction_mode` config determines which debiasing components are included.
        """
        user_norms = torch.norm(self.user_embeds_orig, p=2, dim=1, keepdim=True)
        alpha_term = self.alphas.unsqueeze(1) * user_norms * self.popularity_direction_norm
        
        beta_term = 0.0
        if self.use_epre:
            beta_term = self.betas.unsqueeze(1) * user_norms * self.e_pre_all_users

        # Modify embeddings based on the prediction mode
        if self.prediction_mode == "full": # Original + Pop + Pref
            modified_user_embeddings = self.user_embeds_orig + alpha_term + beta_term
        elif self.prediction_mode == "no_pop": # Original + Pref
            modified_user_embeddings = self.user_embeds_orig + beta_term
        elif self.prediction_mode == "only_pop": # Original + Pop
            modified_user_embeddings = self.user_embeds_orig + alpha_term
        else: # Default to original if mode is unknown
            modified_user_embeddings = self.user_embeds_orig
            
        # Item embeddings are not modified
        return modified_user_embeddings, self.item_embeds_orig

    def calculate_loss(self, interaction):
        """
        Calculates the BPR loss using the modified user embeddings.
        Also includes a regularization loss on the learned alpha and beta parameters.
        """
        # Clear cache from full-sort prediction
        if self.restore_user_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
            
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Parse the loss combination string (e.g., 'a_b' -> for_pos='a', for_neg='b')
        parts = self.loss_combination.split('_')
        for_pos_str, for_neg_str = (parts[0], parts[1]) if len(parts) == 2 else ('ab', 'ab')
        if 'none' in for_pos_str: for_pos_str = ''
        if 'none' in for_neg_str: for_neg_str = ''
        
        # Get the modified user embeddings for positive and negative items
        u_embeds_pos, u_embeds_neg = self._get_modified_user_embeddings(user, for_pos_str, for_neg_str)

        # Get original item embeddings
        pos_embeds = self.item_embeds_orig[pos_item]
        neg_embeds = self.item_embeds_orig[neg_item]

        # Calculate scores
        pos_scores = torch.mul(u_embeds_pos, pos_embeds).sum(dim=1)
        neg_scores = torch.mul(u_embeds_neg, neg_embeds).sum(dim=1)
        
        # Calculate BPR loss
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Calculate L2 regularization loss for the learned parameters
        reg_loss = self.alphas[user].norm(2).pow(2)
        if self.use_epre:
            reg_loss += self.betas[user].norm(2).pow(2)
        reg_loss = reg_loss / user.size(0)

        # Combine losses
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, interaction):
        """Calculates prediction scores for a batch of user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        Calculates scores for a user against all items (for full ranking evaluation).
        Uses a cache to avoid recomputing the full user/item embeddings on each call.
        """
        user = interaction[self.USER_ID]
        # Compute and cache the full embeddings if not already done for this evaluation step
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = self.restore_user_e[user]
        # Score is the dot product of user embeddings with all item embeddings
        scores = torch.matmul(u_embeddings, self.restore_item_e.t())
        return scores.view(-1)


if __name__ == '__main__':
    # --- Argument Parsing for Finetuning Script ---
    parser = argparse.ArgumentParser(description="Run MF-DDC Finetuning")
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--dataset_index', '-di', type=int, default=0, help='Index of the dataset to use.')
    parser.add_argument('--use_epre', action='store_true', help='Enable learning with personalized preference direction (e_pre).')
    parser.add_argument('--epre_sort_mode', type=str, default='y_uio', help='Sorting mode for e_pre construction.')
    parser.add_argument('--epre_select_mode', type=str, default='top', help='Selection mode for e_pre construction.')
    parser.add_argument('--epre_agg_mode', type=str, default='mean', help='Aggregation mode for e_pre construction.')
    parser.add_argument('--epre_topk', type=float, default=0.2, help='Proportion of items to select for e_pre (e.g., 0.3 for top 30%%).')
    parser.add_argument('--pred_mode', type=str, default='full', choices=['full', 'no_pop', "only_pop"], help="Prediction embedding mode.")
    parser.add_argument('--loss_combination', type=str, default='a_a', help="Defines user embedding modification in loss. Format: 'pos_neg'.")
    parser.add_argument('--reg_weight', type=float, default=0.0001, help='L2 regularization weight for alphas and betas.')
    parser.add_argument('--stopping_step', type=int, default=10, help='Early stopping patience.')
    args = parser.parse_args()

    # --- Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Load config from the pre-trained model's checkpoint
    print("Loading pretrained model and data...")
    checkpoint = torch.load(DEFAULT_MODEL_FILE[args.dataset_index])
    config = checkpoint['config']
    
    # Update config with new parameters for the finetuning task
    config['eval_step'] = 5
    config['stopping_step'] = args.stopping_step
    config['gpu_id'] = 0 # The device is already set by CUDA_VISIBLE_DEVICES
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["prediction_mode"] = args.pred_mode
    config['checkpoint_dir'] = './saved_finetune/'
    config["epochs"] = 50000
    config['use_epre'] = args.use_epre
    config['epre_sort_mode'] = args.epre_sort_mode
    config['epre_select_mode'] = args.epre_select_mode
    config['epre_agg_mode'] = args.epre_agg_mode
    config['epre_topk'] = args.epre_topk
    config['loss_combination'] = args.loss_combination
    config['reg_weight'] = args.reg_weight
    
    # Initialize logger and seed
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Finetuning arguments: \n{args}\n")
    
    # --- Data and Model Loading ---
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Load the pre-trained model architecture and state
    pretrained_model = get_model(config["model"])(config, train_data.dataset).cpu()
    pretrained_model.load_state_dict(checkpoint["state_dict"])
    pretrained_model.load_other_parameter(checkpoint.get("other_parameter"))
    logger.info("Pretrained model loaded successfully.")

    # Load the pre-calculated popularity direction
    logger.info("Loading and normalizing popularity direction...")
    with open(DEFAULT_REP_DIRECTION_FILE[args.dataset_index], 'r') as f:
        rep_direction_data = json.load(f)
    # The JSON is expected to be a list of dicts; we extract the direction from the second entry
    rep_direction = rep_direction_data[1]["value"]["0"]
    popularity_direction = torch.tensor(rep_direction, dtype=torch.float).squeeze(0)
    popularity_direction_norm = F.normalize(popularity_direction, p=2, dim=0)
    logger.info(f"Popularity direction vector loaded. Shape: {popularity_direction_norm.shape}")

    # --- Model Initialization ---
    logger.info("Initializing the new model for finetuning...")
    finetune_model = MFDebiasFinetune(
        config=config,
        dataset=train_data.dataset,
        pretrained_model=pretrained_model,
        popularity_direction_norm=popularity_direction_norm
    ).to(config['device'])
    logger.info(finetune_model)
    
    # Log the number of trainable parameters (should be n_users for alphas + n_users for betas)
    trainable_params = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
    expected_params = finetune_model.n_users * (1 + (1 if args.use_epre else 0))
    logger.info(f"Number of trainable parameters: {trainable_params} (Expected: {expected_params})")

    # --- Training and Evaluation ---
    logger.info("Starting the finetuning process...")
    trainer = Trainer(config, finetune_model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    
    logger.info("Finetuning finished. Evaluating on the test set...")
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    # --- Final Results ---
    logger.info(set_color('best valid result', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
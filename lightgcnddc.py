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
from datetime import datetime
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


# =====================================================================================
# 1. Define the New Model for Secondary Training (Fine-tuning)
# =====================================================================================

class LightGCNDebiasFinetune(GeneralRecommender):
    """
    A specialized class for advanced fine-tuning of a pre-trained LightGCN model.
    It freezes the parameters of the original LightGCN and learns personalized 
    alpha and beta coefficients for each user.
    - alpha: Adjusts the user's offset along the predefined "popularity direction".
    - beta:  Adjusts the user's offset along the dynamically constructed "personalized preference direction".
    
    This model implements various configurable loss weighting schemes and training strategies.
    """
    def __init__(self, config, dataset, pretrained_model, popularity_direction_norm):
        super(LightGCNDebiasFinetune, self).__init__(config, dataset)
        self.logger = getLogger()
        
        # Load and freeze the pre-trained model
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval() # Ensure consistent behavior (e.g., disable dropout)

        self.random_pop_direction = config['random_pop_direction']
        self.random_pre_direction = config['random_pre_direction']

        # Store the popularity direction vector
        self.popularity_direction_norm = popularity_direction_norm.to(self.device)
        if self.random_pop_direction:
            self.popularity_direction_norm = F.normalize(torch.randn(config["embedding_size"]).to(self.device), p=2, dim=0)
            # self.popularity_direction_norm = (-1) * popularity_direction_norm.to(self.device)
        
        # ------------------- Define New Learnable Parameters -------------------
        # alpha: For the popularity direction
        self.alphas = nn.Parameter(torch.FloatTensor(self.n_users).uniform_(-0.5, 0.5))
        # beta: For the personalized preference direction (if enabled)
        self.use_epre = config['use_epre']
        self.not_use_normalization = config['no_epre_normalization']
        self.use_all_item = config['use_all_item']
        if self.use_epre:
            self.betas = nn.Parameter(torch.FloatTensor(self.n_users).uniform_(-0.5, 0.5))
        
        # ------------------- Caching and Pre-calculation -------------------
        self.logger.info("Pre-calculating embeddings and score components...")
        with torch.no_grad():
            # 1. Retrieve original, final embeddings
            self.pretrained_model.norm_adj_matrix = self.pretrained_model.norm_adj_matrix.cpu()
            self.user_embeds_orig, self.item_embeds_orig = self.pretrained_model.forward()
            self.user_embeds_orig, self.item_embeds_orig = self.user_embeds_orig.to(self.device), self.item_embeds_orig.to(self.device)

            if self.use_epre:
                self.logger.info("Constructing personalized preference direction (e_pre)...")
                self.e_pre_all_users = self._construct_e_pre(dataset, config).to(self.device)
        self.logger.info("Pre-calculation finished.")

        # ------------------- Loss and Configuration -------------------
        self.mf_loss = BPRLoss()
        self.prediction_mode = config["prediction_mode"]
        self.gamma_mode = 0 # config['gamma_mode']
        self.loss_combination = config['loss_combination'] # e.g., 'ab_ab', 'a_none', 'b_ab'
        self.reg_weight = config["reg_weight"]
        self.neg_sample_scalar = config["neg_sample_scalar"]
        
        # Used for evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None


    def _decompose_embeddings(self):
        """Decompose original embeddings into components parallel and orthogonal to the popularity direction."""
        e_pop = self.popularity_direction_norm
        
        # Decompose user embeddings
        proj_user = torch.matmul(self.user_embeds_orig, e_pop)
        self.user_embeds_pop = torch.outer(proj_user, e_pop)
        self.user_embeds_prefer = self.user_embeds_orig - self.user_embeds_pop
        
        # Decompose item embeddings
        proj_item = torch.matmul(self.item_embeds_orig, e_pop)
        self.item_embeds_pop = torch.outer(proj_item, e_pop)
        self.item_embeds_prefer = self.item_embeds_orig - self.item_embeds_pop


    def _construct_e_pre(self, dataset, config):
        """Construct the personalized preference direction (e_pre) for all users efficiently."""
        # Get configuration
        sort_mode = config['epre_sort_mode']
        select_mode = config['epre_select_mode']
        agg_mode = config['epre_agg_mode']
        top_k_ratio = config['epre_topk']

        # [Modification]: 1. Efficiently retrieve user interaction history using a sparse matrix.
        # RecBole's dataset object allows easy export of the interaction matrix.
        user_item_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        all_e_pre = []
        
        # If sorting by popularity is required, move the popularity tensor to GPU in advance.
        if sort_mode == 'pop':
            item_degree = self._get_item_degree(dataset).to(self.device)

        for user_id in tqdm(range(self.n_users), desc="Constructing e_pre"):
            # [Modification]: 2. Fast retrieval of the interaction item list for a single user.
            start = user_item_matrix.indptr[user_id]
            end = user_item_matrix.indptr[user_id + 1]
            interacted_item_indices = user_item_matrix.indices[start:end]
            
            if len(interacted_item_indices) == 0:
                # Use a zero vector if the user has no interactions.
                all_e_pre.append(torch.zeros(config["embedding_size"], device=self.device))
                continue

            interacted_items = torch.from_numpy(interacted_item_indices).long().to(self.device)
            
            # [Modification]: 3. Calculate sorting scores on the GPU in real-time.
            with torch.no_grad():
                user_embed = self.user_embeds_orig[user_id]
                interacted_item_embeds = self.item_embeds_orig[interacted_items]

                if sort_mode == 'pop':
                    sort_values = item_degree[interacted_items]
                elif sort_mode == 'y_uio':
                    # Calculate scores only for the user and their interacted items on the GPU.
                    sort_values = torch.matmul(user_embed, interacted_item_embeds.t())
                # elif sort_mode == 'cosine': # If cosine is needed, it can be calculated as follows:
                #     user_embed_norm = F.normalize(user_embed.unsqueeze(0))
                #     item_embeds_norm = F.normalize(interacted_item_embeds)
                #     sort_values = torch.matmul(user_embed_norm, item_embeds_norm.t()).squeeze(0)
                else:
                    raise ValueError(f"Unknown e_pre sort mode: {sort_mode}")
            
            # Sorting, selection, and aggregation logic remains the same but executes efficiently on the GPU.
            sorted_indices = torch.argsort(sort_values, descending=True)
            
            num_to_select = int(len(interacted_items) * top_k_ratio)
            if num_to_select == 0: num_to_select = 1

            if select_mode == 'top':
                selected_indices = sorted_indices[:num_to_select]
            elif select_mode == 'bottom':
                selected_indices = sorted_indices[-num_to_select:]
            else:
                raise ValueError(f"Unknown e_pre select mode: {select_mode}")
            
            selected_item_embeds = interacted_item_embeds[selected_indices]
            
            # Normalize each item embedding before combination
            selected_item_embeds = F.normalize(selected_item_embeds, p=2, dim=1)

            if agg_mode == 'mean':
                e_pre_user = torch.mean(selected_item_embeds, dim=0)
            elif agg_mode == 'sort_weight':
                weights = F.softmax(sort_values[selected_indices], dim=0).unsqueeze(1)
                e_pre_user = torch.sum(selected_item_embeds * weights, dim=0)
            elif agg_mode == 'sort_inv_weight':
                weights = F.softmax(1.0 / (sort_values[selected_indices] + 1e-9), dim=0).unsqueeze(1)
                e_pre_user = torch.sum(selected_item_embeds * weights, dim=0)
            else:
                raise ValueError(f"Unknown e_pre aggregation mode: {agg_mode}")

            # Final normalization
            all_e_pre.append(F.normalize(e_pre_user, p=2, dim=0))
            
        return torch.stack(all_e_pre, dim=0)

    def _get_gamma(self, users, items):
        """Calculate the weighting coefficient gamma based on the gamma_mode."""
        mode = self.gamma_mode
        if mode == 0: # Original loss
            return 1.0
        else:
            raise ValueError(f"Unknown gamma_mode: {mode}")

        # return gamma.detach() # gamma does not participate in gradient calculation

    def _get_modified_user_embeddings(self, users, for_pos, for_neg):
        """Construct different user_embeddings for positive and negative samples based on the loss_combination configuration."""
        user_embeds_orig_batch = self.user_embeds_orig[users]
        user_norms = torch.norm(user_embeds_orig_batch, p=2, dim=1, keepdim=True)

        alpha_term = self.alphas[users].unsqueeze(1) * user_norms * self.popularity_direction_norm
        
        beta_term = 0.0
        if self.use_epre:
            beta_term = self.betas[users].unsqueeze(1) * user_norms * self.e_pre_all_users[users]

        def get_term(term_char):
            if term_char == 'a': return alpha_term
            if term_char == 'b': return beta_term
            return 0.0

        u_embeds_pos = user_embeds_orig_batch
        for term in for_pos:
            u_embeds_pos = u_embeds_pos + get_term(term)

        u_embeds_neg = user_embeds_orig_batch
        for term in for_neg:
            u_embeds_neg = u_embeds_neg + get_term(term)

        return u_embeds_pos, u_embeds_neg

    def _get_item_degree(self, dataset):
        """Get item popularity (number of interactions)."""
        iid_field = dataset.iid_field
        iid_tensor = dataset.inter_feat[iid_field]
        item_counts = torch.bincount(iid_tensor, minlength=self.n_items)
        return item_counts

    def forward(self):
        """
        Forward pass for final prediction.
        The final user embedding is always: e_orig + alpha_term + beta_term.
        """
        user_norms = torch.norm(self.user_embeds_orig, p=2, dim=1, keepdim=True)
        alpha_term = self.alphas.unsqueeze(1) * user_norms * self.popularity_direction_norm
        
        beta_term = 0.0
        if self.use_epre:
            beta_term = self.betas.unsqueeze(1) * user_norms * self.e_pre_all_users
        if self.prediction_mode == "full":    
            modified_user_embeddings = self.user_embeds_orig + alpha_term + beta_term
        if self.prediction_mode == "no_pop":    
            modified_user_embeddings = self.user_embeds_orig + beta_term
        if self.prediction_mode == "only_pop":    
            modified_user_embeddings = self.user_embeds_orig + alpha_term
        return modified_user_embeddings, self.item_embeds_orig

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
            
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        # 1. Construct user embeddings for loss calculation
        # loss_combination = 'ab_ab' -> for_pos='ab', for_neg='ab'
        # loss_combination = 'a_b'   -> for_pos='a', for_neg='b'
        # loss_combination = 'none_a' -> for_pos='', for_neg='a'
        parts = self.loss_combination.split('_')
        for_pos_str, for_neg_str = (parts[0], parts[1]) if len(parts) == 2 else ('ab', 'ab')
        if 'none' in for_pos_str: for_pos_str = ''
        if 'none' in for_neg_str: for_neg_str = ''
        
        u_embeds_pos, u_embeds_neg = self._get_modified_user_embeddings(user, for_pos_str, for_neg_str)

        # 2. Retrieve item embeddings
        pos_embeds = self.item_embeds_orig[pos_item]
        neg_embeds = self.item_embeds_orig[neg_item]

        # 3. Compute scores
        pos_scores = torch.mul(u_embeds_pos, pos_embeds).sum(dim=1)
        neg_scores = torch.mul(u_embeds_neg, neg_embeds).sum(dim=1)
        
        # 4. Calculate and apply gamma coefficients
        gamma_pos = self._get_gamma(user, pos_item)
        gamma_neg = self._get_gamma(user, neg_item) * self.neg_sample_scalar
        
        # 5. Compute BPR Loss
        mf_loss = self.mf_loss(gamma_pos * pos_scores, gamma_neg * neg_scores)

        # 6. Compute Regularization Loss (only for learnable alpha and beta parameters)
        reg_loss = self.alphas[user].norm(2).pow(2)
        if self.use_epre:
            reg_loss += self.betas[user].norm(2).pow(2)
        reg_loss = reg_loss / user.size(0)

        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.t())
        return scores.view(-1)


# =====================================================================================
# 2. Main Execution Script
# =====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Advanced Debiased LightGCN Finetuning")
    
    # --- Basic Settings ---
    # parser.add_argument('--model_file', type=str, default=DEFAULT_MODEL_FILE, help='Path to the pretrained LightGCN model file.')
    # parser.add_argument('--rep_dir_file', type=str, default=DEFAULT_REP_DIRECTION_FILE, help='Path to the popularity direction json file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--dataset_index', type=int, default=0)


    # --- Loss Weighting (Gamma) ---
    parser.add_argument('--gamma_mode', type=int, default=0, choices=range(23), 
                        help='Gamma coefficient mode (0-22). 0 means no re-weighting.')
    
    # --- Personalized Preference Direction (e_pre & beta) ---
    parser.add_argument('--use_epre', action='store_true', help='Enable learning with personalized preference direction (e_pre).')
    parser.add_argument('--not_use_normalization', action='store_true')
    parser.add_argument('--use_all_item', action='store_true')
    parser.add_argument('--random_pop_direction', action='store_true')
    parser.add_argument('--random_pre_direction', action='store_true')
    parser.add_argument('--epre_sort_mode', type=str, default='pop', choices=['pop', 'y_uio', 'y_uiprefer', 'y_uipop', 'pop_ratio', 'cosine'], help='Sorting mode for e_pre construction.')
    parser.add_argument('--epre_select_mode', type=str, default='top', choices=['top', 'bottom'], help='Selection mode for e_pre construction.')
    parser.add_argument('--epre_agg_mode', type=str, default='mean', choices=['mean', 'orig_weight', 'sort_weight', 'sort_inv_weight'], help='Aggregation mode for e_pre construction.')
    parser.add_argument('--epre_topk', type=float, default=0.2, help='Proportion of items to select for e_pre (e.g., 0.2 for top 20%%).')
    parser.add_argument('--neg_sample_scalar', type=float, default=1.0)
    parser.add_argument(
        '--pred_mode', 
        type=str, 
        default='full', 
        choices=['full', 'no_pop', "only_pop"],
        help="Prediction embedding mode: 'full' (e_orig+alpha_term+e_pref) or 'no_pop' (e_orig+e_pref)"
    )

    # --- Loss Combination ---
    parser.add_argument('--loss_combination', type=str, default='a_a', 
                        help="Defines user embedding modification in loss. Format: 'pos_neg'. 'a' for alpha, 'b' for beta. E.g., 'a_a', 'ab_ab', 'b_none'.")

    # --- Training Hyperparameters ---
    parser.add_argument('--reg_weight', type=float, default=0.0001, help='L2 regularization weight for alphas.')

    parser.add_argument('--stopping_step', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--seed', '-s', type=int, default=2020, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # -- 1. Load Pre-trained Model and Data --
    logger = getLogger()
    logger.info("Loading pretrained model and data...")
    checkpoint = torch.load(DEFAULT_MODEL_FILE[args.dataset_index])
    config = checkpoint['config']
    
    # Update config to adapt to the fine-tuning task
    config['eval_step'] = 5
    config['stopping_step'] = args.stopping_step
    config['gpu_id'] = 0
    config["device"] = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    config["prediction_mode"] = args.pred_mode  # Set prediction mode from command line arguments
    config['checkpoint_dir'] = './saved_finetune/'
    config["epochs"] = 50000
    # Inject our own parameters into config for internal model access
    config['gamma_mode'] = args.gamma_mode
    config['use_epre'] = args.use_epre
    config['no_epre_normalization'] = args.not_use_normalization
    config['random_pop_direction'] = args.random_pop_direction
    config['random_pre_direction'] = args.random_pre_direction
    config['use_all_item'] = args.use_all_item
    config['epre_sort_mode'] = args.epre_sort_mode
    config['epre_select_mode'] = args.epre_select_mode
    config['epre_agg_mode'] = args.epre_agg_mode
    config['epre_topk'] = args.epre_topk
    config['loss_combination'] = args.loss_combination
    config['reg_weight'] = args.reg_weight
    config['neg_sample_scalar'] = args.neg_sample_scalar
    config['seed'] = 2020
    config['metric_decimal_place'] = 5

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Finetuning arguments: \n{args}\n")
    logger.info(f"New finetuning task started with prediction mode: {args.pred_mode}")
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)


    # # To sort by 'pop' for e_pre, item popularity needs to be calculated
    # item_degrees = torch.tensor(train_data.item_df['item_id'].value_counts().sort_index().values, dtype=torch.float32).to(config['device'])
    # config['item_degrees'] = item_degrees
    config['seed'] = args.seed
    init_seed(config['seed'], config['reproducibility'])
    pretrained_model = get_model(config["model"])(config, train_data.dataset).cpu()
    pretrained_model.load_state_dict(checkpoint["state_dict"])
    pretrained_model.load_other_parameter(checkpoint.get("other_parameter"))
    logger.info("Pretrained LightGCN model loaded successfully.")

    # -- 2. Load and Process Popularity Direction Vector --
    logger.info("Loading and normalizing popularity direction...")
    with open(DEFAULT_REP_DIRECTION_FILE[args.dataset_index], 'r') as f:
        rep_direction_data = json.load(f)
    
    rep_direction = rep_direction_data[1]["value"]["4"]
    popularity_direction = torch.tensor(rep_direction, dtype=torch.float).squeeze(0)
    popularity_direction_norm = F.normalize(popularity_direction, p=2, dim=0)
    logger.info(f"Popularity direction vector loaded. Shape: {popularity_direction_norm.shape}")

    # -- 3. Initialize the New Fine-tuning Model --
    logger.info("Initializing the new model for finetuning...")
    finetune_model = LightGCNDebiasFinetune(
        config=config,
        dataset=train_data.dataset,
        pretrained_model=pretrained_model,
        popularity_direction_norm=popularity_direction_norm
    ).to(config['device'])
    logger.info(finetune_model)
    
    trainable_params = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
    expected_params = finetune_model.n_users
    if args.use_epre:
        expected_params += finetune_model.n_users
    logger.info(f"Number of trainable parameters: {trainable_params} (Expected: {expected_params})")
    # assert trainable_params == expected_params, "Error: Incorrect number of trainable parameters!"

    # -- 4. Start the Fine-tuning Process --
    logger.info("Starting the finetuning process...")
    trainer = Trainer(config, finetune_model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    
    # -- 5. Evaluate the Final Model on the Test Set --
    logger.info("Finetuning finished. Evaluating on the test set...")
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    logger.info(set_color('best valid result', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    # -- 6. Save the Final Learned Coefficients --
    final_alphas = finetune_model.alphas.detach().cpu().numpy()
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    alpha_save_path = os.path.join(config['checkpoint_dir'], f'learned_alphas_{TIMESTAMP}.npy')
    np.save(alpha_save_path, final_alphas)
    logger.info(f"Learned alphas saved to {alpha_save_path}")

    if args.use_epre:
        final_betas = finetune_model.betas.detach().cpu().numpy()
        beta_save_path = os.path.join(config['checkpoint_dir'], f'learned_betas_{TIMESTAMP}.npy')
        np.save(beta_save_path, final_betas)
        logger.info(f"Learned betas saved to {beta_save_path}")
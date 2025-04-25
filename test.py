# Import required packages
from typing import Any, Dict, List, Tuple
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from anndata import AnnData
import logging
import scvi, scANVI
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from pereggrn import Pereggrn, load_data, preprocess_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configure logging and verbosity
sc.settings.verbosity = 3
sc.logging.print_header()

# Data Loading and Preprocessing
adata = sc.read_10x_mtx("path/to/mouse/data")
adata.var_names_make_unique()

def convert_mouse_to_human_gene_names(gene_list):
    return [gene.upper() for gene in gene_list]

# Filter cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Calculate quality control metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Convert gene names to human format
adata.var_names = convert_mouse_to_human_gene_names(adata.var_names)

# Split data into train, validation, and holdout sets
train_idx, temp_idx = train_test_split(
    np.arange(adata.n_obs),
    test_size=0.3,
    random_state=42,
    stratify=adata.obs['condition']
)

val_idx, holdout_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=adata[temp_idx].obs['condition']
)

train_data = adata[train_idx]
val_data = adata[val_idx]
holdout_data = adata[holdout_idx]

# Visualize highest expressed genes
sc.pl.highest_expr_genes(train_data, n_top=20)

# Define target genes
target_genes = ['MYC', 'AKT1', 'CYC', 'STAT3', 'BIP', 'JUN', 'FOS', 'Cox41', 'HIF1A', 'HSPA9']

# Initialize and train model
scvi.settings.seed = 0
scvi.model.SCVI.setup_anndata(train_data)

model = scvi.model.SCVI(
    train_data,
    n_layers=1,
    n_hidden=64,
    dropout_rate=0.1,
    n_latent=10
)

model.train(
    max_epochs=100,
    early_stopping=True,
    early_stopping_patience=5,
    batch_size=128,
    train_size=0.8,
    validation_size=0.2,
    plan_kwargs={
        'lr': 1e-3,
        'reduce_lr_on_plateau': True,
        'lr_patience': 3,
        'lr_factor': 0.5,
        'min_lr': 1e-5
    }
)

# Evaluate on validation set
val_latent = model.get_latent_representation(val_data)
val_predictions = model.get_normalized_expression(val_data)

# Evaluate on holdout set
holdout_latent = model.get_latent_representation(holdout_data)
holdout_predictions = model.get_normalized_expression(holdout_data)

# Differential Expression analysis
de_df = model.differential_expression(
    groupby="injury",
    genes_to_test=target_genes
)

de_df = de_df.join(adata.var[['gene_id']], how='inner')
de_df = de_df.sort_values('lfc_mean', ascending=False)

# Visualization
sc.pl.heatmap(train_data, 
              var_names=target_genes,
              groupby='treatment',
              show_gene_labels=True,
              figsize=(10,8))

# ContrastiveVI setup
scvi.external.ContrastiveVI.setup_anndata(train_data, layer="count")

contrastive_vi_model = scvi.external.ContrastiveVI(
    train_data, 
    n_salient_latent=10, 
    n_background_latent=10, 
    use_observed_lib_size=False
)

# MRVI Analysis
from scvi.external import MRVI

MRVI.setup_anndata(train_data, sample_key="treatment")
mrvi_model = MRVI(train_data)
mrvi_model.train(max_epochs=400)

# Get and visualize latent representations
u = mrvi_model.get_latent_representation()
u_mde = scvi.model.utils.mde(u)

# Visualize embeddings
sc.pl.embedding(
    train_data,
    basis="u_mde", 
    color=["cell_type", "injury", "tissue"],
    ncols=3,
    title=['Cell Type (MRVI)', 'Injury (MRVI)', 'Tissue (MRVI)']
)

# Differential expression and abundance analysis
de_results = mrvi_model.differential_expression(
    sample_cov_keys=["injury"],
    store_lfc=True
)

da_results = mrvi_model.differential_abundance(sample_cov_keys=["treatment"])

import logging
import scanpy as sc
import scgen
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from scgen import SCGEN
import sclambda_model
import models
import evaluate
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneformerDataset(Dataset):
    """Dataset for gene expression data using Geneformer."""
    def __init__(self, adata: AnnData, tokenizer: Any):
        self.adata = adata
        self.tokenizer = tokenizer
        self.gene_expression = adata.X.toarray()
        self.conditions = adata.obs['condition'].values
        
    def __len__(self) -> int:
        return len(self.adata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        expression = self.gene_expression[idx]
        condition = self.conditions[idx]
        
        # Convert expression to rank normalized values
        expression_rank = np.argsort(expression).argsort() / len(expression)
        
        # Tokenize using Geneformer's tokenizer
        tokens = self.tokenizer(
            expression_rank.tolist(),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'condition': torch.tensor(1 if condition == 'stimulated' else 0),
            'expression': torch.tensor(expression, dtype=torch.float32)
        }

class GeneformerModel(pl.LightningModule):
    """Lightning module for Geneformer-based gene expression prediction."""
    def __init__(self, model_name: str = "ctheodoris/Geneformer"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            output_hidden_states=True
        )
        self.classifier = torch.nn.Linear(768, 1)  # Assuming 768 is the hidden size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token
        return self.classifier(pooled_output)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.mse_loss(outputs, batch['expression'])
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.mse_loss(outputs, batch['expression'])
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

class ModelEnsemble:
    """Class for training and evaluating multiple models."""
    
    CONFIG = {
        'condition_key': 'condition',
        'cell_type_key': 'cell_type',
        'ctrl_key': 'control',
        'stim_key': 'stimulated',
        'pred_key': 'predict',
        'test_size': 0.2,
        'random_state': 42,
        'perturbation_types': ['knockout', 'overexpression'],
        'model_config': {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 6,
            'intermediate_size': 3072
        }
    }
    
    MODELS = {
        "Geneformer": GeneformerModel,
        "scGen": SCGEN,
        "scLAMBDA": sclambda_model.Model,
        "scPRAM": models.SCPRAM,
        "Geneformer": AutoModel.from_pretrained("ctheodoris/Geneformer")
    }
    
    @staticmethod
    def prepare_training_data(
        adata: sc.AnnData, 
        target_cell_type: str,
        perturbation_type: str = 'knockout'
    ) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
        """Prepare training, validation, and holdout data with proper splitting."""
        if perturbation_type not in all_models.CONFIG['perturbation_types']:
            raise ValueError(f"Perturbation type must be one of {all_models.CONFIG['perturbation_types']}")
            
        filtered_data = adata[~((adata.obs[all_models.CONFIG['cell_type_key']] == target_cell_type) &
                              (adata.obs[all_models.CONFIG['condition_key']] == all_models.CONFIG['stim_key']))]
        
        train_idx, temp_idx = train_test_split(
            np.arange(filtered_data.n_obs),
            test_size=0.3,
            random_state=all_models.CONFIG['random_state'],
            stratify=filtered_data.obs[all_models.CONFIG['cell_type_key']]
        )
        
        val_idx, holdout_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=all_models.CONFIG['random_state'],
            stratify=filtered_data[temp_idx].obs[all_models.CONFIG['cell_type_key']]
        )
        
        return filtered_data[train_idx], filtered_data[val_idx], filtered_data[holdout_idx]

    @staticmethod
    def train_and_evaluate(
        model: Any,
        model_name: str,
        train_data: sc.AnnData,
        val_data: sc.AnnData,
        holdout_data: sc.AnnData,
        target_cell_type: str,
        perturbation_type: str = 'knockout',
        epochs: int = 100
    ) -> Tuple[sc.AnnData, float, float]:
        """Train model and evaluate predictions on both validation and holdout sets."""
        try:
            if model_name == "Geneformer":
                # Initialize tokenizer and dataset
                tokenizer = AutoTokenizer.from_pretrained("ctheodoris/Geneformer")
                train_dataset = GenePerturbationDataset(train_data, tokenizer)
                val_dataset = GenePerturbationDataset(val_data, tokenizer)
                
                # Setup data loaders
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32)
                
                # Train model
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    callbacks=[
                        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                        pl.callbacks.ModelCheckpoint(monitor='val_loss')
                    ]
                )
                trainer.fit(model, train_loader, val_loader)
                
                # Generate predictions
                model.eval()
                with torch.no_grad():
                    pred = model.predict(train_loader)
                
            elif model_name == "scGen":
                model.train(
                    max_epochs=epochs,
                    batch_size=32,
                    early_stopping=True,
                    patience=10
                )
                pred = model.predict(perturbation_type=perturbation_type)
                
            elif model_name == "scLAMBDA":
                model.train(
                    epochs=epochs,
                    batch_size=32,
                    early_stopping=True,
                    patience=10
                )
                pred = model.predict(perturbation_type=perturbation_type)
                
            else:  # scPRAM
                model.train_SCPRAM(
                    train_data=train_data,
                    val_data=val_data,
                    epochs=epochs,
                    early_stopping=True,
                    patience=10
                )
                pred = model.predict(
                    train_adata=train_data,
                    cell_to_pred=target_cell_type,
                    key_dic=TrainableModels.CONFIG,
                    ratio=0.005,
                    perturbation_type=perturbation_type
                )
            
            # Evaluate on validation set
            val_ground_truth = val_data[val_data.obs[TrainableModels.CONFIG['cell_type_key']] == target_cell_type]
            val_eval_adata = val_ground_truth.concatenate(pred)
            val_score = evaluate.evaluate_adata(
                eval_adata=val_eval_adata,
                cell_type=target_cell_type,
                key_dic=TrainableModels.CONFIG
            )
            
            # Evaluate on holdout set
            holdout_ground_truth = holdout_data[holdout_data.obs[TrainableModels.CONFIG['cell_type_key']] == target_cell_type]
            holdout_eval_adata = holdout_ground_truth.concatenate(pred)
            holdout_score = evaluate.evaluate_adata(
                eval_adata=holdout_eval_adata,
                cell_type=target_cell_type,
                key_dic=TrainableModels.CONFIG
            )
            
            return pred, val_score, holdout_score
            
        except Exception as e:
            logging.error(f"Error during {model_name} training and evaluation: {str(e)}")
            raise

    @staticmethod
    def setup_model(model_name: str, adata: sc.AnnData) -> Any:
        """Initialize the specified model with the given data."""
        if model_name not in TrainableModels.MODELS:
            raise ValueError(f"Model {model_name} not supported")
        return TrainableModels.MODELS[model_name](adata)

if __name__ == "__main__":
    # Train and evaluate each model for both perturbation types
    for model_name in TrainableModels.MODELS.keys():
        for perturbation_type in TrainableModels.CONFIG['perturbation_types']:
            logging.info(f"Training {model_name} for {perturbation_type}...")
            
            # Initialize model
            model = TrainableModels.setup_model(model_name, train_new)
            
            # Prepare training and validation data
            train_data, val_data, holdout_data = TrainableModels.prepare_training_data(
                train_new, 
                perturbation_type=perturbation_type
            )
            
            # Train and evaluate
            pred, val_score, holdout_score = all_models.train_and_evaluate(
                model=model,
                model_name=model_name,
                train_data=train_data,
                val_data=val_data,
                holdout_data=holdout_data,
                perturbation_type=perturbation_type
            )
            
            logging.info(f"{model_name} {perturbation_type} validation score: {val_score}")
            logging.info(f"{model_name} {perturbation_type} holdout score: {holdout_score}")

            # Initialize both XGBoost and Random Forest multi-output models
            import xgboost as xgb
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            import numpy as np
            import shap
            
            # Prepare features and targets
            X = train_data.X.toarray()  # Initial expression profiles
            y = pred.X.toarray()  # Predicted expression profiles
            
            # Initialize and train XGBoost model
            xgb_model = xgb.XGBRegressor(
                tree_method="hist",
                multi_strategy="multi_output_tree",
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X, y)
            
            # Initialize and train Random Forest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            multi_rf = MultiOutputRegressor(rf)
            multi_rf.fit(X, y)
            
            # Get XGBoost feature importances using SHAP
            xgb_explainer = shap.TreeExplainer(xgb_model)
            xgb_shap_values = xgb_explainer.shap_values(X)
            xgb_mean_shap = np.abs(xgb_shap_values).mean(axis=0)
            
            # Get Random Forest feature importances
            rf_feature_importances = np.mean([estimator.feature_importances_ 
                                            for estimator in multi_rf.estimators_], axis=0)
            
            # Get top genes from both models
            xgb_top_genes_idx = np.argsort(xgb_mean_shap)[::-1][:10]
            rf_top_genes_idx = np.argsort(rf_feature_importances)[::-1][:10]
            
            xgb_top_genes = train_data.var_names[xgb_top_genes_idx]
            rf_top_genes = train_data.var_names[rf_top_genes_idx]
            
            # Log results from both models
            logging.info("Top 10 important genes identified by XGBoost SHAP values:")
            for gene, importance in zip(xgb_top_genes, xgb_mean_shap[xgb_top_genes_idx]):
                logging.info(f"{gene}: {importance:.4f}")
                
            logging.info("\nTop 10 important genes identified by Random Forest:")
            for gene, importance in zip(rf_top_genes, rf_feature_importances[rf_top_genes_idx]):
                logging.info(f"{gene}: {importance:.4f}")
                
            # Evaluate both models
            val_X = val_data.X.toarray()
            val_y = val_data.X.toarray()
            
            xgb_score = xgb_model.score(val_X, val_y)
            rf_score = multi_rf.score(val_X, val_y)
            
            logging.info(f"\nXGBoost R2 score: {xgb_score:.4f}")
            logging.info(f"Random Forest R2 score: {rf_score:.4f}")

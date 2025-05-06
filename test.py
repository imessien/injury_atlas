import os
import tempfile
import scanpy as sc
import scvi
import seaborn as sns
import torch
from rich import print
import logging
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scanpy import tl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from anndata.experimental.pytorch import AnnLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import Tuple
import numpy as np
import scanpy as sc
import optuna
import os
from transformers import BertForSequenceClassification, Trainer
from anndata.experimental.pytorch import AnnLoader

adata = sc.read_h5ad("/cis/net/r41/data/iessien1/Multi_Injury_Atlas.h5ad")
print(adata)

target_genes = [
    "MYC",
    "AKT1",
    "CYC",
    "STAT3",
    "BIP",
    "JUN",
    "FOS",
    "Cox41",
    "HIF1A",
    "HSPA9",
]

cell_states_to_model = {
    "cell_type": adata.obs["finalannotationv1"].unique(),
    "start_state": adata[adata.obs["Condition"] == "Control"][
        "finalannotationv1"
    ].unique()[0],
    "goal_state": adata[adata.obs["Condition"] == "Spinal Cord Injury"][
        "finalannotationv1"
    ].unique()[0],
    "alt_states": adata[adata.obs["Condition"] != "Spinal Cord Injury"][
        "finalannotationv1"
    ].unique(),
}

filter_data_dict = {"genes_to_perturb": target_genes}

# global imports
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from rules_extraction.plot import plot_accuracy, plot_frontier
from rules_extraction.rules import EnsembleRule, Rule, RuleRanker
from rules_extraction.utils import *
from scripts.custom_dataset import CustomDataset
from scripts.helpers import *
from scripts.models import FineTunedVGG
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("CUDA (GPU) not available. Using CPU.")

# load trained model VGG
model = FineTunedVGG()
model = model.to(device)
model.eval()

# Specify the map_location argument when loading the model
load_path = "models_weight/VGGSportsImageClassification.pth"
checkpoint = torch.load(load_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# get loaders  using function define in scripts for this notebook
train_loader, valid_loader, test_loader = get_dataloaders()

# get a dictionary that links class names to integers, making it easier for us as we proceed in the notebook
with open("./data/idx_to_names.json", "r") as file:
    idx_to_names = json.load(file)
names_to_idx = {v: k for k, v in idx_to_names.items()}


# create filtered loaders
model = model
loader = test_loader
device = device

# test set
correct_test_idx = filter_dataset(model=model, loader=loader, device=device)
test_filtered_dataset = Subset(test_loader.dataset, correct_test_idx)
test_filtered_dataloader = DataLoader(
    dataset=test_filtered_dataset, batch_size=test_loader.batch_size, shuffle=False
)

loader = train_loader
# train set
correct_train_idx = filter_dataset(model=model, loader=loader, device=device)
train_filtered_dataset = Subset(train_loader.dataset, correct_train_idx)
train_filtered_dataloader = DataLoader(
    dataset=train_filtered_dataset, batch_size=train_loader.batch_size, shuffle=False
)

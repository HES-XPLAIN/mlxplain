# global imports
import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from omnixai.data.image import Image as OmniImage
from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard
from requests.packages import target
from rules_extraction.plot import plot_accuracy, plot_frontier
from rules_extraction.rules import EnsembleRule, Rule, RuleRanker
from rules_extraction.utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from mlxplain.explainers.vision.specific.rulesextract import RulesExtractImage
from scripts.custom_dataset import CustomDataset
from scripts.helpers import *
from scripts.models import FineTunedVGG

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
train_loader, test_loader = get_dataloaders()  # todo: check valid_loader

# get a dictionary that links class names to integers, making it easier for us as we proceed in the notebook
with open("./data/idx_to_names.json", "r") as file:
    idx_to_names = json.load(file)
names_to_idx = {v: k for k, v in idx_to_names.items()}
class_names = list(idx_to_names.values())

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

### data transformation (image)
imgs = load_images_to_ndarray(
    "data"
)  # todo: use 'data', this is only a small subset for dev
image_data = OmniImage(
    data=imgs,
    batched=True,
)
# print(image_data)

np.random.seed(1)
# transformer = TabularTransform().fit(tabular_data)
# class_names = transformer.class_names
# x = transformer.transform(tabular_data)
x = image_data
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    x[:, :-1], x[:, -1], train_size=0.80
)
print("Training data shape: {}".format(train.shape))
print("Test data shape:     {}".format(test.shape))

### data transformation (tabular)

# data = np.genfromtxt(os.path.join("./data", "adult.data"), delimiter=", ", dtype=str)
# tabular_data = Tabular(
#     data,
#     feature_columns=feature_names,
#     categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
#     target_column="label",
# )
# print(tabular_data)

# np.random.seed(1)
# transformer = TabularTransform().fit(tabular_data)
# class_names = transformer.class_names
# x = transformer.transform(tabular_data)
# train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
#     x[:, :-1], x[:, -1], train_size=0.80
# )
# print("Training data shape: {}".format(train.shape))
# print("Test data shape:     {}".format(test.shape))
## data transformation end


# explainer = RulesExtractImage(
#     model=model,
#     dataloader=train_filtered_dataloader,
#     idx_to_names=idx_to_names,
#     target_class="air hockey",
#     top_rules=30,
#     mode="classification",
# )

# Apply an inverse transform, i.e., converting the numpy array back to `Tabular`
# test_instances = transformer.invert(test)
# test_x = test_instances[1653:1655]

# todo: is this necessary for images?
# Convert the transformed data back to Tabular instances
# train_data = transformer.invert(train)
# test_data = transformer.invert(test)

# Initialize a TabularExplainer
explainers = VisionExplainer(
    explainers=[ "rulesextract" ],  # The explainers to apply # "lime", "shap", "gradcam",
    mode="classification",  # The task type
    model=model,  # The ML model to explain
    # todo: check image transformer
    # preprocess=lambda z: transformer.transform(
    #     z
    # ),  # Converts raw features into the model inputs
    params={
        "rulesextract": {
            "dataloader": train_filtered_dataloader,
            "idx_to_names": idx_to_names,
            "target_class": "air hockey",
            "top_rules": 30,
        },
    }
)

# Generate explanations
# test_instances = test_data[:5] # todo: is this necessary for images (see above)?
test_instances = test[:5]
# local_explanations = explainers.explain(X=test_instances)
# global_explanations = explainer.explain()
global_explanations = explainers.explain_global()

# Launch a dashboard for visualization
dashboard = Dashboard(
    instances=None,  # The instances to explain
    local_explanations=None,  # Set the generated local explanations
    global_explanations=global_explanations,  # Set the generated global explanations
    class_names=class_names,  # Set class names
    # params={
    #     "rulesextract": {
    #         "dataloader": train_filtered_dataloader,
    #         "idx_to_names": idx_to_names,
    #         "target_class": "air hockey",
    #         "top_rules": 30,
    #     },
    # }
)
dashboard.show()  # Launch the dashboard

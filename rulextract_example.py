# global imports
import json

import numpy as np
import sklearn
import torch
from omnixai.data.image import Image as OmniImage
from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard
from rules_extraction.utils import filter_dataset
from torch.utils.data import DataLoader, Subset

from mlxplain.explainers.vision.specific.rulesextract import (  # noqa: F401
    RulesExtractImage,
)
from scripts.helpers import get_dataloaders, load_images_to_ndarray
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
load_path = "models_weight/VGGFineTuned.pth"
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

# data transformation (image)
imgs = load_images_to_ndarray("data")
image_data = OmniImage(
    data=imgs,
    batched=True,
)

np.random.seed(1)
x = image_data
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    x[:, :-1], x[:, -1], train_size=0.80
)
print("Training data shape: {}".format(train.shape))
print("Test data shape:     {}".format(test.shape))


# Initialize a TabularExplainer
explainers = VisionExplainer(
    explainers=["rulesextract"],  # The explainers to apply # "lime", "shap", "gradcam",
    mode="classification",  # The task type
    model=model,  # The ML model to explain
    # todo: check image transformer
    # preprocess=lambda z: transformer.transform(
    #     z
    # ),  # Converts raw features into the model inputs
    params={
        "rulesextract": {
            "dataloader": train_filtered_dataloader,
            "class_names": class_names,
            "target_class": "air hockey",
            "top_rules": 30,
        },
    },
)

# local_explanations = explainers.explain(X=test_instances)
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

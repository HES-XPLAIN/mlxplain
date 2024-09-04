#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Rules Extraction methods for vision tasks.
"""
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from omnixai_community.explainers.base import ExplainerBase
from omnixai_community.utils.misc import is_torch_available
from rules_extraction.rules import RuleRanker
from rules_extraction.utils import (
    compute_avg_features,
    extract_all_rules,
    make_target_df,
)
from torch.utils.data import DataLoader

from mlxplain.explanations.images.rules import RuleImportance


class RulesExtractImage(ExplainerBase):
    """
    The Rules Extraction method for generating visual explanations.
    If using this explainer, please cite `Improving neural network interpretability via rule extraction`
    https://doi.org/10.1007/978-3-030-01418-6
    """

    explanation_type = "global"
    alias = ["rulesextract"]

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        target_class: str,
        top_rules: int,
        mode: str = "classification",
        dataloader: Optional[DataLoader] = None,
        feature_activations: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `torch.nn.Module`.
        :param dataloader: The torch dataloader.
        :param class_names: The list of available classes
        :param target_class: The target class to extract rules for.
        :param top_rules: The number of rules to extract.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.target_class = target_class
        self.top_rules = top_rules
        self.mode = mode
        if dataloader is None and feature_activations is None:
            raise ValueError(
                "Either dataloader or feature_activations must be provided."
            )
        self.dataloader = dataloader
        self.feature_activations = feature_activations

        if not is_torch_available():
            raise EnvironmentError("Torch cannot be found.")

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"`model` should be a torch.nn.Module instead of {type(model)}"
            )

    def explain(self) -> RuleImportance:
        """
        Generates the explanations for the input instances.

        :return: The tuples explanations for all the instances, e.g., rules and their associated classes.
        :rtype: RuleImportance
        """
        if self.dataloader is not None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            index_to_classes = {
                str(index): name for index, name in enumerate(self.class_names)
            }
            avg_activations = compute_avg_features(
                self.model, self.dataloader, index_to_classes, device
            )
            target_feature_activations = make_target_df(
                avg_activations, self.target_class
            )
        else:
            if self.feature_activations is not None:
                feature_activations = self.feature_activations
                target_feature_activations = make_target_df(
                    feature_activations, self.target_class
                )

        columns_to_drop = ["binary_label", "label", "path"]
        existing_columns = [
            col for col in columns_to_drop if col in target_feature_activations.columns
        ]

        X = target_feature_activations.drop(existing_columns, axis=1)

        if "binary_label" in target_feature_activations.columns:
            y = target_feature_activations["binary_label"]
        elif "label" in target_feature_activations.columns:
            y = target_feature_activations["label"]
        else:
            raise ValueError(
                "Neither 'binary_label' nor 'label' column found in the DataFrame."
            )

        if y.empty:
            raise ValueError("The target column is empty.")
        all_rules = extract_all_rules(
            X, y, n_estimators=200, max_depth=2, random_state=1
        )
        explanations = RuleRanker(all_rules, X, y).rank_rules(N=self.top_rules)
        return RuleImportance(
            mode=self.mode, explanations=explanations, target_class=self.target_class
        )

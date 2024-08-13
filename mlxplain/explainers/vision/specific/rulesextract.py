#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Rules Extraction methods for vision tasks.
"""
import torch
import torch.nn as nn
from omnixai.explainers.base import ExplainerBase
from omnixai.utils.misc import is_torch_available
from rules_extraction.rules import RuleRanker
from rules_extraction.utils import (
    compute_avg_features,
    extract_all_rules,
    make_target_df,
)

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
        model,
        dataloader,
        class_names,
        target_class: str,
        top_rules: int,
        mode: str = "classification",
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
        self.dataloader = dataloader
        self.class_names = class_names
        self.target_class = target_class
        self.top_rules = top_rules
        self.mode = mode

        if not is_torch_available():
            # import torch.nn as nn
            raise EnvironmentError("Torch cannot be found.")

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"`model` should be a torch.nn.Module instead of {type(model)}"
            )

    def explain(self):
        """
        Generates the explanations for the input instances.

        :return: The tuples explanations for all the instances, e.g., rules and their associated classes.
        :rtype: RuleImportance
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        index_to_classes = {
            str(index): name for index, name in enumerate(self.class_names)
        }
        train_features = compute_avg_features(
            self.model, self.dataloader, index_to_classes, device
        )

        df_train = make_target_df(train_features, self.target_class)
        X_train, y_train = df_train.iloc[:, :-3], df_train.iloc[:, -1]

        all_rules = extract_all_rules(
            X_train, y_train, n_estimators=200, max_depth=2, random_state=1
        )

        explanations = RuleRanker(all_rules, X_train, y_train).rank_rules(
            N=self.top_rules
        )
        return RuleImportance(explanations=explanations)

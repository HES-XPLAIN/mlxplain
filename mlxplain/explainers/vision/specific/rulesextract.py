#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Rules Extraction methods for vision tasks.
"""
from typing import Callable

# TODO: clean up
import torch
import torch.nn as nn
from mlxplain.explanations.images.rules import RuleImportance
from omnixai.data.image import Image
from omnixai.explainers.base import ExplainerBase
from omnixai.explanations.image.plain import PlainExplanation
from omnixai.utils.misc import is_tf_available, is_torch_available
from rules_extraction.rules import EnsembleRule, RuleRanker
from rules_extraction.utils import (
    compute_avg_features,
    extract_all_rules,
    make_target_df,
)


class RulesExtractImage(ExplainerBase):
    """
    The Rules Extraction method for generating visual explanations.
    @TODO
    If using this explainer, please cite `Grad-CAM: Visual Explanations from Deep Networks
    https://doi.org/10.1007/978-3-030-01418-6
    """

    explanation_type = "global"
    alias = ["rulesextract"]

    # todo: compute_avg_features(model, test_filtered_dataloader, idx_to_names, device)
    def __init__(
        self,
        model,
        dataloader,
        idx_to_names,
        target_class: str,
        top_rules: int,
        mode: str = "classification",
        **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param target_layer: The target layer for explanation, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        # todo: doc
        super().__init__()
        self.model = model
        self.mode = mode
        self.dataloader = dataloader
        self.idx_to_names = idx_to_names
        self.target_class = target_class
        self.top_rules = top_rules

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

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        train_features = compute_avg_features(
            self.model, self.dataloader, self.idx_to_names, device
        )

        df_train = make_target_df(train_features, self.target_class)
        X_train, y_train = df_train.iloc[:, :-3], df_train.iloc[:, -1]
        # todo: check params hardcode
        all_rules = extract_all_rules(
            X_train, y_train, n_estimators=200, max_depth=2, random_state=1
        )



        rules = RuleRanker(all_rules, X_train, y_train).rank_rules(
            N=self.top_rules
        )
        explanations = RuleImportance(explanations=rules)
        print(explanations)
        return explanations

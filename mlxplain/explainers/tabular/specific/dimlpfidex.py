#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import json
import pathlib as pl
from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from dimlpfidex.dimlp import densCls, dimlpBT
from dimlpfidex.fidex import fidex, fidexGlo, fidexGloRules, fidexGloStats
from omnixai_community.data.tabular import Tabular
from omnixai_community.explainers.base import ExplainerBase
from trainings.gradBoostTrn import gradBoostTrn
from trainings.mlpTrn import mlpTrn
from trainings.randForestsTrn import randForestsTrn
from trainings.svmTrn import svmTrn

from ....explanations.tabular.dimlpfidex import DimlpfidexExplanation


def csv_to_list(path: str, sep=None) -> list[str]:
    res = []

    with open(path, "r") as f:
        for line in f:
            res += line.strip().split(sep)

    return res


def tabular_to_csv(data: Tabular, path: pl.Path) -> None:
    """
    Converts a Tabular object to a CSV file.

    :param data: The Tabular object containing the data.
    :param path: The path where the CSV file will be saved.
    """
    data.to_pd().to_csv(path, index=False, header=False)


def csv_to_tabular(path: str, sep=None) -> Tabular:
    """
    Loads a CSV file and converts it into a Tabular object.

    :param path: The path to the CSV file to load.
    :return: The Tabular object corresponding to the data in the CSV file.
    """
    df = pd.read_csv(path, sep=sep, index_col=False, header=None, skip_blank_lines=True)
    df.dropna(axis=1, how="all")
    res = Tabular(data=df)

    return res


def sanatize_list(data: list) -> str:
    """
    Converts a list into a string of a list without spaces.
    :param data: The list to convert.
    :return: The string corresponding to the list with spaces removed.
    """
    return str(data).replace(" ", "")


class DimlpfidexModel(metaclass=ABCMeta):
    """
    Abstract class providing a template for implementing a model that can be used by the DimlpfidexExplainer.
    """

    def __init__(self) -> None:
        super.__init__()

    @abstractmethod
    def _set_preprocess_function(self, preprocess_function: Callable = None):
        raise NotImplementedError("_set_preprocess_function() method not implemented.")

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError("_preprocess() method not implemented.")

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("__call__() method not implemented.")


class DimlpfidexAlgorithm(metaclass=ABCMeta):
    """
    Abstract class providing a template for implementing an explanation algorithm that can be used by the DimlpfidexExplainer.
    """

    def __init__(self) -> None:
        super.__init__()

    @abstractmethod
    def _postprocess(self) -> dict:
        raise NotImplementedError("_postprocess() method not implemented.")

    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError("execute() method not implemented.")


class DimlpBTModel(DimlpfidexModel):
    """
    A model class for the DimlpBT model, implementing the DimlpfidexModel interface.
    This model uses the DIMLP neural network with optional Dimlp rule extraction.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/dimlp/dimlpbt/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this model may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes.

    :param root_path: Directory where input files are located and output files will be generated.
    :param training_data: Tabular data containing attributes and classes for training.
    :param testing_data: Tabular data containing attributes and classes for testing.
    :param nb_attributes: Number of attributes in the data.
    :param nb_classes: Number of classes in the data.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param nb_dimlp_nets: Number of DIMLP networks to train with bagging.
    :param attributes_file: Optional file specifying attribute and class names.
    :param first_hidden_layer: Number of neurons in the first hidden layer.
    :param hidden_layers: List of integers specifying the number of neurons in each hidden layer, from the second to the last.
    :param with_rule_extraction: If True, rule extraction will be performed.
    :param momentum: Momentum parameter for training.
    :param flat: Flatness parameter for training.
    :param error_thresh: Error threshold to stop training.
    :param acc_thresh: Accuracy threshold to stop training.
    :param abs_error_thresh: Absolute error threshold for training.
    :param nb_epochs: Number of training epochs.
    :param nb_epochs_error: Number of epochs abefore showing error.
    :param nb_ex_per_net: Number of examples per network.
    :param nb_quant_levels: Number of "stairs" in the staircase function.
    :param normalization_file: File containing the mean and standard deviation for specified attributes that have been normalized.
    :param mus: Mean or median of each attribute index to denormalize in the rules.
    :param sigmas: Standard deviation of each attribute index to denormalize in the rules.
    :param normalization_indices: Indices of attributes to be denormalized in the rules.
    :param seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        root_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        verbose_console: bool = False,
        nb_dimlp_nets: int = 25,
        attributes_file: str = None,
        first_hidden_layer: int = None,
        hidden_layers: list[int] = None,
        with_rule_extraction: bool = False,
        momentum: float = 0.6,
        flat: float = 0.01,
        error_thresh: float = None,
        acc_thresh: float = None,
        abs_error_thresh: float = 0,
        nb_epochs: int = 1500,
        nb_epochs_error: int = 10,
        nb_ex_per_net: int = 0,
        nb_quant_levels: int = 50,
        normalization_file: str = None,
        mus: list[float] = None,
        sigmas: list[float] = None,
        normalization_indices: list[int] = None,
        seed: int = 0,
    ):
        self.root_path = root_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.verbose_console = verbose_console
        self.nb_dimlp_nets = nb_dimlp_nets
        self.attributes_file = attributes_file
        self.first_hidden_layer = first_hidden_layer
        if self.first_hidden_layer is None:
            self.first_hidden_layer = self.nb_attributes
        self.hidden_layers = hidden_layers
        self.with_rule_extraction = with_rule_extraction
        self.momentum = momentum
        self.flat = flat
        self.error_thresh = error_thresh
        self.acc_thresh = acc_thresh
        self.abs_error_thresh = abs_error_thresh
        self.nb_epochs = nb_epochs
        self.nb_epochs_error = nb_epochs_error
        self.nb_ex_per_net = nb_ex_per_net
        self.nb_quant_levels = nb_quant_levels
        self.normalization_file = normalization_file
        self.mus = mus
        self.sigmas = sigmas
        self.normalization_indices = normalization_indices
        if self.normalization_indices is None:
            self.normalization_indices = list(range(nb_attributes))
        self.seed = seed
        self.preprocess_function = None
        self.densCls_test_samples_file = None
        self.densCls_predictions_file = None
        self.has_predicted = False
        self.preprocess_function = None

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "dimlpBTTrain.out",
            "test_pred_outfile": "dimlpBTTest.out",
            "weights_outfile": "dimlpBT.wts",
            "hidden_layers_file": "hidden_layers.out",
        }

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.root_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.root_path.joinpath(self.test_data_filename)
        )

    def train(self):
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --nb_dimlp_nets {self.nb_dimlp_nets}
                    --with_rule_extraction {self.with_rule_extraction}
                    --momentum {self.momentum}
                    --flat {self.flat}
                    --abs_error_thresh {self.abs_error_thresh}
                    --nb_epochs {self.nb_epochs}
                    --nb_epochs_error {self.nb_epochs_error}
                    --nb_ex_per_net {self.nb_ex_per_net}
                    --nb_quant_levels {self.nb_quant_levels}
                    --seed {self.seed}
                    """
        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not self.verbose_console:
            command += " --console_file dimlpBTResult.txt"
        if self.first_hidden_layer is not None:
            command += f" --first_hidden_layer {self.first_hidden_layer}"
        if self.hidden_layers is not None:
            command += f" --hidden_layers {sanatize_list(self.hidden_layers)}"
        if self.with_rule_extraction:
            command += " --global_rules_outfile dimlpBTRules.rls"
        if self.error_thresh is not None:
            command += f" --error_thresh {self.error_thresh}"
        if self.acc_thresh is not None:
            command += f" --acc_thresh {self.acc_thresh}"
        if self.normalization_file is not None:
            command += f" --normalization_file {self.normalization_file}"
        if self.mus is not None:
            command += f" --mus {sanatize_list(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatize_list(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatize_list(self.normalization_indices)}"
            )

        status = dimlpBT(command)
        return status

    def __call__(self, data: Tabular) -> int:
        self.predict_test_samples_file = "densCls_test_samples.txt"
        self.predict_predictions_file = "densCls_test_preds.txt"

        if self.preprocess_function is not None:
            self.preprocess_function(self.predict_predictions_file)

        tabular_to_csv(data, self.root_path.joinpath(self.predict_test_samples_file))

        command = f"""
                            --root_folder {self.root_path}
                            --train_data_file {self.train_data_filename}
                            --weights_file {self._outputs["weights_outfile"]}
                            --test_data_file {self.predict_test_samples_file}
                            --test_pred_outfile {self.predict_predictions_file}
                            --hidden_layers_file {self._outputs["hidden_layers_file"]}
                            --nb_attributes {self.nb_attributes}
                            --nb_classes {self.nb_classes}
                            --console_file densClsResult.txt
                         """
        self.has_predicted = True
        status = densCls(command)

        if status != 0:
            print("Something went wrong with DensCls (DimlpBT prediction) execution...")

        results = []
        with open(
            self.root_path.joinpath(self.predict_predictions_file), "r"
        ) as predictions:
            for line in predictions:
                results.append([float(e) for e in line.strip().split(" ")])

        return np.array(results)


class GradBoostModel(DimlpfidexModel):
    """
    A model class for Gradient Boosting, implementing the DimlpfidexModel interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/training-methods/gradboosttrn/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this model may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes.

    :param root_path: Directory where input files are located and output files will be generated.
    :param training_data: Tabular data containing attributes and classes for training.
    :param testing_data: Tabular data containing attributes and classes for testing.
    :param nb_attributes: Number of attributes in the data.
    :param nb_classes: Number of classes in the data.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param n_estimators: Number of generated trees in the forest.
    :param loss: Loss function to optimize.
    :param learning_rate: Learning rate shrinks the contribution of each tree.
    :param subsample: The fraction of samples to use for fitting the individual base learners.
    :param criterion: The function to measure the quality of a split.
    :param max_depth: Maximum depth of the individual estimators.
    :param min_samples_split: Minimum number of samples required to split an internal node.
    :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
    :param min_weight_fraction_leaf: Minimum weighted fraction of the input samples required to be at a leaf node.
    :param max_features: The number of features to consider when looking for the best split.
    :param max_leaf_nodes: Maximum number of leaf nodes.
    :param min_impurity_decrease: A node will be split if this split induces a decrease in the impurity greater than or equal to this value.
    :param init: An estimator object that is used to compute the initial predictions.
    :param seed: Random seed for reproducibility.
    :param verbose_scikit: Controls the verbosity when fitting and predicting.
    :param warm_start: Reuse the solution of the previous call to fit and add more estimators to the ensemble.
    :param validation_fraction: The proportion of training data to set aside as validation set for early stopping.
    :param n_iter_no_change: Number of iterations with no improvement to stop fitting.
    :param tol: Tolerance for stopping criterion.
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
    """

    def __init__(
        self,
        root_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        verbose_console: bool = False,
        n_estimators: int = 100,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf: float = 0.0,
        max_features="sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        init: str = None,
        seed: int = None,
        verbose_scikit: int = 0,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        self.root_path = root_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.verbose_console = verbose_console
        self.seed = seed
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.seed = seed
        self.verbose_scikit = verbose_scikit
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.preprocess_function = None

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "predTrain.out",
            "test_pred_outfile": "predTest.out",
            "rules_outfile": "GB_rules.rls",
        }

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.root_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.root_path.joinpath(self.test_data_filename)
        )

    def train(self):
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --n_estimators {self.n_estimators}
                    --loss {self.loss}
                    --learning_rate {self.learning_rate}
                    --subsample {self.subsample}
                    --criterion {self.criterion}
                    --max_depth {self.max_depth}
                    --min_samples_leaf {self.min_samples_leaf}
                    --min_samples_split {self.min_samples_split}
                    --min_weight_fraction_leaf {self.min_weight_fraction_leaf}
                    --max_features {self.max_features}
                    --min_impurity_decrease {self.min_impurity_decrease}
                    --verbose {self.verbose_scikit}
                    --warm_start {self.warm_start}
                    --validation_fraction {self.validation_fraction}
                    --tol {self.tol}
                    --ccp_alpha {self.ccp_alpha}
                    """
        if not self.verbose_console:
            command += " --console_file GBResult.txt"
        if self.max_leaf_nodes is not None:
            command += f" --max_leaf_nodes {self.max_leaf_nodes}"
        if self.init is not None:
            command += f" --init {self.init}"
        if self.seed is not None:
            command += f" --seed {self.seed}"
        if self.n_iter_no_change is not None:
            command += f" --n_iter_no_change {sanatize_list(self.n_iter_no_change)}"

        status = gradBoostTrn(command)
        return status

    def __call__(self, data) -> None:
        print("Prediction is not available for this model.")
        return


class RandomForestModel(DimlpfidexModel):
    """
    A model class for Random Forest, implementing the DimlpfidexModel interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/training-methods/randforeststrn/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this model may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes.

    :param root_path: Directory where input files are located and output files will be generated.
    :param training_data: Tabular data containing attributes and classes for training.
    :param testing_data: Tabular data containing attributes and classes for testing.
    :param nb_attributes: Number of attributes in the data.
    :param nb_classes: Number of classes in the data.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param n_estimators: Number of trees in the forest.
    :param criterion: Function to measure the quality of a split.
    :param max_depth: Maximum depth of the tree.
    :param min_samples_split: Minimum number of samples required to split an internal node.
    :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
    :param min_weight_fraction_leaf: Minimum weighted fraction of the input samples required to be at a leaf node.
    :param max_features: Number of features to consider when looking for the best split.
    :param max_leaf_nodes: Maximum number of leaf nodes.
    :param min_impurity_decrease: A node will be split if this split induces a decrease in the impurity.
    :param bootstrap: Whether bootstrap samples are used when building trees.
    :param oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.
    :param n_jobs: Number of jobs to run in parallel.
    :param seed: Random seed for reproducibility.
    :param verbose_scikit: Controls the verbosity when fitting and predicting.
    :param warm_start: Reuse the solution of the previous call to fit and add more estimators to the ensemble.
    :param class_weight: Weights associated with classes.
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
    :param max_samples: If bootstrap is True, the number of samples to draw from X to train each base estimator.
    """

    def __init__(
        self,
        root_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        verbose_console: bool = False,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf: float = 0.0,
        max_features="sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        seed: int = None,
        verbose_scikit: int = 0,
        warm_start: bool = False,
        class_weight=None,
        ccp_alpha: float = 0.0,
        max_samples=None,
    ):
        self.root_path = root_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.verbose_console = verbose_console
        self.seed = seed
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose_scikit = verbose_scikit
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.preprocess_function = None

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "predTrain.out",
            "test_pred_outfile": "predTest.out",
            "rules_outfile": "RF_rules.rls",
        }

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.root_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.root_path.joinpath(self.test_data_filename)
        )

    def train(self) -> int:
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --n_estimators {self.n_estimators}
                    --criterion {self.criterion}
                    --min_samples_leaf {self.min_samples_leaf}
                    --min_samples_split {self.min_samples_split}
                    --min_weight_fraction_leaf {self.min_weight_fraction_leaf}
                    --max_features {self.max_features}
                    --min_impurity_decrease {self.min_impurity_decrease}
                    --bootstrap {self.bootstrap}
                    --oob_score {self.oob_score}
                    --n_jobs {self.n_jobs}
                    --verbose {self.verbose_scikit}
                    --warm_start {self.warm_start}
                    --ccp_alpha {self.ccp_alpha}
                    """
        if not self.verbose_console:
            command += " --console_file RFResult.txt"
        if self.max_depth is not None:
            command += f" --max_depth {self.max_depth}"
        if self.max_leaf_nodes is not None:
            command += f" --max_leaf_nodes {self.max_leaf_nodes}"
        if self.seed is not None:
            command += f" --seed {self.seed}"
        if self.class_weight is not None:
            if isinstance(self.class_weight, dict):
                self.class_weight = sanatize_list(self.class_weight)
            command += f" --class_weight {self.class_weight}"
        if self.max_samples is not None:
            command += f" --max_samples {self.max_samples}"

        status = randForestsTrn(command)
        return status

    def __call__(self, data) -> int:
        print("Prediction is not available for this model.")
        return


class SVMModel(DimlpfidexModel):
    """
    A model class for Support Vector Machines (SVM), implementing the DimlpfidexModel interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/training-methods/svmtrn/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this model may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes.

    :param root_path: Directory where input files are located and output files will be generated.
    :param training_data: Tabular data containing attributes and classes for training.
    :param testing_data: Tabular data containing attributes and classes for testing.
    :param nb_attributes: Number of attributes in the data.
    :param nb_classes: Number of classes in the data.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param output_roc: Path to the file where the ROC curve will be saved.
    :param positive_class_index: Index of the positive class for ROC calculation.
    :param nb_quant_levels: Number of "stairs" in the staircase function.
    :param K: Parameter to improve dynamics by normalizing input data.
    :param C: Regularization parameter.
    :param kernel: Specifies the kernel type to be used in the algorithm.
    :param degree: Degree of the polynomial kernel function (‘poly’).
    :param gamma: Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’.
    :param coef0: Independent term in kernel function.
    :param shrinking: Whether to use the shrinking heuristic.
    :param tol: Tolerance for stopping criterion.
    :param cache_size: Specify the size of the kernel cache.
    :param class_weight: Class balance.
    :param verbose_scikit: Controls the verbosity when fitting and predicting.
    :param max_iterations: Hard limit on iterations within solver.
    :param decision_function_shape: Decision function shape.
    :param break_ties: Whether to break tie decision for one-vs-rest decision function shape with more than 2 classes.
    """

    def __init__(
        self,
        root_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        verbose_console: bool = False,
        output_roc: str | None = None,
        positive_class_index: int = None,
        nb_quant_levels: int = 50,
        K: float = 1.0,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma="scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 0.001,
        cache_size: float = 200,
        class_weight=None,
        verbose_scikit: bool = False,
        max_iterations: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
    ):
        self.root_path = root_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.verbose_console = verbose_console
        self.output_roc = output_roc
        self.positive_class_index = positive_class_index
        self.nb_quant_levels = nb_quant_levels
        self.K = K
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose_scikit = verbose_scikit
        self.max_iterations = max_iterations
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.preprocess_function = None

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "predTrain.out",
            "test_pred_outfile": "predTest.out",
            "weights_outfile": "weights.wts",
        }

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.root_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.root_path.joinpath(self.test_data_filename)
        )

    def train(self):
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --nb_quant_levels {self.nb_quant_levels}
                    --K {self.K}
                    --C {self.C}
                    --kernel {self.kernel}
                    --degree {self.degree}
                    --gamma {self.gamma}
                    --coef0 {self.coef0}
                    --shrinking {self.shrinking}
                    --tol {self.tol}
                    --cache_size {self.cache_size}
                    --verbose {self.verbose_scikit}
                    --max_iterations {self.max_iterations}
                    --decision_function_shape {self.decision_function_shape}
                    --break_ties {self.break_ties}
                    """

        if not self.verbose_console:
            command += " --console_file SVMResult.txt"
        if self.positive_class_index is not None:
            command += f" --positive_class_index {self.positive_class_index}"
        if self.class_weight is not None:
            if isinstance(self.class_weight, dict):
                self.class_weight = sanatize_list(self.class_weight)
            command += f" --class_weight {self.class_weight}"
        if self.output_roc is not None:
            command += f" --output_roc {self.output_roc}"

        status = svmTrn(command)
        return status

    def __call__(self, data) -> int:
        print("Prediction is not available for this model.")
        return


class MLPModel(DimlpfidexModel):
    """
    A model class for Multilayer Perceptron (MLP), implementing the DimlpfidexModel interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/training-methods/mlptrn/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this model may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes.

    :param root_path: Directory where input files are located and output files will be generated.
    :param training_data: Tabular data containing attributes and classes for training.
    :param testing_data: Tabular data containing attributes and classes for testing.
    :param nb_attributes: Number of attributes in the data.
    :param nb_classes: Number of classes in the data.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param nb_quant_levels: Number of "stairs" in the staircase function.
    :param K: Parameter to improve dynamics by normalizing input data.
    :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    :param activation: Activation function for the hidden layer.
    :param solver: The solver for weight optimization.
    :param alpha: L2 penalty (regularization term) parameter.
    :param batch_size: Size of minibatches for stochastic optimizers.
    :param learning_rate: Learning rate schedule for weight updates.
    :param learning_rate_init: Initial learning rate used.
    :param power_t: The exponent for inverse scaling learning rate.
    :param max_iterations: Maximum number of iterations.
    :param shuffle: Whether to shuffle samples in each iteration.
    :param seed: Random seed for reproducibility.
    :param tol: Tolerance for the optimization.
    :param verbose_scikit: Controls the verbosity when fitting and predicting.
    :param warm_start: Reuse the solution of the previous call to fit and add more iterations to the model.
    :param momentum: Momentum for gradient descent update.
    :param nesterovs_momentum: Whether to use Nesterov’s momentum.
    :param early_stopping: Whether to use early stopping to terminate training when validation score is not improving.
    :param validation_fraction: Proportion of training data to set aside as validation set for early stopping.
    :param beta_1: Exponential decay rate for estimates of first moment vector in Adam.
    :param beta_2: Exponential decay rate for estimates of second moment vector in Adam.
    :param epsilon: Value for numerical stability in Adam.
    :param n_iter_no_change: Maximum number of epochs to not meet tol improvement.
    :param max_fun: Maximum number of loss function calls.
    """

    def __init__(
        self,
        root_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        verbose_console: bool = False,
        nb_quant_levels: int = 50,
        K: float = 1.0,
        hidden_layer_sizes: list[int] = [100],
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size="auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iterations: int = 200,
        shuffle: bool = True,
        seed: int = None,
        tol: float = 0.0001,
        verbose_scikit: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ):
        self.root_path = root_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.verbose_console = verbose_console
        self.nb_quant_levels = nb_quant_levels
        self.K = K
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iterations = max_iterations
        self.shuffle = shuffle
        self.seed = seed
        self.tol = tol
        self.verbose_scikit = verbose_scikit
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.preprocess_function = None

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "predTrain.out",
            "test_pred_outfile": "predTest.out",
            "weights_outfile": "weights.wts",
        }

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.root_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.root_path.joinpath(self.test_data_filename)
        )

    def train(self) -> int:
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --nb_quant_levels {self.nb_quant_levels}
                    --K {self.K}
                    --hidden_layer_sizes {sanatize_list(self.hidden_layer_sizes)}
                    --activation {self.activation}
                    --solver {self.solver}
                    --alpha {self.alpha}
                    --batch_size {self.batch_size}
                    --learning_rate {self.learning_rate}
                    --learning_rate_init {self.learning_rate_init}
                    --power_t {self.power_t}
                    --max_iterations {self.max_iterations}
                    --shuffle {self.shuffle}
                    --tol {self.tol}
                    --verbose {self.verbose_scikit}
                    --warm_start {self.warm_start}
                    --momentum {self.momentum}
                    --nesterovs_momentum {self.nesterovs_momentum}
                    --early_stopping {self.early_stopping}
                    --validation_fraction {self.validation_fraction}
                    --beta_1 {self.beta_1}
                    --beta_2 {self.beta_2}
                    --epsilon {self.epsilon}
                    --n_iter_no_change {self.n_iter_no_change}
                    --max_fun {self.max_fun}
                    """

        if not self.verbose_console:
            command += " --console_file MLPResult.txt"
        if self.seed is not None:
            command += f" --seed {self.seed}"

        status = mlpTrn(command)
        return status

    def __call__(self, data) -> int:
        print("Prediction is not available for this model.")
        return


class FidexAlgorithm(DimlpfidexAlgorithm):
    """
    An algorithm class for the Fidex explanation method, implementing the DimlpfidexAlgorithm interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/fidex/fidex/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this explanation algorithm may differ from this use case. Please use it with caution.
    - The `training_data` and `testing_data` must contain both attributes and classes altogether.

    :param model: The model to explain, which should be a subclass of DimlpfidexModel.
    :param kwargs: Dictionary of additional parameters that can be provided to configure the algorithm. The following keys are recognized:
        - verbose_console (bool, optional): If True, verbose output will be printed to the console, else it will be saved in a file.
        - attributes_file (str, optional): Optional file specifying attribute and class names.
        - max_iterations (int, optional): Maximum number of iterations to generate a rule. Defaults to 10.
        - min_covering (int, optional): Minimum number of examples a rule must cover. Defaults to 2.
        - covering_strategy (bool, optional): Whether or not the algorithm uses a dichotomic strategy to compute a rule. Defaults to True.
        - max_failed_attempts (int, optional): Maximum number of failed attempts to generate a rule. Defaults to 30.
        - min_fidelity (float, optional): Lowest fidelity score allowed for a rule. Defaults to 1.0.
        - lowest_min_fidelity (float, optional): Lowest fidelity score to which we agree to go down when a rule must be generated. Defaults to 0.75.
        - dropout_dim (float, optional): Percentage of dimensions that are ignored during an iteration. Defaults to 0.0.
        - dropout_hyp (float, optional): Percentage of hyperplanes that are ignored during an iteration. Defaults to 0.0.
        - decision_threshold (float, optional): Threshold for predictions to be considered as correct.
        - positive_class_index (int, optional): Index of the positive class for the usage of decision threshold.
        - nb_quant_levels (int, optional): Number of "stairs" in the staircase function. Defaults to 50.
        - normalization_file (str, optional): File containing the mean and standard deviation for specified attributes that have been normalized.
        - mus (list[float], optional): Mean or median of each attribute index to denormalize in the rules.
        - sigmas (list[float], optional): Standard deviation of each attribute index to denormalize in the rules.
        - normalization_indices (list[int], optional): Indices of attributes to be denormalized in the rules. Defaults to all attributes.
        - seed (int, optional): Random seed for reproducibility. Defaults to 0.
    """

    def __init__(self, model: DimlpfidexModel, **kwargs):

        self.model = model

        if self.model is None:
            raise TypeError(
                "A model must be provided when running FidexGloRules explainer."
            )

        self.nb_attributes = self.model.nb_attributes
        self.nb_classes = self.model.nb_classes

        if self.nb_attributes is None or self.nb_classes is None:
            raise TypeError(
                "The FidexGloRules explainer could not retreive the number of attributes or the number of classes from the given model."
            )

        self.rules_outfile = "fidex_output_rules.json"
        self.stats_file = "statsFidex.txt"
        self.test_samples_filename = "fidex_test_samples.txt"

        self.process_kwargs(**kwargs)

    def process_kwargs(self, **kwargs):
        self.verbose_console = kwargs.get("verbose_console", False)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.min_covering = kwargs.get("min_covering", 2)
        self.covering_strategy = kwargs.get("covering_strategy", True)
        self.attributes_file = kwargs.get("attributes_file", None)
        self.max_failed_attempts = kwargs.get("max_failed_attempts", 30)
        self.min_fidelity = kwargs.get("min_fidelity", 1.0)
        self.lowest_min_fidelity = kwargs.get("lowest_min_fidelity", 0.75)
        self.dropout_dim = kwargs.get("dropout_dim", 0.0)
        self.dropout_hyp = kwargs.get("dropout_hyp", 0.0)
        self.decision_threshold = kwargs.get("decision_threshold", None)
        self.positive_class_index = kwargs.get("positive_class_index", None)
        self.nb_quant_levels = kwargs.get("nb_quant_levels", 50)
        self.normalization_file = kwargs.get("normalization_file", None)
        self.mus = kwargs.get("mus", None)
        self.sigmas = kwargs.get("sigmas", None)
        self.normalization_indices = kwargs.get(
            "normalization_indices", list(range(self.nb_attributes))
        )
        self.seed = kwargs.get("seed", 0)

    def _postprocess(self) -> dict:
        absolute_path = self.model.root_path.joinpath(self.rules_outfile)
        try:
            with open(absolute_path) as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {absolute_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file at {absolute_path} is not a valid JSON.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def execute(self, test_data: Tabular) -> dict:
        tabular_to_csv(
            test_data, self.model.root_path.joinpath(self.test_samples_filename)
        )

        if self.model.has_predicted:
            test_preds_file = self.model.predict_predictions_file
        else:
            test_preds_file = self.model._outputs["test_pred_outfile"]

        command = f"""
                --root_folder {self.model.root_path}
                --train_data_file {self.model.train_data_filename}
                --train_pred_file {self.model._outputs["train_pred_outfile"]}
                --test_data_file {self.test_samples_filename}
                --test_pred_file {test_preds_file}
                --rules_outfile {self.rules_outfile}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                --stats_file {self.stats_file}
                --max_iterations {self.max_iterations}
                --min_covering {self.min_covering}
                --covering_strategy {self.covering_strategy}
                --max_failed_attempts {self.max_failed_attempts}
                --min_fidelity {self.min_fidelity}
                --lowest_min_fidelity {self.lowest_min_fidelity}
                --dropout_dim {self.dropout_dim}
                --dropout_hyp {self.dropout_hyp}
                --nb_quant_levels {self.nb_quant_levels}
                --seed {self.seed}
                """
        if "weights_outfile" in self.model._outputs:
            command += f" --weights_file {self.model._outputs['weights_outfile']}"
        else:
            command += f" --rules_file {self.model._outputs['rules_outfile']}"
        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not self.verbose_console:
            command += " --console_file fidexResult.txt"
        if self.decision_threshold is not None:
            command += f" --decision_threshold {self.decision_threshold}"
        if self.positive_class_index is not None:
            command += f" --positive_class_index {self.positive_class_index}"
        if self.normalization_file is not None:
            command += f" --normalization_file {self.normalization_file}"
        if self.mus is not None:
            command += f" --mus {sanatize_list(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatize_list(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatize_list(self.normalization_indices)}"
            )

        status = fidex(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the Fidex explainer execution..."
            )

        return self._postprocess()


class FidexGloRulesAlgorithm(DimlpfidexAlgorithm):
    """
    An algorithm class for the FidexGloRules explanation method, implementing the DimlpfidexAlgorithm interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/fidex/fidexglorules/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this explanation algorithm may differ from this use case. Please use it with caution.
    - The `training_data` must contain both attributes and classes altogether.
    - The algorithm can execute the FidexGlo if the `with_fidexGlo` parameter is True. The documentation for fidexGlo can be found here :
    https://hes-xplain.github.io/documentation/algorithms/fidex/fidexglo/
    - Parameters `with_minimal_version` and `nb_fidex_rules` are only used with FidexGlo.

    :param model: The model to explain, which should be a subclass of DimlpfidexModel.
    :param kwargs: Dictionary of additional parameters that can be provided to configure the algorithm. The following keys are recognized:
        - heuristic (int, optional): The heuristic to use for rule generation. Defaults to 1.
        - with_fidexGlo (bool, optional): If True, FidexGlo will also be executed. Defaults to False.
        - fidexGlo (dict, optional): If `with_fidexGlo` is True, you can pass an additional dictionary with the key `'fidexGlo'`, containing keyword arguments to configure the FidexGlo algorithm. Refer to the `FidexGloAlgorithm` class documentation for detailed information on the available parameters.
        - verbose_console (bool, optional): If True, verbose output will be printed to the console, else it will be saved in a file. Defaults to False.
        - attributes_file (str, optional): Optional file specifying attribute and class names.
        - max_iterations (int, optional): Maximum number of iterations to generate a rule. Defaults to 10.
        - min_covering (int, optional): Minimum number of examples a rule must cover. Defaults to 2.
        - covering_strategy (bool, optional): Whether or not the algorithm uses a dichotomic strategy to compute a rule. Defaults to True.
        - max_failed_attempts (int, optional): Maximum number of failed attempts to generate a rule. Defaults to 30.
        - min_fidelity (float, optional): Lowest fidelity score allowed for a rule. Defaults to 1.0.
        - lowest_min_fidelity (float, optional): Lowest fidelity score to which we agree to go down when a rule must be generated. Defaults to 0.75.
        - dropout_dim (float, optional): Percentage of dimensions that are ignored during an iteration. Defaults to 0.0.
        - dropout_hyp (float, optional): Percentage of hyperplanes that are ignored during an iteration. Defaults to 0.0.
        - decision_threshold (float, optional): Threshold for predictions to be considered as correct.
        - positive_class_index (int, optional): Index of the positive class for the usage of decision threshold.
        - nb_quant_levels (int, optional): Number of "stairs" in the staircase function. Defaults to 50.
        - normalization_file (str, optional): File containing the mean and standard deviation for specified attributes that have been normalized.
        - mus (list[float], optional): Mean or median of each attribute index to denormalize in the rules.
        - sigmas (list[float], optional): Standard deviation of each attribute index to denormalize in the rules.
        - normalization_indices (list[int], optional): Indices of attributes to be denormalized in the rules. Defaults to all attributes.
        - nb_threads (int, optional): Number of threads to use for processing. Defaults to 1.
        - seed (int, optional): Random seed for reproducibility. Defaults to 0.
        - with_minimal_version (bool, optional): Whether to use the minimal version, which only gets correct activated rules. Defaults to False.
        - nb_fidex_rules (int, optional): Number of Fidex rules to compute per sample when launching the Fidex algorithm. Defaults to 1.
    """

    # - train_data must contain attributes and classes altogether
    # - You can ask for the execution of fidexGlo with the parameter fidexGlo
    # - parameters with_minimal_version and nb_fidex_rules only used with fidexGlo

    def __init__(self, model: DimlpfidexModel, **kwargs):
        self.model = model
        if self.model is None:
            raise TypeError(
                "A model must be provided when running FidexGloRules explainer."
            )

        self.nb_attributes = self.model.nb_attributes
        self.nb_classes = self.model.nb_classes

        if self.nb_attributes is None or self.nb_classes is None:
            raise TypeError(
                "The FidexGloRules explainer could not retreive the number of attributes or the number of classes from the given model."
            )

        self.fidexGlo = None
        self.global_rules_outfile = "fidexGloRules_output_rules.json"

        self.process_kwargs(**kwargs)

    def process_kwargs(self, **kwargs):
        self.heuristic = kwargs.get("heuristic", 1)
        self.attributes_file = kwargs.get("attributes_file", None)
        self.with_fidexGlo = kwargs.get("with_fidexGlo", False)
        if self.with_fidexGlo:
            self.fidexGlo = FidexGloAlgorithm(self.model, **kwargs)
        self.verbose_console = kwargs.get("verbose_console", False)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.min_covering = kwargs.get("min_covering", 2)
        self.covering_strategy = kwargs.get("covering_strategy", True)
        self.max_failed_attempts = kwargs.get("max_failed_attempts", 30)
        self.min_fidelity = kwargs.get("min_fidelity", 1.0)
        self.lowest_min_fidelity = kwargs.get("lowest_min_fidelity", 0.75)
        self.dropout_dim = kwargs.get("dropout_dim", 0.0)
        self.dropout_hyp = kwargs.get("dropout_hyp", 0.0)
        self.decision_threshold = kwargs.get("decision_threshold", None)
        self.positive_class_index = kwargs.get("positive_class_index", None)
        self.nb_quant_levels = kwargs.get("nb_quant_levels", 50)
        self.normalization_file = kwargs.get("normalization_file", None)
        self.mus = kwargs.get("mus", None)
        self.sigmas = kwargs.get("sigmas", None)
        self.normalization_indices = kwargs.get(
            "normalization_indices", list(range(self.nb_attributes))
        )
        self.nb_threads = kwargs.get("nb_threads", 1)
        self.seed = kwargs.get("seed", 0)
        self.with_minimal_version = kwargs.get("with_minimal_version", False)
        self.nb_fidex_rules = kwargs.get("nb_fidex_rules", 1)

    def _postprocess(self) -> dict:
        absolute_path = self.model.root_path.joinpath(self.global_rules_outfile)
        try:
            with open(absolute_path) as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {absolute_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file at {absolute_path} is not a valid JSON.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def execute(self) -> dict:
        command = f"""
                --root_folder {self.model.root_path}
                --train_data_file {self.model.train_data_filename}
                --train_pred_file {self.model._outputs["train_pred_outfile"]}
                --global_rules_outfile {self.global_rules_outfile}
                --heuristic {self.heuristic}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                --max_iterations {self.max_iterations}
                --min_covering {self.min_covering}
                --covering_strategy {self.covering_strategy}
                --max_failed_attempts {self.max_failed_attempts}
                --min_fidelity {self.min_fidelity}
                --lowest_min_fidelity {self.lowest_min_fidelity}
                --dropout_dim {self.dropout_dim}
                --dropout_hyp {self.dropout_hyp}
                --nb_quant_levels {self.nb_quant_levels}
                --nb_threads {self.nb_threads}
                --seed {self.seed}
                """
        if "weights_outfile" in self.model._outputs:
            command += f" --weights_file {self.model._outputs['weights_outfile']}"
        else:
            command += f" --rules_file {self.model._outputs['rules_outfile']}"
        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not self.verbose_console:
            command += " --console_file fidexGloRulesResult.txt"
        if self.decision_threshold is not None:
            command += f" --decision_threshold {self.decision_threshold}"
        if self.positive_class_index is not None:
            command += f" --positive_class_index {self.positive_class_index}"
        if self.normalization_file is not None:
            command += f" --normalization_file {self.normalization_file}"
        if self.mus is not None:
            command += f" --mus {sanatize_list(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatize_list(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:

            command += (
                f" --normalization_indices {sanatize_list(self.normalization_indices)}"
            )

        status = fidexGloRules(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the FidexGloRules explainer execution..."
            )

        stats = FidexGloStatsAlgorithm(
            model=self.model,
            verbose_console=self.verbose_console,
            attributes_file=self.attributes_file,
            positive_class_index=self.positive_class_index,
        )
        stats.execute()

        if self.fidexGlo is not None:
            self.fidexGlo.execute()

        return self._postprocess()


class FidexGloStatsAlgorithm(DimlpfidexAlgorithm):
    """
    An algorithm class for calculating statistics from FidexGlo rules, implementing the DimlpfidexAlgorithm interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/fidex/fidexglostats/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this explanation algorithm may differ from this use case. Please use it with caution.
    - The `test_data_filename` must contain both attributes and classes altogether.

    :param model: The model to explain, which should be a subclass of DimlpfidexModel.
    :param verbose_console: If True, verbose output will be printed to the console, else it will be saved in a file.
    :param attributes_file: Optional file specifying attribute and class names.
    :param positive_class_index: Index of positive class to compute true/false positive/negative rates.
    """

    def __init__(
        self,
        model: DimlpfidexModel,
        verbose_console: bool = False,
        attributes_file: str = None,
        positive_class_index: int = None,
    ):
        self.model = model
        self.verbose_console = verbose_console
        self.nb_attributes = model.nb_attributes
        self.nb_classes = model.nb_classes
        self.global_rules_file = "fidexGloRules_output_rules.json"
        self.attributes_file = attributes_file
        self.stats_file = "statsFidexGloRules.txt"
        self.positive_class_index = positive_class_index

    def _postprocess(self):
        pass

    def execute(self):
        command = f"""
                --root_folder {self.model.root_path}
                --test_data_file {self.model.test_data_filename}
                --test_pred_file {self.model._outputs["test_pred_outfile"]}
                --global_rules_file {self.global_rules_file}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                --stats_file {self.stats_file}
                """
        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not self.verbose_console:
            command += " --console_file fidexGloStatsResult.txt"
        if self.positive_class_index is not None:
            command += f" --positive_class_index {self.positive_class_index}"

        status = fidexGloStats(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the FidexGloStats explainer execution..."
            )

        return None


class FidexGloAlgorithm(DimlpfidexAlgorithm):
    """
    An algorithm class for the FidexGlo explanation method, implementing the DimlpfidexAlgorithm interface.
    The documentation for this model can be found here:
    https://hes-xplain.github.io/documentation/algorithms/fidex/fidexglo/

    **Important Notes:**
    - HES-XPLAIN documentation concerning this explanation algorithm may differ from this use case. Please use it with caution.
    - `training_data` and `testing_data` must contain both attributes and classes altogether.

    :param model: The model to explain, which should be a subclass of DimlpfidexModel.
    :param kwargs: Additional parameters for configuring the algorithm. The following keys are recognized:
        - verbose_console (bool, optional): If True, verbose output will be printed to the console; otherwise, it will be saved in a file.
        - with_minimal_version (bool, optional): Whether to use the minimal version, which only retrieves correctly activated rules.
        - attributes_file (str, optional): Path to an optional file specifying attribute and class names.
        - max_iterations (int, optional): Maximum number of iterations to generate a rule.
        - min_covering (int, optional): Minimum number of examples a rule must cover.
        - covering_strategy (bool, optional): Whether or not the algorithm uses a dichotomic strategy to compute a rule.
        - max_failed_attempts (int, optional): Maximum number of failed attempts to generate a rule.
        - min_fidelity (float, optional): Lowest fidelity score allowed for a rule.
        - lowest_min_fidelity (float, optional): Lowest fidelity score to which we agree to go down when a rule must be generated.
        - nb_fidex_rules (int, optional): Number of Fidex rules to compute per sample when launching the Fidex algorithm.
        - dropout_dim (float, optional): Percentage of dimensions that are ignored during an iteration.
        - dropout_hyp (float, optional): Percentage of hyperplanes that are ignored during an iteration.
        - nb_quant_levels (int, optional): Number of "stairs" in the staircase function.
        - normalization_file (str, optional): File containing the mean and standard deviation for specified attributes that have been normalized.
        - mus (list[float], optional): Mean or median of each attribute index to denormalize in the rules.
        - sigmas (list[float], optional): Standard deviation of each attribute index to denormalize in the rules.
        - normalization_indices (list[int], optional): Indices of attributes to be denormalized in the rules.
        - seed (int, optional): Random seed for reproducibility.
    """

    def __init__(self, model: DimlpfidexModel, **kwargs):
        self.model = model

        if self.model is None:
            raise TypeError(
                "A model must be provided when running FidexGloRules explainer."
            )

        self.nb_attributes = self.model.nb_attributes
        self.nb_classes = self.model.nb_classes

        if self.nb_attributes is None or self.nb_classes is None:
            raise TypeError(
                "The FidexGloRules explainer could not retreive the number of attributes or the number of classes from the given model."
            )

        self.global_rules_file = "fidexGloRules_output_rules.json"
        self.explanation_file = "explanations.json"

        self.process_kwargs(**kwargs)

    def process_kwargs(self, **kwargs):
        kwargs = kwargs.get("fidexGlo", {})
        self.verbose_console = kwargs.get("verbose_console", False)
        self.with_minimal_version = kwargs.get("with_minimal_version", False)
        self.attributes_file = kwargs.get("attributes_file", None)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.min_covering = kwargs.get("min_covering", 2)
        self.covering_strategy = kwargs.get("covering_strategy", True)
        self.max_failed_attempts = kwargs.get("max_failed_attempts", 30)
        self.min_fidelity = kwargs.get("min_fidelity", 1.0)
        self.lowest_min_fidelity = kwargs.get("lowest_min_fidelity", 0.75)
        self.nb_fidex_rules = kwargs.get("nb_fidex_rules", 1)
        self.dropout_dim = kwargs.get("dropout_dim", 0.0)
        self.dropout_hyp = kwargs.get("dropout_hyp", 0.0)
        self.nb_quant_levels = kwargs.get("nb_quant_levels", 50)
        self.normalization_file = kwargs.get("normalization_file", None)
        self.mus = kwargs.get("mus", None)
        self.sigmas = kwargs.get("sigmas", None)
        self.normalization_indices = kwargs.get(
            "normalization_indices", list(range(self.nb_attributes))
        )
        self.seed = kwargs.get("seed", 0)

    def _postprocess(self) -> dict:
        absolute_path = self.model.root_path.joinpath(self.explanation_file)
        try:
            with open(absolute_path) as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {absolute_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file at {absolute_path} is not a valid JSON.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def execute(self) -> None:
        command = f"""
                --root_folder {self.model.root_path}
                --train_data_file {self.model.train_data_filename}
                --train_pred_file {self.model._outputs["train_pred_outfile"]}
                --test_data_file {self.model.test_data_filename}
                --test_pred_file {self.model._outputs["test_pred_outfile"]}
                --with_fidex True
                --global_rules_file {self.global_rules_file}
                --explanation_file {self.explanation_file}
                --with_minimal_version {self.with_minimal_version}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                --max_iterations {self.max_iterations}
                --min_covering {self.min_covering}
                --covering_strategy {self.covering_strategy}
                --max_failed_attempts {self.max_failed_attempts}
                --min_fidelity {self.min_fidelity}
                --lowest_min_fidelity {self.lowest_min_fidelity}
                --nb_fidex_rules {self.nb_fidex_rules}
                --dropout_dim {self.dropout_dim}
                --dropout_hyp {self.dropout_hyp}
                --nb_quant_levels {self.nb_quant_levels}
                --seed {self.seed}
                """
        if "weights_outfile" in self.model._outputs:
            command += f" --weights_file {self.model._outputs['weights_outfile']}"
        else:
            command += f" --rules_file {self.model._outputs['rules_outfile']}"
        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not self.verbose_console:
            command += " --console_file fidexGloRulesResult.txt"
        if self.normalization_file is not None:
            command += f" --normalization_file {self.normalization_file}"
        if self.mus is not None:
            command += f" --mus {sanatize_list(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatize_list(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatize_list(self.normalization_indices)}"
            )

        status = fidexGlo(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the FidexGlo explainer execution..."
            )

        # return self._postprocess() #TODO nice to have JSON file format (TODO in C++)


class FidexExplainer(ExplainerBase):
    """
    FidexExplainer is a local explanation class for the 'DimlpBTModel' model.

    This class provides a method to generate local explanations for classification tasks.
    It uses the Fidex algorithm and integrates with a given 'DimlpBTModel' model.
    """

    alias = ["fidex"]
    mode = "classification"
    explanation_type = "local"

    def __init__(
        self,
        training_data: Tabular,
        model: DimlpBTModel,
        preprocess_function: Callable = None,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(model, DimlpBTModel):
            raise RuntimeError(
                "Cannot use the FidexExplainer with another model than a DimlpBTModel."
            )

        self.model = model
        self.training_data = training_data
        self.preprocess_function = preprocess_function
        self.algorithm = FidexAlgorithm(model, **kwargs)
        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self, X: Tabular) -> DimlpfidexExplanation:
        """
        Generates local explanation(s) using the specified explanation algorithm.

        :param X: Tabular test data for which to generate explanations.
        :return: A DimlpfidexExplanation object containing the explanations.
        """

        self.model(X)
        result = self.algorithm.execute(X)
        attributes = None
        classes = None

        if self.algorithm.attributes_file is not None:
            labels = csv_to_list(
                self.model.root_path.joinpath(self.algorithm.attributes_file)
            )

            attributes = labels[: self.algorithm.nb_attributes]
            classes = labels[self.algorithm.nb_attributes :]

        return DimlpfidexExplanation(
            self.mode, self.training_data.shape[0], result, attributes, classes
        )


class FidexGloRulesExplainer(ExplainerBase):
    """
    FidexGloRulesExplainer is a global explanation class for models of type DimlpfidexModel.

    This class provides a method to generate global explanations for classification tasks.
    It uses the FidexGloRules algorithm to generate global explanations based on rules.
    """

    alias = ["fidexGloRules"]
    mode = "classification"
    explanation_type = "global"

    def __init__(
        self,
        training_data: Tabular,
        model: DimlpfidexModel,
        preprocess_function: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.training_data = training_data
        self.preprocess_function = preprocess_function
        self.algorithm = FidexGloRulesAlgorithm(model, **kwargs)
        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self):
        """
        Generates global explanations using the specified explanation algorithm.

        :return: A DimlpfidexExplanation object containing the global explanations.
        """

        result = self.algorithm.execute()
        attributes = None
        classes = None

        if self.algorithm.attributes_file is not None:
            labels = csv_to_list(
                self.model.root_path.joinpath(self.algorithm.attributes_file)
            )

            attributes = labels[: self.algorithm.nb_attributes]
            classes = labels[self.algorithm.nb_attributes :]

        return DimlpfidexExplanation(
            self.mode, self.training_data.shape[0], result, attributes, classes
        )

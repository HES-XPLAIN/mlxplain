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

import pandas as pd
from dimlpfidex.dimlp import dimlpBT
from dimlpfidex.fidex import fidex, fidexGlo, fidexGloRules, fidexGloStats
from omnixai.data.tabular import Tabular
from omnixai.explainers.base import ExplainerBase
from trainings.gradBoostTrn import gradBoostTrn
from trainings.mlpTrn import mlpTrn
from trainings.randForestsTrn import randForestsTrn
from trainings.svmTrn import svmTrn

from ....explanations.tabular.dimlpfidex import DimlpfidexExplanation


def tabular_to_csv(data: Tabular, path: pl.Path) -> None:
    data.to_pd().to_csv(path, index=False, header=False)


def csv_to_tabular(path: str) -> Tabular:
    return Tabular(pd.read_csv(path))


def sanatizeList(data: list) -> str:
    return str(data).replace(" ", "")


class DimlpfidexModel(metaclass=ABCMeta):
    """
    Abstract class giving a template to follow when implementing a model that can be used by the DimlpfidexExplainer.
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
    Abstract class giving a template to follow when implementing an explaining algorithm that can be used by the DimlpfidexExplainer.
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
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - root_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data and test_data must contain attributes and classes altogether

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

        # values to be known for further output manipulation
        self._outputs = {
            "train_pred_outfile": "dimlpBTTrain.out",
            "test_pred_outfile": "dimlpBTTest.out",
            "weights_outfile": "dimlpBT.wts",
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

    def __call__(self, data) -> int:
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
            command += f" --hidden_layers {sanatizeList(self.hidden_layers)}"
        if self.with_rule_extraction:
            command += " --global_rules_outfile dimlpBTRules.rls"
        if self.error_thresh is not None:
            command += f" --error_thresh {self.error_thresh}"
        if self.acc_thresh is not None:
            command += f" --acc_thresh {self.acc_thresh}"
        if self.normalization_file is not None:
            command += f" --normalization_file {self.normalization_file}"
        if self.mus is not None:
            command += f" --mus {sanatizeList(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatizeList(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatizeList(self.normalization_indices)}"
            )

        status = dimlpBT(command)
        return status


class GradBoostModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - root_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data and test_data must contain attributes and classes altogether

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

    def __call__(self, data) -> int:
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
            command += f" --n_iter_no_change {sanatizeList(self.n_iter_no_change)}"

        status = gradBoostTrn(command)
        return status


class RandomForestModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - root_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data and test_data must contain attributes and classes altogether

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

    def __call__(self, data) -> int:
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
                self.class_weight = sanatizeList(self.class_weight)
            command += f" --class_weight {self.class_weight}"
        if self.max_samples is not None:
            command += f" --max_samples {self.max_samples}"

        status = randForestsTrn(command)
        return status


class SVMModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - root_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data and test_data must contain attributes and classes altogether

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

    def __call__(self, data) -> int:
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
                self.class_weight = sanatizeList(self.class_weight)
            command += f" --class_weight {self.class_weight}"
        if self.output_roc is not None:
            command += f" --output_roc {self.output_roc}"

        status = svmTrn(command)
        return status


class MLPModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - root_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data and test_data must contain attributes and classes altogether

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

    def __call__(self, data) -> int:
        self._preprocess()

        command = f"""
                    --root_folder {self.root_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --nb_quant_levels {self.nb_quant_levels}
                    --K {self.K}
                    --hidden_layer_sizes {sanatizeList(self.hidden_layer_sizes)}
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


class FidexAlgorithm(DimlpfidexAlgorithm):
    # - train_data and test_data must contain attributes and classes altogether

    def __init__(
        self,
        model: DimlpfidexModel,
        verbose_console: bool = False,
        attributes_file: str = None,
        max_iterations: int = 10,
        min_covering: int = 2,
        covering_strategy: bool = True,
        max_failed_attempts: int = 30,
        min_fidelity: float = 1.0,
        lowest_min_fidelity: float = 0.75,
        dropout_dim: float = 0.0,
        dropout_hyp: float = 0.0,
        decision_threshold: float = None,
        positive_class_index: int = None,
        nb_quant_levels: int = 50,
        normalization_file: str = None,
        mus: list[float] = None,
        sigmas: list[float] = None,
        normalization_indices: list[int] = None,
        seed: int = 0,
    ):
        self.model = model
        self.verbose_console = verbose_console
        self.nb_attributes = model.nb_attributes
        self.nb_classes = model.nb_classes
        self.rules_outfile = "fidex_output_rules.json"
        self.attributes_file = attributes_file
        self.stats_file = "statsFidex.txt"
        self.max_iterations = max_iterations
        self.min_covering = min_covering
        self.covering_strategy = covering_strategy
        self.max_failed_attempts = max_failed_attempts
        self.min_fidelity = min_fidelity
        self.lowest_min_fidelity = lowest_min_fidelity
        self.dropout_dim = dropout_dim
        self.dropout_hyp = dropout_hyp
        self.decision_threshold = decision_threshold
        self.positive_class_index = positive_class_index
        self.nb_quant_levels = nb_quant_levels
        self.normalization_file = normalization_file
        self.mus = mus
        self.sigmas = sigmas
        self.normalization_indices = normalization_indices
        if self.normalization_indices is None:
            self.normalization_indices = list(range(self.nb_attributes))
        self.seed = seed
        self.explanation_type = "local"

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

    def execute(self) -> dict:
        command = f"""
                --root_folder {self.model.root_path}
                --train_data_file {self.model.train_data_filename}
                --train_pred_file {self.model._outputs["train_pred_outfile"]}
                --test_data_file {self.model.test_data_filename}
                --test_pred_file {self.model._outputs["test_pred_outfile"]}
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
            command += f" --mus {sanatizeList(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatizeList(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatizeList(self.normalization_indices)}"
            )

        status = fidex(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the Fidex explainer execution..."
            )

        return self._postprocess()


class FidexGloRulesAlgorithm(DimlpfidexAlgorithm):
    # - train_data must contain attributes and classes altogether
    # - You can ask for the execution of fidexGlo with the parameter fidexGlo
    # - parameters with_minimal_version and nb_fidex_rules only used with fidexGlo

    def __init__(
        self,
        model: DimlpfidexModel,
        heuristic: int,
        with_fidexGlo: bool,
        verbose_console: bool = False,
        attributes_file: str = None,
        max_iterations: int = 10,
        min_covering: int = 2,
        covering_strategy: bool = True,
        max_failed_attempts: int = 30,
        min_fidelity: float = 1.0,
        lowest_min_fidelity: float = 0.75,
        dropout_dim: float = 0.0,
        dropout_hyp: float = 0.0,
        decision_threshold: float = None,
        positive_class_index: int = None,
        nb_quant_levels: int = 50,
        normalization_file: str = None,
        mus: list[float] = None,
        sigmas: list[float] = None,
        normalization_indices: list[int] = None,
        nb_threads: int = 1,
        seed: int = 0,
        with_minimal_version: bool = False,
        nb_fidex_rules: int = 1,
    ):
        self.model = model
        self.heuristic = heuristic
        self.with_fidexGlo = with_fidexGlo
        self.nb_attributes = model.nb_attributes
        self.nb_classes = model.nb_classes
        self.global_rules_outfile = "fidexGloRules_output_rules.json"
        self.verbose_console = verbose_console
        self.attributes_file = attributes_file
        self.max_iterations = max_iterations
        self.min_covering = min_covering
        self.covering_strategy = covering_strategy
        self.max_failed_attempts = max_failed_attempts
        self.min_fidelity = min_fidelity
        self.lowest_min_fidelity = lowest_min_fidelity
        self.dropout_dim = dropout_dim
        self.dropout_hyp = dropout_hyp
        self.decision_threshold = decision_threshold
        self.positive_class_index = positive_class_index
        self.nb_quant_levels = nb_quant_levels
        self.normalization_file = normalization_file
        self.mus = mus
        self.sigmas = sigmas
        self.normalization_indices = normalization_indices
        if self.normalization_indices is None:
            self.normalization_indices = list(range(self.nb_attributes))
        self.nb_threads = nb_threads
        self.seed = seed
        self.with_minimal_version = with_minimal_version
        self.nb_fidex_rules = nb_fidex_rules
        self.explanation_type = "local"

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
            command += f" --mus {sanatizeList(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatizeList(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatizeList(self.normalization_indices)}"
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

        if self.with_fidexGlo:
            fidexGlo = FidexGloAlgorithm(
                model=self.model,
                verbose_console=self.verbose_console,
                attributes_file=self.attributes_file,
                seed=self.seed,
                with_minimal_version=self.with_minimal_version,
                nb_fidex_rules=self.nb_fidex_rules,
                max_iterations=self.max_iterations,
                min_covering=self.min_covering,
                covering_strategy=self.covering_strategy,
                max_failed_attempts=self.max_failed_attempts,
                min_fidelity=self.min_fidelity,
                lowest_min_fidelity=self.lowest_min_fidelity,
                dropout_dim=self.dropout_dim,
                dropout_hyp=self.dropout_hyp,
                nb_quant_levels=self.nb_quant_levels,
                normalization_file=self.normalization_file,
                mus=self.mus,
                sigmas=self.sigmas,
                normalization_indices=self.normalization_indices,
            )
            fidexGlo.execute()

        return self._postprocess()


class FidexGloStatsAlgorithm(DimlpfidexAlgorithm):
    # - test_data must contain attributes and classes altogether

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
    # - train_data and test_data must contain attributes and classes altogether

    def __init__(
        self,
        model: DimlpfidexModel,
        verbose_console: bool = False,
        attributes_file: str = None,
        with_minimal_version: bool = False,
        max_iterations: int = 10,
        min_covering: int = 2,
        covering_strategy: bool = True,
        max_failed_attempts: int = 30,
        min_fidelity: float = 1.0,
        lowest_min_fidelity: float = 0.75,
        nb_fidex_rules: int = 1,
        dropout_dim: float = 0.0,
        dropout_hyp: float = 0.0,
        nb_quant_levels: int = 50,
        normalization_file: str = None,
        mus: list[float] = None,
        sigmas: list[float] = None,
        normalization_indices: list[int] = None,
        seed: int = 0,
    ):
        self.model = model
        self.nb_attributes = model.nb_attributes
        self.nb_classes = model.nb_classes
        self.global_rules_file = "fidexGloRules_output_rules.json"
        self.explanation_file = "explanations.txt"
        self.verbose_console = verbose_console
        self.attributes_file = attributes_file
        self.with_minimal_version = with_minimal_version
        self.max_iterations = max_iterations
        self.min_covering = min_covering
        self.covering_strategy = covering_strategy
        self.max_failed_attempts = max_failed_attempts
        self.min_fidelity = min_fidelity
        self.lowest_min_fidelity = lowest_min_fidelity
        self.nb_fidex_rules = nb_fidex_rules
        self.dropout_dim = dropout_dim
        self.dropout_hyp = dropout_hyp
        self.nb_quant_levels = nb_quant_levels
        self.normalization_file = normalization_file
        self.mus = mus
        self.sigmas = sigmas
        self.normalization_indices = normalization_indices
        if self.normalization_indices is None:
            self.normalization_indices = list(range(self.nb_attributes))
        self.seed = seed
        self.explanation_type = "local"

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

    def execute(self) -> dict:
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
            command += f" --mus {sanatizeList(self.mus)}"
        if self.sigmas is not None:
            command += f" --sigmas {sanatizeList(self.sigmas)}"
        if self.normalization_indices is not None and self.normalization_file is None:
            command += (
                f" --normalization_indices {sanatizeList(self.normalization_indices)}"
            )

        status = fidexGlo(command)
        if status != 0:
            raise ValueError(
                "Something went wrong with the FidexGlo explainer execution..."
            )

        return self._postprocess()


# !all optional parameters must be specified inside KWARGS:
class DimlpfidexExplainer(ExplainerBase):
    # - An explainer algorithm must be instanciated (see DimlpfidexAlgorithms based classes) and specified
    # - Optional argument verbose
    # - preprocess_function must have train_data as only parameter
    alias = ["dimlpfidex"]
    mode = "classification"

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

        if "explainer" not in kwargs:
            raise ValueError(
                "Dimlpfidex explainer error: you must add an 'explainer' algorithm inside the kwargs"
            )
        else:
            self.explainer = kwargs["explainer"]

        if not isinstance(self.model, DimlpfidexModel):
            raise ValueError("Model must an instance of DimlpfidexModel based classes.")

        self.model._set_preprocess_function(self.preprocess_function)

    @property
    def explanation_type(self):
        return self.explainer.explanation_type

    def explain(self, X) -> DimlpfidexExplanation:
        _ = X  # X is ignored because all needed data is already given at model initialization
        status = self.model(None)  # ? Not sure if this is the way to do it
        if status != 0:
            raise ValueError("Something went wrong with the model execution...")
        result = self.explainer.execute()

        return DimlpfidexExplanation(self.mode, result)

    def explain_global(self):
        pass  # TODO ?

    def predict(self):
        pass  # TODO ?

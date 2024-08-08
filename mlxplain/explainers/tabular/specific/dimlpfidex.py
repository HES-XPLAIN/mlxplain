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
from dimlpfidex.fidex import fidex
from omnixai.data.tabular import Tabular
from omnixai.explainers.base import ExplainerBase

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
    # - output_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data must contain attributes and classes altogether

    # Déplacer output_path comme verbose?
    # le output_path devrait peut-être s'appeler root_path si on veut pouvoir définir un path pour les attributes_file et normalization_file -> change la manière dont on avait imaginé le tout. À voir

    def __init__(
        self,
        output_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
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
        self.output_path = output_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
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
            self.training_data, self.output_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.output_path.joinpath(self.test_data_filename)
        )

    def __call__(self, verbose) -> int:
        self._preprocess()

        command = f"""
                    --root_folder {self.output_path}
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
        if not verbose:
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


class RFModel(DimlpfidexModel):
    def __init__(self):
        pass


class FidexAlgorithm(DimlpfidexAlgorithm):
    # - test_data must contain attributes and classes altogether

    def __init__(
        self,
        model: DimlpfidexModel,
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

    def _postprocess(self) -> dict:
        absolute_path = self.model.output_path.joinpath(self.rules_outfile)
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

    def execute(self, verbose=True) -> dict:
        # TODO: weight file or rule file if model is RF
        command = f"""
                --root_folder {self.model.output_path}
                --train_data_file {self.model.train_data_filename}
                --train_pred_file {self.model._outputs["train_pred_outfile"]}
                --test_data_file {self.model.test_data_filename}
                --test_pred_file {self.model._outputs["test_pred_outfile"]}
                --weights_file {self.model._outputs["weights_outfile"]}
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

        if self.attributes_file is not None:
            command += f" --attributes_file {self.attributes_file}"
        if not verbose:
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


# !all optional parameters must be specified inside KWARGS:
class DimlpfidexExplainer(ExplainerBase):
    # - An explainer algorithm must be instanciated (see DimlpfidexAlgorithms based classes) and specified
    # - Optional argument verbose
    # - preprocess_function must have train_data as only parameter
    explanation_type = "local"
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
        self.verbose = True

        if "explainer" not in kwargs:
            raise ValueError(
                "Dimlpfidex explainer error: you must add an 'explainer' algorithm inside the kwargs"
            )
        else:
            self.explainer = kwargs["explainer"]

        if not isinstance(self.model, DimlpfidexModel):
            raise ValueError("Model must an instance of DimlpfidexModel based classes.")

        if "verbose" in kwargs:
            if not kwargs["verbose"]:
                self.verbose = False

        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self, X) -> DimlpfidexExplanation:
        _ = X  # X is ignored because all needed data is already given at model initialization
        status = self.model(self.verbose)  # ? Not sure if this is the way to do it
        if status != 0:
            raise ValueError("Something went wrong with the model execution...")
        result = self.explainer.execute(self.verbose)

        return DimlpfidexExplanation(self.mode, result)

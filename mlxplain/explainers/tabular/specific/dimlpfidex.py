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

    # attributes_file? (pas besoin de mettre les stats car on a un nom par défaut)
    # Créer un mode verbose général pour les console file?
    # hidden layers (formater les listes)

    def __init__(
        self,
        output_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes: int,
        nb_classes: int,
        nb_dimlp_nets=25,
        first_hidden_layer=None,
        hidden_layers=None,
    ):
        self.output_path = output_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.nb_dimlp_nets = nb_dimlp_nets
        self.first_hidden_layer = first_hidden_layer
        if self.first_hidden_layer is None:
            self.first_hidden_layer = self.nb_attributes
        self.hidden_layers = hidden_layers
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

    def __call__(self, data) -> int:
        _ = data  # data is ignored because all needed data is already given at model initialization
        self._preprocess()

        command = f"""
                    --root_folder {self.output_path}
                    --train_data_file {self.train_data_filename}
                    --test_data_file {self.test_data_filename}
                    --nb_attributes {self.nb_attributes}
                    --nb_classes {self.nb_classes}
                    --nb_dimlp_nets {self.nb_dimlp_nets}
                    """
        if self.first_hidden_layer is not None:
            command += f" --first_hidden_layer {self.first_hidden_layer}"
        if self.hidden_layers is not None:
            command += f" --hidden_layers {self.hidden_layers}"
        print(command)
        status = dimlpBT(command)

        return status


class RFModel(DimlpfidexModel):
    def __init__(self):
        pass


class FidexAlgorithm(DimlpfidexAlgorithm):
    # - test_data must contain attributes and classes altogether

    def __init__(self, model: DimlpfidexModel):
        self.model = model
        self.nb_attributes = model.nb_attributes
        self.nb_classes = model.nb_classes
        self.rules_outfile = "fidex_output_rules.json"

    def _postprocess(self) -> dict:
        absolute_path = self.model.output_path.joinpath(self.rules_outfile)
        # TODO try catch this
        with open(absolute_path) as file:
            return json.load(file)

    def execute(self) -> dict:
        # TODO: weight file or rule file if model is RF
        fidex(
            f"""
            --root_folder {self.model.output_path}
            --train_data_file {self.model.train_data_filename}
            --train_pred_file {self.model._outputs["train_pred_outfile"]}
            --test_data_file {self.model.test_data_filename}
            --test_pred_file {self.model._outputs["test_pred_outfile"]}
            --weights_file {self.model._outputs["weights_outfile"]}
            --rules_outfile {self.rules_outfile}
            --nb_attributes {self.nb_attributes}
            --nb_classes {self.nb_classes}
        """
        )

        return self._postprocess()


# !all optional parameters must be specified inside KWARGS:
class DimlpfidexExplainer(ExplainerBase):
    # - An explainer algorithm must be instanciated (see DimlpfidexAlgorithms based classes) and specified
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

        if "explainer" not in kwargs:
            raise ValueError(
                "Dimlpfidex explainer error: you must add an 'explainer' algorithm inside the kwargs"
            )
        else:
            self.explainer = kwargs["explainer"]

        if not isinstance(self.model, DimlpfidexModel):
            raise ValueError("Model must an instance of DimlpfidexModel based classes.")

        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self, X) -> DimlpfidexExplanation:
        _ = X  # X is ignored because all needed data is already given at model initialization
        status = self.model(None)  # ? Not sure if this is the way to do it
        if status != 0:
            raise ValueError("Something went wrong with the model execution...")
        result = self.explainer.execute()

        return DimlpfidexExplanation(self.mode, result)

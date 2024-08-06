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


def tabular_to_csv(data: Tabular, path: pl.Path) -> str:
    data.to_pd().to_csv(path, index=False, header=False)
    return path


def csv_to_tabular(path: str) -> Tabular:
    return Tabular(pd.read_csv(path))


class DimlpfidexModel(metaclass=ABCMeta):

    def __init__(self) -> None:
        super.__init__()

    @abstractmethod
    def _set_preprocess_function(self, preprocess_function: Callable = None):
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError


class DimlpfidexAlgorithm(metaclass=ABCMeta):

    def __init__(self) -> None:
        super.__init__()

    @abstractmethod
    def _postprocess(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError


class DimlpBTModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - output_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data must contain attributes and classes altogether

    def __init__(
        self,
        output_path: pl.Path,
        training_data: Tabular,
        testing_data: Tabular,
        nb_attributes,
        nb_classes,
    ):
        # TODO: check if path is writable
        self.output_path = output_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.train_data_filename = "train_data.txt"
        self.test_data_filename = "test_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
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
        print(self.preprocess_function)
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)
            self.preprocess_function(self.testing_data)

        tabular_to_csv(
            self.training_data, self.output_path.joinpath(self.train_data_filename)
        )
        tabular_to_csv(
            self.testing_data, self.output_path.joinpath(self.test_data_filename)
        )

    def train(self) -> int:
        self._preprocess()

        status = dimlpBT(
            f"""
                --root_folder {self.output_path}
                --train_data_file {self.train_data_filename}
                --test_data_file {self.test_data_filename}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                """
        )

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


class DimlpfidexExplainer(ExplainerBase):
    # - preprocess_function must have train_data as only parameter
    explanation_type = "local"
    alias = ["fidex"]
    mode = "classification"

    def __init__(
        self,
        training_data: Tabular,
        model: DimlpfidexModel,
        explainer: DimlpfidexAlgorithm,
        preprocess_function: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.training_data = training_data
        self.preprocess_function = preprocess_function
        self.explainer = explainer

        if not isinstance(self.model, DimlpfidexModel):
            raise ValueError("Model must an instance of DimlpfidexModel based classes.")

        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self) -> dict:
        self.model.train()

        return self.explainer.execute()
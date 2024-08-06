import pathlib as pl
from typing import Callable

import pandas as pd
from dimlpfidex.dimlp import dimlpBT

# from dimlpfidex.fidex import fidex
from omnixai.data.tabular import Tabular
from omnixai.explainers.base import ExplainerBase


def tabular_to_csv(data: Tabular, path: str) -> str:
    data.to_pd().to_csv(path, index=False)
    return path


def csv_to_tabular(path: str) -> Tabular:
    return Tabular(pd.read_csv(path))


class DimlpfidexModel:
    # TODO: implement interface if suitable (?)
    pass


class DimlpBTModel(DimlpfidexModel):
    # to put in the doc:
    # - beware, HES-XPLAIN documentation concerning this model differs from this use case. Please, follow it with precaution.
    # - output_path is directory where temporary files and output files will be generated, it must be a low permission directory
    # - train_data must contain attributes and classes altogether

    def __init__(
        self, output_path: pl.Path, training_data: Tabular, nb_attributes, nb_classes
    ):
        # TODO: check if path is writable
        self.output_path = output_path
        self.training_data = training_data
        self.train_data_filename = "dimlpbt_train_data.txt"
        self.nb_attributes = nb_attributes
        self.nb_classes = nb_classes
        self.preprocess_function = None

    def _set_preprocess_function(self, preprocess_function: Callable):
        self.preprocess_function = preprocess_function

    def _preprocess(self):
        if self.preprocess_function is not None:
            self.preprocess_function(self.training_data)

        tabular_to_csv(self.training_data, self.train_data_filename)

    def _train(self) -> int:
        self._preprocess()

        status = dimlpBT(
            f"""
                --root_folder {self.output_path}
                --train_data_file {self.train_data_filename}
                --nb_attributes {self.nb_attributes}
                --nb_classes {self.nb_classes}
                """
        )

        return status


class FidexExplainer(ExplainerBase):
    # - preprocess_function must have train_data as only parameter
    explanation_type = "local"
    alias = ["fidex"]
    mode = "classification"

    def __init__(
        self,
        model: DimlpfidexModel,
        training_data: Tabular,
        preprocess_function: Callable = None,
        **kwargs,
    ):
        super().__init__(
            training_data=training_data,
            predict_function=model,
            mode=self.mode,
            **kwargs,
        )
        self.model = model
        self.training_data = training_data
        self.preprocess_function = preprocess_function

        if not isinstance(self.model, DimlpfidexModel):
            raise ValueError("Model must an instance of DimlpfidexModel based classes.")

        self.model._set_preprocess_function(self.preprocess_function)

    def explain(self):
        self.model._train()
        print("YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        # fidex("""""")

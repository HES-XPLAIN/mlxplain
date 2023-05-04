"""
The SHAP explainer for tabular data.
"""

import shap
import numpy as np
import pandas as pd

from hes_xplain.methods.tabular.TabularXAIMethod import TabularXAIMethod


class ShapTabular(TabularXAIMethod):
    def __init__(self, data, model):
        super().__init__(data)
        self.model = model

    def preprocess_data(self):
        # Preprocess tabular data here (e.g. normalization, one-hot encoding, etc.)
        self.preprocessed_data = self.tabular_data

    def explain(self, instance = None):
        # Calculate SHAP values for the specified instance
        explainer = shap.TreeExplainer(model = self.model)

        if instance is None:
            print("Shap values computed for self.data")
            shap_values = explainer.shap_values(self.data)
        else:
            print("Shap values computed for instance")
            shap_values = explainer.shap_values(instance)

        return shap_values


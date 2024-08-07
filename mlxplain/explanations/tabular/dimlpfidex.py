#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import matplotlib as plt
import plotly.graph_objects as go
from omnixai.explanations.base import DashFigure, ExplanationBase

# Rules JSON object hierarchy (what should be inside explanation):
# {
#   "positive index class": int
#   "rules": object list [
#                           {
#                               "accuracy": float
#                               "antecedents": object list [
#                                                               "attribute": int
#                                                               "inequality": boolean
#                                                               "value": float
#                                                          ]
#                               "confidence": float
#                               "coveredSamples": int list
#                               "coveringSize": int
#                               "fidelity": float
#                               "outputClass": int
#                           }
#                       ]
#
# }


# Plot ideas:
# - 5 best rules (by covering size)
# - 5 most covered samples
# - 5 least covered samples
# - rule's metrics means and stds
# - covering size evolution graph
# - stats plot
# - antecedents plots (5 most used attributes, 5 least used attributes)


class DimlpfidexExplanation(ExplanationBase):

    def __init__(self, mode, explanations: dict = {}) -> None:
        super().__init__()
        self.mode = mode
        self.explanations = explanations

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return "salut"

    def get_explanations(self) -> dict:
        return self.explanations

    def plot(self, **kwargs) -> None:
        figure, ax = plt.subplots()
        rules = self.explanations["rules"]
        idxs = []
        covered_samples = []

        for i, rule in enumerate(rules):
            idxs.append(f"Rule #{i}")
            covered_samples.append(rule["coveringSize"])

        ax.bar(idxs, covered_samples)

        ax.set_ylabel("Number of covered samples")
        ax.set_xlabel("Rules id's")

        return figure

    def plotly_plot(self, **kwargs) -> None:
        rules = self.explanations["rules"]
        idxs = []
        covered_samples = []

        for i, rule in enumerate(rules):
            idxs.append(f"Rule #{i}")
            covered_samples.append(rule["coveringSize"])

        fig = go.Figure(
            data=[go.Bar(x=idxs, y=covered_samples)],
            layout_title_text="Generated rules",
        )
        return DashFigure(fig)

    def ipython_plot(self, **kwargs) -> None:
        pass  # TODO

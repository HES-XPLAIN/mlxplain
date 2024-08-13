#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import matplotlib as plt
import plotly.graph_objects as go
from matplotlib.figure import Figure
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
# - 10 best rules (by covering size)
# - 10 most covered samples
# - 10 least covered samples
# - rule's metrics means, stds and other stats
# - covering size evolution graph
# - antecedents plots (5 most used attributes, 5 least used attributes)


class DimlpfidexExplanation(ExplanationBase):

    def __init__(self, mode, nsamples: int, explanations: dict = {}) -> None:
        super().__init__()
        self.mode = mode
        self.nsamples = nsamples
        self.explanations = explanations

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return "salut"

    def get_explanations(plt, self) -> dict:
        return self.explanations

    def _plot_best_rules(axes, self) -> None:
        rules = self.explanations["rules"]
        idxs = []
        covered_samples = []

        for i, rule in enumerate(rules):
            idxs.append(f"Rule #{i}")
            covered_samples.append(rule["coveringSize"])

        axes.bar(idxs, covered_samples)

        axes.set_xlabel("Rules id")
        axes.set_ylabel("Number of covered samples")

    def _plot_most_covered_samples(axes, self) -> None:
        samplesIds = list(range(self.nsamples))
        samplesCount = [0] * self.nsamples

        for rule in self.explanations["rules"]:
            for covered_sample in rule["coveredSamples"]:
                samplesCount[covered_sample] += 1

        axes.bar(samplesIds, samplesCount)

        axes.set_xlabel("Samples id")
        axes.set_ylabel("Most covered samples")

    def plot(self, **kwargs) -> Figure:
        figure, (ax1, ax2) = plt.subplots(2, 2)
        self._plot_best_rules(ax1),
        self._plot_most_covered_samples(ax2),

        return figure

    # TODO: implement plots here for dashboard
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

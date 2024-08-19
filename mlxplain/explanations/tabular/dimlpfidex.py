#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import matplotlib.pyplot as plt
import plotly
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


# TODO add comments
class DimlpfidexExplanation(ExplanationBase):

    def __init__(
        self,
        mode,
        nsamples: int,
        explanations: dict = {},
        attribute_names: list[str] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.nsamples = nsamples
        self.explanations = explanations
        self.attribute_names = attribute_names
        self.class_names = class_names

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
        rules = self.explanations["rules"]
        idxs = []
        covered_samples = []

        for i, rule in enumerate(rules):
            idxs.append(i)
            covered_samples.append(rule["coveringSize"])

        figure, axes = plt.subplots(1, 1)

        axes.bar(idxs, covered_samples)

        plt.xlabel("Generated rules ID")
        plt.ylabel("Number of samples covered")

        return figure

    def _plotly_figure(self, **kwargs):
        rules = self.explanations["rules"]
        idxs = []
        covered_samples = []
        rule_str = []

        for i, rule in enumerate(rules):
            idxs.append(f"Rule #{i}")
            covered_samples.append(rule["coveringSize"])
            rule_str.append(self.json_rule_to_string(i, rule))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=idxs,
                    y=covered_samples,
                    customdata=rule_str,
                    hovertemplate="%{customdata}",
                    name="",
                )
            ],
            layout_title_text="Generated rules",
        )

        return fig

    def plotly_plot(self, **kwargs) -> None:
        return DashFigure(self._plotly_figure())

    def ipython_plot(self, **kwargs) -> None:
        plotly.offline.iplot(self._plotly_figure())

    def json_rule_to_string(self, i, json_rule):
        rule = f"Rule #{i}: "

        for antecedant in json_rule["antecedents"]:

            if self.attribute_names is not None:
                attribute = self.attribute_names[antecedant["attribute"]]
            else:
                attribute = f"X{antecedant['attribute']}"

            rule += f"{attribute}"

            if antecedant["inequality"]:
                rule += ">="
            else:
                rule += "<"

            rule += "{:.2f}".format(antecedant["value"]) + " "

        if self.class_names is not None:
            output_class = self.class_names[json_rule["outputClass"]]
        else:
            output_class = f"class {json_rule['outputClass']}"

        rule += f" = {output_class}"

        accuracy = json_rule["accuracy"]
        confidence = json_rule["confidence"]
        coveringSize = len(json_rule["coveredSamples"])
        fidelity = json_rule["fidelity"]

        rule += "<br>Confidence: {:.2f}".format(confidence)
        rule += "<br>Accuracy: {:.2f}".format(accuracy)
        rule += f"<br>Samples covered: {coveringSize}"
        rule += "<br>Fidelity: {:.2f}".format(fidelity)

        return rule

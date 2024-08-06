#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Rules Extraction explanations for vision tasks.
"""

import numpy as np
import pandas as pd
from omnixai.explanations.base import DashFigure, ExplanationBase


class RuleImportance(ExplanationBase):
    """
    The class for rules extraction explanation. It stores a batch of rules and the corresponding
    class names. Each rule represents a rule importance explanation.
    """

    def __init__(self, mode: str = "classification", explanations=None):
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def get_explanations(self):
        """
        Gets the generated explanations.

        :return: The explanation for one specific image (a tuple of 'rule' and concerned 'class')
            or the explanations for all the instances (a list of tuples).
        """
        return self.explanations if len(self.explanations) > 1 else self.explanations[0]

    @staticmethod
    def _get_changed_columns(query, cfs):
        """
        Gets the differences between the instance and the generated counterfactual examples.

        :param query: The input instance.
        :param cfs: The counterfactual examples.
        :return: The feature columns that have been changed in ``cfs``.
        :rtype: List
        """
        columns = []
        for col in query.columns:
            u = query[[col]].values[0]
            for val in cfs[[col]].values:
                if val != u:
                    columns.append(col)
                    break
        return columns

    @staticmethod
    def _plot(plt, index, query, cfs, context=None, font_size=10, bar_width=0.4):
        """
        Plots a table showing the generated counterfactual examples.
        """
        df = pd.concat([query, cfs], axis=0)
        rows = [f"Instance {index}"] + [f"CF {k}" for k in range(1, df.shape[0])]
        counts = np.zeros(len(df.columns))
        for i in range(df.shape[1] - 1):
            for j in range(1, df.shape[0]):
                counts[i] += int(df.values[0, i] != df.values[j, i])
        # Context
        if context is not None:
            df = pd.concat([context, df], axis=0)
            rows = [f"Context {k + 1}" for k in range(context.shape[0])] + rows

        plt.bar(np.arange(len(df.columns)) + 0.5, counts, bar_width)
        table = plt.table(
            cellText=df.values, rowLabels=rows, colLabels=df.columns, loc="bottom"
        )
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.ylabel("The number of feature changes")
        plt.yticks(np.arange(max(counts)))
        plt.xticks([])
        plt.title("Counterfactual Examples")
        plt.grid()

        # Highlight the differences between the query and the CF examples
        for k in range(df.shape[1]):
            table[(0, k)].set_facecolor("#C5C5C5")
            for i in range(1, df.shape[0] - cfs.shape[0] + 1):
                table[(i, k)].set_facecolor("#E2DED0")
        for j in range(df.shape[0] - cfs.shape[0], df.shape[0]):
            for k in range(df.shape[1] - 1):
                if query.values[0][k] != df.values[j][k]:
                    table[(j + 1, k)].set_facecolor("#56b5fd")

        # Change the font size if `font_size` is set
        if font_size is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

    def plot(self, index=None, class_names=None, font_size=10, **kwargs):
        """
        Returns a list of matplotlib figures showing the explanations of
        one or the first 5 instances.

        :param index: The index of an explanation result stored in ``CFExplanation``. For
            example, it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots the explanations of the first 5 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param font_size: The font size of table entries.
        :return: A list of matplotlib figures plotting counterfactual examples.
        """
        import warnings

        import matplotlib.pyplot as plt

        explanations = self.get_explanations()
        explanations = (
            {index: explanations}
            if isinstance(explanations, dict)
            else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if len(indices) > 5:
            warnings.warn(
                f"There are too many instances ({len(indices)} > 5), "
                f"so only the first 5 instances are plotted."
            )
            indices = indices[:5]

        figures = []
        for i, index in enumerate(indices):
            fig = plt.figure()
            figures.append(fig)
            exp = explanations[index]
            if exp["counterfactual"] is None:
                continue
            if len(exp["query"].columns) > 5:
                columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
            else:
                columns = exp["query"].columns
            query, cfs = exp["query"][columns], exp["counterfactual"][columns]
            context = exp["context"][columns] if "context" in exp else None

            dfs = [query, cfs, context]
            if class_names is not None:
                for df in dfs:
                    if df is not None:
                        df["label"] = [
                            class_names[label] for label in df["label"].values
                        ]
            self._plot(plt, index, query, cfs, context, font_size)
        return figures

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the generated rule explanations in Dash.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure showing the counterfactual examples.
        """

        # todo: display more than one rules (add params?)
        # Assuming self.explanations is now a list of tuples as provided in the input
        exp = self.explanations[0]
        conditions, label = exp  # Unpack the tuple

        # Create a DataFrame for the conditions and label
        query_df = pd.DataFrame({"conditions": [conditions], "label": [label]})

        # If class_names is provided, map the labels to class names
        if class_names is not None:
            query_df["label"] = query_df["label"].map(lambda x: class_names[x])

        # Create a DashFigure to display the DataFrame
        return DashFigure(self._plotly_table(query_df, None))

    def ipython_plot(self, **kwargs):
        """
        Plots figures in IPython.
        """
        raise NotImplementedError

    @staticmethod
    def _plotly_table(query, context):
        """
        Plots a table showing the generated counterfactual examples.
        """
        from dash import dash_table

        feature_columns = query.columns
        columns = [{"name": "#", "id": "#"}] + [
            {"name": c, "id": c} for c in feature_columns
        ]
        context_size = context.shape[0] if context is not None else 0
        highlight_row_offset = query.shape[0] + context_size + 1

        highlights = []
        query = query.values

        data = []
        # Context row
        if context is not None:
            for x in context.values:
                row = {"#": "Context"}
                row.update({c: d for c, d in zip(feature_columns, x)})
                data.append(row)
        # Query row
        for x in query:
            row = {"#": "Rule"}
            row.update({c: d for c, d in zip(feature_columns, x)})
            data.append(row)
        # Separator
        row = {"#": "-"}
        row.update({c: "-" for c in feature_columns})
        data.append(row)

        style_data_conditional = [
            {"if": {"row_index": 0}, "backgroundColor": "rgb(240, 240, 240)"}
        ]
        for i, j in highlights:
            c = feature_columns[j]
            cond = {
                "if": {
                    "filter_query": "{{{0}}} != ''".format(c),
                    "column_id": c,
                    "row_index": i + highlight_row_offset,
                },
                "backgroundColor": "dodgerblue",
            }
            style_data_conditional.append(cond)

        table = dash_table.DataTable(
            id="table",
            columns=columns,
            data=data,
            style_header_conditional=[{"textAlign": "center"}],
            style_cell_conditional=[{"textAlign": "center"}],
            style_data_conditional=style_data_conditional,
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_table={"overflowX": "scroll", "overflowY": "auto", "height": "260px"},
        )
        return table

#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Rules Extraction explanations for vision tasks.
"""

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
    def _plot(data, font_size=10):

        import matplotlib.pyplot as plt

        # Create a figure and an axis
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed

        # Create a table from the data
        ax.table(
            cellText=data.values,
            colLabels=data.columns,
            cellLoc="center",
            loc="center",
            fontsize=font_size,
        )

        # Hide axes
        ax.axis("off")

        # show the plot
        plt.show()

        return fig, ax

    def plot(self, class_names=None, font_size=10, **kwargs):
        # Determine the number of rules to include
        num_rules = 5 if len(self.explanations) > 5 else len(self.explanations)

        # Prepare lists to store rules and labels
        rules_list = []
        labels_list = []

        # Extract rules and labels
        for i in range(num_rules):
            rules, label = self.explanations[i]
            rules_list.append(rules)
            labels_list.append(label)

        # Create a DataFrame from the rules and labels
        data_df = pd.DataFrame({"rules": rules_list, "labels": labels_list})

        # If class_names is provided, map the labels to class names
        if class_names is not None:
            data_df["labels"] = data_df["labels"].map(lambda x: class_names[x])

        # Call _plot to generate the table and get the figure and axes
        fig, ax = self._plot(data_df, font_size, **kwargs)

        # Return the figure and axes
        return fig, ax

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots figures in IPython.
        """
        import plotly
        import plotly.figure_factory as ff

        # Determine the number of rules to include
        num_rules = 5 if len(self.explanations) > 5 else len(self.explanations)

        # Prepare lists to store rules and labels
        rules_list = []
        labels_list = []

        # Loop through the explanations and unpack the rules and labels
        for i in range(num_rules):
            rules, label = self.explanations[i]
            rules_list.append(rules)
            labels_list.append(label)

        # Create a DataFrame from the rules and labels lists
        query_df = pd.DataFrame({"rules": rules_list, "label": labels_list})

        # If class_names is provided, map the labels to class names
        if class_names is not None:
            query_df["label"] = query_df["label"].map(lambda x: class_names[x])

        # Create and display the table using Plotly
        return plotly.offline.iplot(ff.create_table(query_df))

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the generated rule explanations in Dash.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure showing the counterfactual examples.
        """

        # Determine the number of rules to include
        num_rules = 5 if len(self.explanations) > 5 else len(self.explanations)

        # Prepare lists to store rules and labels
        rules_list = []
        labels_list = []

        # Loop through the explanations and unpack the rules and labels
        for i in range(num_rules):
            rules, label = self.explanations[i]
            rules_list.append(rules)
            labels_list.append(label)

        # Create a DataFrame from the rules and labels lists
        query_df = pd.DataFrame({"rules": rules_list, "label": labels_list})

        # If class_names is provided, map the labels to class names
        if class_names is not None:
            query_df["label"] = query_df["label"].map(lambda x: class_names[x])

        # Create a DashFigure to display the DataFrame
        return DashFigure(self._plotly_table(query_df, None))

    @staticmethod
    def _plotly_table(query, context):
        """
        Plots a table showing the generated rules.
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

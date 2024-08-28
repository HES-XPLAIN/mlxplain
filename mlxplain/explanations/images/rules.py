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
from omnixai_community.explanations.base import DashFigure, ExplanationBase


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
        """
        Plots figures using Matplotlib.

        :param font_size: The font size.
        :return: A generic plot.
        """

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

    def _get_labels(self, class_names=None):
        """
        Extract rules and labels.

        :param class_names: A list of the class names indexed by the labels.
        :return: A dataframe containing the rules and the related labels if class names is provided
        """
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

        return data_df

    def plot(self, class_names=None, font_size=10, **kwargs):
        """
        Plots figures in Matplotlib.

        :param class_names: A list of the class names indexed by the labels.
        :param font_size: The font size.
        :return: A generic plot.
        """
        # Extract rules and labels
        query_df = self._get_labels(class_names)

        # Call _plot to generate the table and get the figure and axes
        fig, ax = self._plot(query_df, font_size, **kwargs)

        # Return the figure and axes
        return fig, ax

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots figures in IPython.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure showing the extracted rules.
        """
        import plotly
        import plotly.figure_factory as ff

        # Extract rules and labels
        query_df = self._get_labels(class_names)

        # Create and display the table using Plotly
        return plotly.offline.iplot(ff.create_table(query_df))

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the generated rule explanations in Dash.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure showing the extracted rules.
        """

        # Extract rules and labels
        query_df = self._get_labels(class_names)

        # Create a DashFigure to display the DataFrame
        return DashFigure(self._plotly_table(query_df, None))

    @staticmethod
    def _plotly_table(query, context):
        """
        Plots a table showing the generated rules.

        :param query: A dataframe of the extracted rules.
        :return: A plotly table figure showing rules and labels.
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

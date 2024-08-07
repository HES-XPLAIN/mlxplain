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
    def _plot(plt, index, query, cfs, context=None, font_size=10, bar_width=0.4):
        raise NotImplementedError


    def plot(self, index=None, class_names=None, font_size=10, **kwargs):
        raise NotImplementedError



    def ipython_plot(self, **kwargs):
        """
        Plots figures in IPython.
        """
        raise NotImplementedError
    

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

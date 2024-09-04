import matplotlib.pyplot as plt
import pandas as pd
from omnixai_community.explanations.base import DashFigure, ExplanationBase


class RuleImportance(ExplanationBase):
    def __init__(
        self, mode: str = "classification", explanations=None, target_class: str = None
    ):
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations
        self.target_class = target_class

    def __repr__(self):
        return repr(self.explanations)

    def get_explanations(self):
        return self.explanations if len(self.explanations) > 1 else self.explanations[0]

    def _format_rule(self, rule, label, rank):
        conditions = [
            f"feature {cond.split()[0]} {cond.split()[1]} {cond.split()[2]}"
            for cond in rule
        ]
        prediction = (
            f'"{self.target_class}"' if label == 1 else f'Not "{self.target_class}"'
        )
        return f"Rule {rank}: If {' and '.join(conditions)}, then the sample is classified as {prediction}"

    def _get_labels(self):
        num_rules = min(5, len(self.explanations))
        rules_list = []
        for i in range(num_rules):
            rules, label = self.explanations[i]
            formatted_rule = self._format_rule(rules, label, i + 1)
            rules_list.append(formatted_rule)
        return pd.DataFrame({"Ranked Rules": rules_list})

    def plot(self, font_size=10):
        data = self._get_labels()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        table = ax.table(
            cellText=data.values, colLabels=data.columns, loc="center", cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.2, 1.5)
        plt.title(
            "Rule Importance Explanation (Ranked)",
            fontsize=font_size + 4,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        return fig, ax

    def ipython_plot(self, **kwargs):
        import plotly.figure_factory as ff

        data = self._get_labels()
        fig = ff.create_table(data)
        fig.update_layout(title="Rule Importance (Ranked)", height=600)
        return fig

    def plotly_plot(self, **kwargs):
        import plotly.graph_objs as go

        data = self._get_labels()
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(data.columns),
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[data[col] for col in data.columns],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(title="Rule Importance (Ranked)", height=600)
        return DashFigure(fig)

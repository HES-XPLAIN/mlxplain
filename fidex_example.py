#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import pathlib as pl

import pandas as pd
from omnixai.data.tabular import Tabular
from omnixai.visualization.dashboard import Dashboard

from mlxplain.explainers.tabular.specific.dimlpfidex import (
    DimlpBTModel,
    DimlpfidexExplainer,
    FidexAlgorithm,
)

if __name__ == "__main__":
    # do stuff with data

    # ensure path exists
    output_path = pl.Path("/tmp/explainer/")

    # dummy datas for testing purposes
    train_data = Tabular(
        data=pd.DataFrame(
            data=[[1, 2, 3, 1, 0], [4, 5, 6, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        categorical_columns=["male", "female"],
    )

    test_data = Tabular(
        data=pd.DataFrame(
            data=[[3, 1, 4, 1, 0], [7, 6, 4, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        categorical_columns=["male", "female"],
    )

    model = DimlpBTModel(output_path, train_data, test_data, 3, 2)
    algorithm = FidexAlgorithm(model)
    explainer = DimlpfidexExplainer(train_data, model, algorithm)
    local_explanations = explainer.explain()

    db = Dashboard(instances=test_data, local_explanations=local_explanations)
    db.show()

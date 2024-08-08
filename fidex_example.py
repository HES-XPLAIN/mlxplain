#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import pathlib as pl

import pandas as pd
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.auto import TabularExplainer
from omnixai.visualization.dashboard import Dashboard

from mlxplain.explainers.tabular.specific.dimlpfidex import (  # DimlpBTModel,; GradBoostModel,; RandomForestModel,; SVMModel,; FidexAlgorithm,
    FidexGloRulesAlgorithm,
    MLPModel,
)

if __name__ == "__main__":
    # do stuff with data

    # ensure path exists
    output_path = pl.Path("/tmp/explainer/")

    # dummy datas for testing purposes
    train_data = Tabular(
        data=pd.DataFrame(
            data=[[1, 2, 3, 1, 0], [4, 5, 6, 0, 1], [1, 2, 3, 1, 0], [4, 5, 6, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        categorical_columns=["male", "female", "male", "female"],
    )

    test_data = Tabular(
        data=pd.DataFrame(
            data=[[3, 1, 4, 1, 0], [7, 6, 4, 0, 1], [3, 1, 4, 1, 0], [7, 6, 4, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        categorical_columns=["male", "female", "male", "female"],
    )

    # model = DimlpBTModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     3,
    #     2,
    #     seed=1,
    # )

    # model = GradBoostModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     3,
    #     2,
    #     seed=1,
    #     n_estimators = 3,
    #     loss="exponential",
    #     learning_rate=32,
    #     subsample=0.3,
    #     criterion="squared_error",
    #     max_depth="no_max_depth",
    #     min_samples_split=0.5,
    #     min_samples_leaf=0.5,
    #     min_weight_fraction_leaf=0.1,
    #     max_features=3,
    #     max_leaf_nodes=3,
    #     min_impurity_decrease=3,
    #     init="zero",
    #     verbose=12,
    #     warm_start=True,
    #     validation_fraction=0.2,
    #     n_iter_no_change=35,
    #     tol=0.1,
    #     ccp_alpha=0.1
    # )

    # model = RandomForestModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     3,
    #     2,
    #     seed=1,
    #     n_estimators = 3,
    #     criterion="entropy",
    #     max_depth=2,
    #     min_samples_split=0.5,
    #     min_samples_leaf=0.5,
    #     min_weight_fraction_leaf=0.1,
    #     max_features=3,
    #     max_leaf_nodes=3,
    #     min_impurity_decrease=3,
    #     bootstrap=True,
    #     oob_score=True,
    #     n_jobs=-1,
    #     verbose=12,
    #     warm_start=True,
    #     class_weight="{0:1.2, 1:2.3}",
    #     ccp_alpha=0.1,
    #     max_samples=2,
    # )

    # model = SVMModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     3,
    #     2,
    #     return_roc=True,
    #     # positive_class_index=1,
    #     nb_quant_levels=45,
    #     K=0.1,
    #     C=0.1,
    #     kernel="sigmoid",
    #     degree=5,
    #     gamma=0.4,
    #     coef0=1.3,
    #     shrinking=False,
    #     tol=1,
    #     cache_size=150.2,
    #     class_weight={0:1.2, 1:3.5},
    #     verbose = True,
    #     max_iterations= 50,
    #     decision_function_shape="ovo",
    #     break_ties=False
    # )

    model = MLPModel(
        output_path,
        train_data,
        test_data,
        3,
        2,
        nb_quant_levels=45,
        K=0.1,
        hidden_layer_sizes=[12, 13, 14],
        activation="tanh",
        solver="sgd",
        alpha=0.1,
        batch_size=2,
        learning_rate="adaptive",
        learning_rate_init=0.1,
        power_t=0.2,
        max_iterations=150,
        shuffle=False,
        seed=1,
        tol=0.3,
        verbose=True,
        warm_start=True,
        momentum=0.1,
        nesterovs_momentum=False,
        # early_stopping=True,
        validation_fraction=0.2,
        beta_1=0.2,
        beta_2=0.2,
        epsilon=0.1,
        n_iter_no_change=1,
        max_fun=2,
    )

    # algorithm = FidexAlgorithm(
    #     model,
    #     seed=1,
    #     attributes_file = "attributes.txt",
    #     max_iterations = 8,
    #     min_covering = 3,
    #     covering_strategy = False,
    #     max_failed_attempts = 35,
    #     min_fidelity = 0.4,
    #     lowest_min_fidelity = 0.4,
    #     dropout_dim = 0.5,
    #     dropout_hyp = 0.5,
    #     decision_threshold = 0.2,
    #     positive_class_index = 0,
    #     nb_quant_levels = 45,
    #     # normalization_file = "normalization.txt",
    #     mus = [1,3],
    #     sigmas = [2,3],
    #     normalization_indices = [0,2]
    #     )

    algorithm = FidexGloRulesAlgorithm(
        model,
        heuristic=1,
        seed=1,
        attributes_file="attributes.txt",
        max_iterations=8,
        min_covering=3,
        covering_strategy=False,
        max_failed_attempts=35,
        min_fidelity=0.4,
        lowest_min_fidelity=0.4,
        dropout_dim=0.5,
        dropout_hyp=0.5,
        decision_threshold=0.2,
        positive_class_index=0,
        nb_quant_levels=45,
        # normalization_file = "normalization.txt",
        mus=[1, 3],
        sigmas=[2, 3],
        normalization_indices=[0, 2],
        nb_threads=4,
    )

    explainer = TabularExplainer(
        explainers=["dimlpfidex"],
        data=train_data,
        mode="classification",
        model=model,
        params={"dimlpfidex": {"explainer": algorithm, "verbose": True}},
    )

    explanations = explainer.explain(test_data, run_predict=False)

    db = Dashboard(local_explanations=explanations)
    db.show()

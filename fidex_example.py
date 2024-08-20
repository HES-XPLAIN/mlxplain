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

from mlxplain.explainers.tabular.specific.dimlpfidex import (  # DimlpBTModel,; GradBoostModel,; RandomForestModel,; SVMModel,; FidexAlgorithm,; SVMModel,; GradBoostModel,; RandomForestModel,; DimlpBTModel
    FidexAlgorithm,
    FidexGloRulesAlgorithm,
    SVMModel,
)


def load_data():
    dataset = pd.read_csv("ObesityDataSet.csv")

    # reducing labels names size
    dataset.rename(
        columns={
            "family_history_with_overweight": "FHWO",
            "NObeyesdad": "OLD",
        },
        inplace=True,
    )

    # shuffle the entire dataset
    dataset = dataset.sample(frac=1)
    nrows = int(dataset.shape[0] * 0.1)
    dataset = dataset.iloc[:nrows, :]

    strToBinDict = {"yes": 1, "no": 0}

    with pd.option_context("future.no_silent_downcasting", True):
        dataset["FHWO"] = dataset["FHWO"].replace(strToBinDict).astype("int8")
        dataset["FAVC"] = dataset["FAVC"].replace(strToBinDict).astype("int8")
        dataset["SMOKE"] = dataset["SMOKE"].replace(strToBinDict).astype("int8")
        dataset["SCC"] = dataset["SCC"].replace(strToBinDict).astype("int8")

    adjToValDict = {"Always": 1.0, "Frequently": 0.66, "Sometimes": 0.33, "no": 0.0}

    # Avoid deprecation warning for Pandas 3.x
    with pd.option_context("future.no_silent_downcasting", True):
        dataset["CAEC"] = dataset["CAEC"].replace(adjToValDict).astype("float64")
        dataset["CALC"] = dataset["CALC"].replace(adjToValDict).astype("float64")

    genderCols = pd.get_dummies(
        dataset["Gender"], prefix="Gender", prefix_sep="_", dtype="int8"
    )
    mtransCols = pd.get_dummies(
        dataset["MTRANS"], prefix="MTRANS", prefix_sep="_", dtype="int8"
    )
    oldCols = pd.get_dummies(dataset["OLD"], prefix="OLD", prefix_sep="_", dtype="int8")
    dataset = pd.concat(
        [genderCols, dataset.iloc[:, :16], mtransCols, dataset.iloc[:, 16:], oldCols],
        axis=1,
    )
    dataset.drop(["Gender", "MTRANS", "OLD"], axis=1, inplace=True)

    labels = list(dataset.columns)

    nclasses = sum(1 for label in labels if label.startswith("OLD_"))
    nattributes = len(labels) - nclasses

    nRecords = dataset.shape[0]
    trainSplit = int(0.75 * nRecords)

    trainds = dataset.iloc[:trainSplit, :]
    testds = dataset.iloc[trainSplit:, :]

    train = Tabular(data=trainds, feature_columns=labels)
    test = Tabular(data=testds, feature_columns=labels)

    return train, test, nattributes, nclasses


if __name__ == "__main__":

    # ensure path exists
    output_path = pl.Path("/tmp/explainer/")
    train_data_file = "train_data.txt"
    test_data_file = "test_data.txt"

    # do load data
    train_data, test_data, nattributes, nclasses = load_data()

    # model = DimlpBTModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     nattributes,
    #     nclasses,
    # )

    # model = GradBoostModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     nattributes,
    #     nclasses,
    #     verbose_console=True,
    #     seed=1,
    #     n_estimators = 3,
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
    #     verbose_scikit=12,
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
    #     nattributes,
    #     nclasses,
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
    #     verbose_scikit=12,
    #     warm_start=True,
    #     class_weight={0:1.2, 1:2.3},
    #     ccp_alpha=0.1,
    #     max_samples=2,
    # )

    model = SVMModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        return_roc=True,
        output_roc="roc_curve.png",
        positive_class_index=1,
        nb_quant_levels=45,
        K=0.1,
        C=0.1,
        kernel="sigmoid",
        degree=5,
        gamma=0.4,
        coef0=1.3,
        shrinking=False,
        tol=1,
        cache_size=150.2,
        class_weight={0: 1.2, 1: 3.5},
        max_iterations=50,
        decision_function_shape="ovo",
        break_ties=False,
        verbose_console=True,
    )

    # model = MLPModel(
    #     output_path,
    #     train_data,
    #     test_data,
    #     nattributes,
    #     nclasses,
    #     verbose_console = True,
    # nb_quant_levels=45,
    # K=0.1,
    # hidden_layer_sizes=[12, 13, 14],
    # activation="tanh",
    # solver="sgd",
    # alpha=0.1,
    # batch_size=2,
    # learning_rate="adaptive",
    # learning_rate_init=0.1,
    # power_t=0.2,
    # max_iterations=150,
    # shuffle=False,
    # seed=1,
    # tol=0.3,
    # verbose=True,
    # warm_start=True,
    # momentum=0.1,
    # nesterovs_momentum=False,
    # early_stopping=True,
    # validation_fraction=0.2,
    # beta_1=0.2,
    # beta_2=0.2,
    # epsilon=0.1,
    # n_iter_no_change=1,
    # max_fun=2,
    # )

    fidex = FidexAlgorithm(
        model,
        seed=1,
        # attributes_file = "attributes.txt",
        max_iterations=10,
        min_covering=3,
        # covering_strategy = False,
        max_failed_attempts=35,
        min_fidelity=0.9,
        lowest_min_fidelity=0.8,
        dropout_dim=0.5,
        dropout_hyp=0.5,
        # decision_threshold = 0.2,
        # positive_class_index = 0,
        nb_quant_levels=45,
        # normalization_file = "normalization.txt",
        # mus = [1,3],
        # sigmas = [2,3],
        # normalization_indices = [0,2]
    )

    fidexGloRules = FidexGloRulesAlgorithm(
        model,
        heuristic=1,
        seed=1,
        # max_iterations=10,
        # min_covering=3,
        # covering_strategy=False,
        # max_failed_attempts=35,
        # min_fidelity=0.9,
        # lowest_min_fidelity=0.8,
        # dropout_dim=0.5,
        # dropout_hyp=0.5,
        # decision_threshold=0.2,
        positive_class_index=0,
        # nb_quant_levels=45,
        # normalization_file = "normalization.txt",
        # mus=[1, 3],
        # sigmas=[2, 3],
        # normalization_indices=[0, 2],
        nb_threads=4,
    )

    global_explainer = TabularExplainer(
        explainers=["dimlpfidex"],
        data=train_data,
        mode="classification",
        model=model,
        params={"dimlpfidex": {"explainer": fidexGloRules, "verbose": True}},
    )

    local_explainer = TabularExplainer(
        explainers=["dimlpfidex"],
        data=train_data,
        mode="classification",
        model=model,
        params={"dimlpfidex": {"explainer": fidex, "verbose": True}},
    )

    le = local_explainer.explain(test_data, run_predict=False)
    ge = global_explainer.explain(test_data, run_predict=False)

    db = Dashboard(local_explanations=le, global_explanations=ge)
    db.show()

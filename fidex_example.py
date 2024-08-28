#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import pathlib as pl

import pandas as pd
from omnixai_community.data.tabular import Tabular
from omnixai_community.explainers.tabular.auto import TabularExplainer

from mlxplain.explainers.tabular.specific.dimlpfidex import (
    DimlpBTModel,
    GradBoostModel,
    MLPModel,
    RandomForestModel,
    SVMModel,
)


def load_data(root_folder: pl.Path):
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
    classes = [label for label in labels if label.startswith("OLD_")]
    attributes = labels[: len(labels) - len(classes)]

    with open(root_folder.joinpath("attributes.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")

    nRecords = dataset.shape[0]
    trainSplit = int(0.75 * nRecords)

    trainds = dataset.iloc[:trainSplit, :]
    testds = dataset.iloc[trainSplit:, :]

    train = Tabular(data=trainds, feature_columns=labels)
    test = Tabular(data=testds, feature_columns=labels)

    return train, test, attributes, classes


def load_dummy_data():
    train_data = Tabular(
        data=pd.DataFrame(
            data=[[1, 2, 3, 1, 0], [4, 5, 6, 0, 1], [1, 2, 3, 1, 0], [4, 5, 6, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        # feature_columns=["a", "b", "c", "male", "female"],
        categorical_columns=["male", "female"],
    )

    test_data = Tabular(
        data=pd.DataFrame(
            data=[[3, 1, 4, 1, 0], [7, 6, 4, 0, 1], [3, 1, 4, 1, 0], [7, 6, 4, 0, 1]],
            columns=["a", "b", "c", "male", "female"],
        ),
        # feature_columns=["a", "b", "c", "male", "female"],
        categorical_columns=["male", "female"],
    )

    return train_data, test_data, 3, 2


def get_local_explainer(model, train_data):
    return TabularExplainer(
        explainers=["fidex"],
        data=train_data,
        model=model,
        mode="classification",
        params={
            "fidex": {
                "seed": 1,
                "max_iterations": 10,
                "attributes_file": "attributes.txt",
                "min_covering": 2,
                "max_failed_attempts": 15,
                "min_fidelity": 1.0,
                "lowest_min_fidelity": 0.9,
                "dropout_dim": 0.8,
                "dropout_hyp": 0.8,
            }
        },
    )


def get_global_explainer(model, train_data):
    return TabularExplainer(
        explainers=["fidexGloRules"],
        data=train_data,
        model=model,
        mode="classification",
        params={
            "fidexGloRules": {
                "heuristic": 1,
                "with_fidexGlo": True,
                "attributes_file": "attributes.txt",
                "seed": 1,
                "positive_class_index": 0,
                "nb_threads": 4,
                "with_minimal_version": True,
                "max_iterations": 10,
                "min_covering": 2,
                "dropout_dim": 0.5,
                "dropout_hyp": 0.5,
            }
        },
    )


def get_MLPModel(output_path, train_data, test_data, nattributes, nclasses):
    return MLPModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        seed=1,
        # verbose_console=True,
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
    )


def get_dimlpBTModel(output_path, train_data, test_data, nattributes, nclasses):
    return DimlpBTModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        seed=1,
        attributes_file="attributes.txt",
    )


def get_gradBoostModel(output_path, train_data, test_data, nattributes, nclasses):
    return GradBoostModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        seed=1,
        # verbose_console=True,
        # n_estimators=3,
        # learning_rate=32,
        # subsample=0.3,
        # criterion="squared_error",
        # max_depth="no_max_depth",
        # min_samples_split=0.5,
        # min_samples_leaf=0.5,
        # min_weight_fraction_leaf=0.1,
        # max_features=3,
        # max_leaf_nodes=3,
        # min_impurity_decrease=3,
        # init="zero",
        # verbose_scikit=12,
        # warm_start=True,
        # validation_fraction=0.2,
        # n_iter_no_change=35,
        # tol=0.1,
        # ccp_alpha=0.1,
    )


def get_randomForrestModel(output_path, train_data, test_data, nattributes, nclasses):
    return RandomForestModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        seed=1,
        # n_estimators=3,
        # criterion="entropy",
        # max_depth=2,
        # min_samples_split=0.5,
        # min_samples_leaf=0.5,
        # min_weight_fraction_leaf=0.1,
        # max_features=3,
        # max_leaf_nodes=3,
        # min_impurity_decrease=3,
        # bootstrap=True,
        # oob_score=True,
        # n_jobs=-1,
        # verbose_scikit=12,
        # warm_start=True,
        # class_weight={0: 1.2, 1: 2.3},
        # ccp_alpha=0.1,
        # max_samples=2,
    )


def get_SVMModel(output_path, train_data, test_data, nattributes, nclasses):
    return SVMModel(
        output_path,
        train_data,
        test_data,
        nattributes,
        nclasses,
        # output_roc="roc_curve.png",
        # positive_class_index=1,
        # nb_quant_levels=45,
        # K=0.1,
        # C=0.1,
        # kernel="sigmoid",
        # degree=5,
        # gamma=0.4,
        # coef0=1.3,
        # shrinking=False,
        # tol=1,
        # cache_size=150.2,
        # class_weight={0: 1.2, 1: 3.5},
        # max_iterations=50,
        # decision_function_shape="ovo",
        # break_ties=False,
        # verbose_console=True,
    )


def run_tests(output_path, train_data, test_data, nattributes, nclasses):
    model_getters = [
        get_dimlpBTModel,
        get_gradBoostModel,
        get_MLPModel,
        get_randomForrestModel,
        get_SVMModel,
    ]

    for model_getter in model_getters:
        model = model_getter(output_path, train_data, test_data, nattributes, nclasses)
        model.train()

        if isinstance(model, DimlpBTModel):
            print(f"Testing fidex & fidexGloRules with {type(model).__name__}")
            local_explainer = get_local_explainer(model, train_data)
            le = local_explainer.explain(X=test_data)
            le["fidex"].ipython_plot()
        else:
            print(f"Testing fidexGloRules with {type(model).__name__}")

        global_explainer = get_global_explainer(model, train_data)
        ge = global_explainer.explain_global()
        ge["fidexGloRules"].ipython_plot()


if __name__ == "__main__":

    # ensure path exists
    output_path = pl.Path("/tmp/explainer/")
    output_path.mkdir(parents=True, exist_ok=True)
    train_data_file = "train_data.txt"
    test_data_file = "test_data.txt"

    # load data
    train_data, test_data, attributes, classes = load_data(output_path)

    run_tests(output_path, train_data, test_data, len(attributes), len(classes))

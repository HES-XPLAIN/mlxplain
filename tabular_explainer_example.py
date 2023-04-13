import os

import numpy as np
import sklearn
import xgboost
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.visualization.dashboard import Dashboard

from mlxplain.explainers.tabular.agnostic.shap2 import Shap2Tabular

feature_names = [
    "Age",
    "Workclass",
    "fnlwgt",
    "Education",
    "Education-Num",
    "Marital Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours per week",
    "Country",
    "label",
]
data = np.genfromtxt(os.path.join("./data", "adult.data"), delimiter=", ", dtype=str)
tabular_data = Tabular(
    data,
    feature_columns=feature_names,
    categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
    target_column="label",
)
print(tabular_data)

np.random.seed(1)
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    x[:, :-1], x[:, -1], train_size=0.80
)
print("Training data shape: {}".format(train.shape))
print("Test data shape:     {}".format(test.shape))

model = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
model.fit(train, labels_train)
print(
    "Test accuracy: {}".format(
        sklearn.metrics.accuracy_score(labels_test, model.predict(test))
    )
)


def predict_function(z):
    return model.predict_proba(transformer.transform(z))


explainer = Shap2Tabular(
    training_data=tabular_data, predict_function=predict_function, nsamples=100
)
# Apply an inverse transform, i.e., converting the numpy array back to `Tabular`
test_instances = transformer.invert(test)
test_x = test_instances[1653:1655]

# explanations = explainer.explain(test_x)
# print(explanations)
#
# fig = explanations.plot()
# fig.savefig("explanations.png")

# Convert the transformed data back to Tabular instances
train_data = transformer.invert(train)
test_data = transformer.invert(test)

# Initialize a TabularExplainer
explainers = TabularExplainer(
    explainers=["lime", "shap2", "mace", "pdp", "ale"],  # The explainers to apply
    mode="classification",  # The task type
    data=train_data,  # The data for initializing the explainers
    model=model,  # The ML model to explain
    preprocess=lambda z: transformer.transform(
        z
    ),  # Converts raw features into the model inputs
    params={
        "mace": {"ignored_features": ["Sex", "Race", "Relationship", "Capital Loss"]}
    },  # Additional parameters
)

# Generate explanations
test_instances = test_data[:5]
local_explanations = explainers.explain(X=test_instances)
global_explanations = explainers.explain_global(
    params={
        "pdp": {
            "features": [
                "Age",
                "Education-Num",
                "Capital Gain",
                "Capital Loss",
                "Hours per week",
                "Education",
                "Marital Status",
                "Occupation",
            ]
        }
    }
)

# Launch a dashboard for visualization
dashboard = Dashboard(
    instances=test_instances,  # The instances to explain
    local_explanations=local_explanations,  # Set the generated local explanations
    global_explanations=global_explanations,  # Set the generated global explanations
    class_names=class_names,  # Set class names
)
dashboard.show()  # Launch the dashboard

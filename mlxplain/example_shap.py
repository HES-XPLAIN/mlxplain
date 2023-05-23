"""
This script shows how to use our shap tabular method
"""

import os
import sys
# add the path to the 'hes_xplain' folder to the Python path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import pandas as pd
import numpy as np
import shap

from sklearn.ensemble import RandomForestRegressor

from mlxplain.tabular.shap import ShapTabular




# Create a simple tabular dataset
data = pd.DataFrame({
    'feature_1': np.random.rand(12),
    'feature_2': np.random.rand(12),
    'feature_3': np.random.rand(12),
    'feature_4': np.random.rand(12),
    'target': np.random.randint(0, 2, 12)
})


X = data.drop('target', axis=1)
y = data['target']


# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X, y)


# Create a ShapTabular object
shap_explainer = ShapTabular(data=X, model=model)
# Compute SHAP values on all data
shap_values = shap_explainer.explain()

# Create a random instance of the dataset for testing
X_test = pd.DataFrame({
    'feature_1': np.random.rand(1),
    'feature_2': np.random.rand(1),
    'feature_3': np.random.rand(1),
    'feature_4': np.random.rand(1)
})

# Compute SHAP values on one instance
shap_values_X_test = shap_explainer.explain(instance=X_test)

# Get the names of the features in the original dataset
feature_names = list(X.columns)


# Print the SHAP values for each feature
for i in range(4):
    feature_name = feature_names[i]
    shap_value = np.mean(np.abs(shap_values[:, i]))
    print(f"Feature '{feature_name}' has mean absolute SHAP value {shap_value:.2f}")
print(f"Shap value on the test instance is {shap_values_X_test}")


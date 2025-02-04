###########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_housing_data, CombinedAttributesAdder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

###########################################
housing = load_housing_data()

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


###########################################
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

############################################
# Discover and Visualize the data to gain insights
# No validation in this case

housing = strat_train_set.copy()

# Transformation Pipelines

num_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ]
)

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, list(housing.drop("ocean_proximity", axis=1))),
        ("cat", OneHotEncoder(), ["ocean_proximity"]),
    ]
)
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape

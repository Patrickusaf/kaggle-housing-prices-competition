# Kaggle - Housing Prices Competition
# 30/03/2022 by Patrick Svensson
# Code based on Kaggle Courses

# IMPORTS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# READ DATA - INPUTS
# save filepath to variable for easier access
melbourne_file_path = "data/melb_data.csv"
# read the data and store in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data (print for console)
print(melbourne_data.describe())
# print data columns
print(melbourne_data.columns)

# DATA PREPROCESSING
# you can drop missing values if is needed
# melbourne_data = melbourne_data.dropna(axis=0)

# SELECTING DATA FOR MODELING
# selecting the prediction target
y = melbourne_data.Price

# choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

# BUILDING THE MODEL
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = RandomForestRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
melb_preds = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

# UNDERFITTING AND OVERFITTING
def get_mae_dtr(max_leaf_nodes, train_X, val_X, train_y, val_y):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes (only for DecistionTreeRegressor)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae_dtr(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
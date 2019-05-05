from __future__ import print_function
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
# Mute the setting with a copy warnings
pd.options.mode.chained_assignment = None

# Get data from csv file
data_path = ['../data']

# Import the data and examine the shape.
# There are 79 features columns and 1 predictor
# There are 3 different types: integers (int64), floats (float64) and strings (object)
filepath = os.sep.join(data_path + ['Ames_Housing_Sales.csv'])
data = pd.read_csv(filepath)
# print (data.shape) # (1379, 80)
# print (data.dtypes.value_counts())

# Select the object (string) columns
mask = data.dtypes == np.object
categorical_cols = data.columns[mask]

# Determine how many extra columns would be created
num_ohc_cols = (data[categorical_cols]
                .apply(lambda x: x.nunique())
                .sort_values(ascending=False))

# No need to encode if there is only one value
small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

# Number of one-hot columns is one less than the number of categories
small_num_ohc_cols -= 1

# Calculate the total extra columns
# print (small_num_ohc_cols.sum())    #215

# Create a new data set with all already one-hot encoded categorical features
# Copy the data set
data_ohc = data.copy()

# The encoders
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()

for col in num_ohc_cols.index:
    # Integer the string categories
    dat = label_encoder.fit_transform(data_ohc[col]).astype(np.int)

    # Remove the original column
    data_ohc = data_ohc.drop(col, axis=1)

    # One hot encode the data - this returns a sparse array
    new_dat = one_hot_encoder.fit_transform(dat.reshape(-1,1))

    # Create unique column names
    n_cols = new_dat.shape[1]
    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]

    # Create new dataframe
    new_df = pd.DataFrame(new_dat.toarray(), index=data_ohc.index, columns=col_names)

    # Append the new data to the dataframe
    data_ohc = pd.concat([data_ohc, new_df], axis=1)

#print (data.shape[1])
# Remove the string columns from the dataframe
data = data.drop(num_ohc_cols.index, axis=1)
#print (data.shape[1])

# Create train and test splits of both data sets (the not one-hot encoded and the encoded)
y_col = 'SalePrice'

# Split the data that is not one-hot encoded
feature_cols = [x for x in data.columns if x != y_col]
X_data = data[feature_cols]
y_data = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# Split the data that is one-hot encoded
feature_cols = [x for x in data_ohc.columns if x != y_col]
X_data_ohc = data_ohc[feature_cols]
y_data_ohc = data_ohc[feature_cols]

X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, test_size=0.3, random_state=42)

# Compare the indices to ensure they are identical
print ((X_train_ohc.index == X_train.index).all())
print('=' * 80)

linear_regression = LinearRegression()

# Storage of error value
error_df = list()

# Data that have not been one-hot encoded
linear_regression = linear_regression.fit(X_train, y_train)
y_train_predicted = linear_regression.predict(X_train)
y_test_predicted = linear_regression.predict(X_test)

error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_predicted),
                           'test': mean_squared_error(y_test, y_test_predicted)},
                          name='no encoded'))

# Data that have been one-hot encoded
linear_regression = linear_regression.fit(X_train_ohc, y_train_ohc)
y_train_ohc_predicted = linear_regression.predict(X_train_ohc)
y_test_ohc_predicted = linear_regression.predict(X_test_ohc)

error_df.append(pd.Series({'train': mean_squared_error(y_train_ohc, y_train_ohc_predicted),
                           'test': mean_squared_error(y_test_ohc, y_test_ohc_predicted)},
                          name='one-hot encoded'))

# Assemble the result
error_df = pd.concat(error_df, axis=1)
print ("Assemble the result")
print (error_df)
print('=' * 80)

# Scale all the non-hot encoded values using one of the following method: StandardScaler, MinMaxScaler, MaxAbsScaler
scalers = {'standard': StandardScaler(),
           'minmax': MinMaxScaler(),
           'maxabs': MaxAbsScaler()}

training_test_sets = {
    'not_encoded': (X_train, y_train, X_test, y_test),
    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}

# Get the list of float columns, and the float data
mask = X_train.dtypes == np.float
float_columns = X_train.columns[mask]

# initialize model
linear_regression = LinearRegression()

# iterate over all possible combinations and get the errors
errors = {}
for encoding_label, (_X_train, _y_train, _X_test, _y_test) in training_test_sets.items():
    for scaler_label, scaler in scalers.items():
        trainingset = _X_train.copy()  # copy because we dont want to scale this more than once.
        testset = _X_test.copy()
        trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])
        testset[float_columns] = scaler.transform(testset[float_columns])
        linear_regression.fit(trainingset, _y_train)
        predictions = linear_regression.predict(testset)
        key = encoding_label + ' - ' + scaler_label + 'scaling'
        errors[key] = mean_squared_error(_y_test, predictions)

errors = pd.Series(errors)
print(errors.to_string())
print('-' * 80)
for key, error_val in errors.items():
    print(key, error_val)

# Plot predictions and actual values for one of the model
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

ax = plt.axes()
ax.scatter(y_test, y_test_predicted, alpha=0.5)

ax.set(xlabel='Ground truth',
       ylabel='Predictions',
       title='Ames, Iowa House price Predictions vs Truth, using Linear Regression')

plt.show()

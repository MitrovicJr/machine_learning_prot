import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Read the data
df = pd.read_csv('breast-cancer.csv.xls')

# Convert 'diagnosis' column to binary (0 for 'B' and 1 for 'M')
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

# Plot histograms for features
for label in df.columns[2:]:
    plt.hist(df[df['diagnosis'] == 1][label], color='blue', label='Malignant', alpha=0.7, density=True)
    plt.hist(df[df['diagnosis'] == 0][label], color='red', label='Benign', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Split the data into train, valid, and test sets
# Use random_state for reproducibility
train = df.sample(frac=0.6, random_state=42)
remaining = df.drop(train.index)
valid = remaining.sample(frac=0.2, random_state=42)
test = remaining.drop(valid.index)

# Define a function to scale the dataset and perform oversampling
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[2:]].values
    y = dataframe[dataframe.columns[1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

# Scale and oversample the datasets
train_data, X_train, y_train = scale_dataset(train, oversample=False)
valid_data, X_valid, y_valid = scale_dataset(valid, oversample=False)
test_data, X_test, y_test = scale_dataset(test, oversample=False)

# Check the lengths of the datasets
print("Length of train_data:", len(train_data))
print("Length of valid_data:", len(valid_data))
print("Length of test_data:", len(test_data))

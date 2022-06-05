import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
TASK 1:- Data Pre-processing and Data Exploration
"""
# a. Load the data and report the number of rows in the dataset.
DataFrame = pd.read_csv('Dataset/winequality-white.csv', sep=';')
print("Number of rows in the dataset: ", DataFrame.shape[0])

# b. Report the number of features in the dataset and the number of data points in each class
print("Number of features in the dataset: ", DataFrame.shape[1] - 1)
print("Number of data points in each class: ")
print(DataFrame['quality'].value_counts())

# c. Perform random permutation using sklearn.utils.shuffle and set a value for the random_state
white_wine = shuffle(DataFrame, random_state=10)

# d. Produce scatter plot using any two features from the dataset.
fig, ax = plt.subplots()
ax.scatter(white_wine['chlorides'], white_wine['pH'], color='blue', label='Points')
ax.set_xlabel('Chlorides')
ax.set_ylabel('pH')
ax.set_title('Scatter Plot of Chlorides v/s pH of wine')
ax.legend()
plt.show()

"""
TASK 2:- PCA Analysis
"""

# a. Perform PCA on the whole white_wine dataset
labels = white_wine['quality']
features = white_wine.drop(columns=['quality'])
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

pca = PCA(n_components=2)
features = pca.fit_transform(features)
new_df = pd.DataFrame(features, columns=['PC1', 'PC2'])
new_df['quality'] = labels

# b. Plot PC1 and PC2 and label them according to their class labels.
fig, ax = plt.subplots()
ax.scatter(new_df[new_df['quality'] == 6]['PC1'], new_df[new_df['quality'] == 6]['PC2'], color='b', label='Quality = 6')
ax.scatter(new_df[new_df['quality'] == 5]['PC1'], new_df[new_df['quality'] == 5]['PC2'], color='g', label='Quality = 5')
ax.scatter(new_df[new_df['quality'] == 7]['PC1'], new_df[new_df['quality'] == 7]['PC2'], color='r', label='Quality = 7')
ax.scatter(new_df[new_df['quality'] == 8]['PC1'], new_df[new_df['quality'] == 8]['PC2'], color='c', label='Quality = 8')
ax.scatter(new_df[new_df['quality'] == 4]['PC1'], new_df[new_df['quality'] == 4]['PC2'], color='m', label='Quality = 4')
ax.scatter(new_df[new_df['quality'] == 3]['PC1'], new_df[new_df['quality'] == 3]['PC2'], color='y', label='Quality = 3')
ax.scatter(new_df[new_df['quality'] == 9]['PC1'], new_df[new_df['quality'] == 9]['PC2'], color='k', label='Quality = 9')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Scatter Plot of PC1 v/s PC2')
ax.legend()
plt.show()

# c. Report the variance captured by each principal component.
print('Variance of PC1: ', pca.explained_variance_ratio_[0])
print('Variance of PC2: ', pca.explained_variance_ratio_[1])

"""
TASK 3:- Divide the white wine dataset into training set, validation set and test set.
"""

# a. First 1000 rows - validation set
validation_set = white_wine.iloc[0:1000, :]

# b. Last 1000 rows - test set
test_set = white_wine.iloc[-1000:, :]

# c. Remaining rows - training set
training_set = white_wine.iloc[1000:-1000, :]
print(training_set.shape)

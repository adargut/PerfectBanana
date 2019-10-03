from sklearn import preprocessing
import numpy as np

data = np.array([[3, -1.5, 5], [5, 7, 8], [2.5, 3.4, -2.7]])
# print(data)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
print(data)

data = np.array([[3, -1.5, 5], [5, 7, 8], [2.5, 3.4, -2.7]])
normalized = preprocessing.normalize(data, 'l1')
print("normalized data =", normalized) # values sum to 1, normalized by l1 norm

binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print(binarized)
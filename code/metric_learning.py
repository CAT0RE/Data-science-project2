import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import neighbors
from metric_learn import LMNN

features = np.loadtxt('C:/Users/123/OneDrive/桌面/practice2/ResNet101'
                      '/AwA2-features.txt')
labels = np.loadtxt('C:/Users/123/OneDrive/桌面/practice2/ResNet101'
                    '/AwA2-labels.txt')

sample_size = int(features.shape[0] * 0.3)
random_indices = np.random.choice(features.shape[0], sample_size, replace=False)
features_small = features[random_indices, :]
labels_small = labels[random_indices]

train_features, test_features, train_labels, test_labels = train_test_split(features_small, labels_small, test_size=0.4,
                                                                            random_state=42)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

print(train_features.shape, train_labels.shape)
print(test_features.shape, test_labels.shape)

#PCA降维
pca = PCA(n_components=300)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

print(train_features.shape, train_labels.shape)
print(test_features.shape, test_labels.shape)

# LMNN
lmnn = LMNN(k=6, init='pca', random_state=42, verbose=1.5)
train_features_lmnn = lmnn.transform(train_features)
test_features_lmnn = lmnn.transform(test_features)

# KNN with learned metric
best_knn_lmnn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance',
                                               metric='euclidean')
best_knn_lmnn.fit(train_features_lmnn, train_labels)
pre_labels_lmnn = best_knn_lmnn.predict(test_features_lmnn)
accuracy_lmnn = accuracy_score(test_labels, pre_labels_lmnn)
print('Final Accuracy with LMNN (euclidean):', accuracy_lmnn)

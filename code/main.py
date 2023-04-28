import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import neighbors

features = np.loadtxt('C:/Users/123/OneDrive/桌面/practice2/ResNet101'
                      '/AwA2-features.txt')
labels = np.loadtxt('C:/Users/123/OneDrive/桌面/practice2/ResNet101'
                    '/AwA2-labels.txt')

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.4,
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

best_knn_1 = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance',
                                            metric='euclidean')
best_knn_1.fit(train_features, train_labels)
pre_labels_1 = best_knn_1.predict(test_features)
accuracy_1 = accuracy_score(test_labels, pre_labels_1)
print('Final Accuracy(euclidean):', accuracy_1)

best_knn_2 = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance',
                                            metric='manhattan')
best_knn_2.fit(train_features, train_labels)
pre_labels_2 = best_knn_2.predict(test_features)
accuracy_2 = accuracy_score(test_labels, pre_labels_2)
print('Final Accuracy(manhattan):', accuracy_2)

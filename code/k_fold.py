import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, KFold

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

k_range = list(range(1, 20))
param_grid = {'n_neighbors': k_range}
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

knn_1 = neighbors.KNeighborsClassifier(weights='distance', metric='euclidean')
grid_search = GridSearchCV(knn_1, param_grid, cv=k_fold, scoring='accuracy', return_train_score=True)
grid_search.fit(train_features, train_labels)
# 打印最优k值及其对应的准确率
print("euclidean最优k值:", grid_search.best_params_['n_neighbors'])
print("euclidean最优k值对应的准确率:", grid_search.best_score_)

k_range = list(range(1, 20))
param_grid = {'n_neighbors': k_range}
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

knn_2 = neighbors.KNeighborsClassifier(weights='distance', metric='manhattan')
grid_search = GridSearchCV(knn_2, param_grid, cv=k_fold, scoring='accuracy', return_train_score=True)
grid_search.fit(train_features, train_labels)
# 打印最优k值及其对应的准确率
print("manhattan最优k值:", grid_search.best_params_['n_neighbors'])
print("manhattan最优k值对应的准确率:", grid_search.best_score_)


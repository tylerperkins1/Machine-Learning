from sklearn.neighbors import KNeighborsClassifier


def knn(x,y):

    print("KNN, Algorithm: KD Tree, weights = distance")
    neigh1 = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree', weights='distance')
    neigh1.fit(x,y)
    score = (neigh1.score(x,y))
    print("Accuracy = ", score)

    print("KNN, Algorithm: Ball Tree, weights = distance")
    neigh2 = KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree', weights='distance')
    neigh2.fit(x, y)
    score = (neigh2.score(x, y))
    print("Accuracy = ", score)

    print("KNN, Algorithm: KD Tree, weights = uniform")
    neigh3 = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree')
    neigh3.fit(x, y)
    score = (neigh3.score(x, y))
    print("Accuracy = ", score)

    print("KNN, Algorithm: Ball Tree, weights = uniform")
    neigh3 = KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree')
    neigh3.fit(x, y)
    score = (neigh3.score(x, y))
    print("Accuracy = ", score)

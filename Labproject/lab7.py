import cv2
import numpy as np
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_data():
    images = glob.glob(r'./ORL/*/*.pgm')
    train_set = []
    train_target = []
    test_set = []
    test_target = []
    for i, img in enumerate(images):
        if i % 10 < 5:
            img = cv2.imread(img, 0)
            img_flip = cv2.flip(img, 1)
            if i % 10 == 1:
                cv2.imshow('Face', img)
                cv2.waitKey(0)
                cv2.imshow('Face', img_flip)
                cv2.waitKey(0)
            temp1 = np.resize(img, (img.shape[0] * img.shape[1], 1))
            temp2 = np.resize(img_flip, (img_flip.shape[0] * img_flip.shape[1], 1))
            train_set.append(temp1.T)
            train_set.append(temp2.T)
            train_target.append(i // 10)
            train_target.append(i // 10)
        else:
            img = cv2.imread(img, 0)
            temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
            # train_set.append(temp.T)
            # train_target.append(i // 10)
            test_set.append(temp.T)
            test_target.append(i // 10)
    train_set = np.array(train_set).squeeze()
    test_set = np.array(test_set).squeeze()
    train_target = np.array(train_target).squeeze()
    test_target = np.array(test_target).squeeze()
    return train_set, test_set, train_target, test_target


def calculate_covariance_matrix(data):
    m = data.shape[0]
    data = data - np.mean(data, axis=0)
    return 1 / m * np.matmul(data.T, data)


def transform(train_set, k):
    conv = calculate_covariance_matrix(train_set.T)
    eigenvalues, eigenvectors = np.linalg.eig(conv)
    eigenvectors = np.dot(train_set.T, eigenvectors)
    idx = np.argsort(eigenvalues[::-1])
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :k]
    return eigenvectors


# def pca_classifier(train_set, test_set, train_target, test_target, k):
#     w = transform(train_set, k)
#     y = np.matmul(train_set, w)
#     embeddings = np.matmul(test_set, w)
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(y, train_target)
#     test_acc = knn.score(embeddings, test_target)
#     print("test accuracy by PCA", test_acc)


def pca_classifier(train_set, test_set, train_target, test_target, k):
    pca = PCA(n_components=k)
    y = pca.fit_transform(train_set)
    embeddings = pca.transform(test_set)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(y, train_target)
    test_acc = knn.score(embeddings, test_target)
    rtest_set = pca.inverse_transform(embeddings)
    loss = np.linalg.norm(test_set - rtest_set) / (np.linalg.norm(test_set) + np.linalg.norm(rtest_set))
    ratio = np.sum(pca.explained_variance_ratio_)
    print("test accuracy and loss by PCA", test_acc, loss)
    return loss, ratio, test_acc


def mda_classifier(train_set, test_set, train_target, test_target, k):
    lda = LinearDiscriminantAnalysis(n_components=k)
    lda.fit(train_set, train_target)
    y = lda.transform(train_set)
    embeddings = lda.transform(test_set)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(y, train_target)
    test_acc = knn.score(embeddings, test_target)
    # print("test accuracy by MDA", test_acc)


def dpdr_classifier(train_set, test_set, train_target, test_target, k):
    w = transform(train_set, 400)
    y = np.matmul(train_set, w)
    embeddings = np.matmul(test_set, w)
    lda = LinearDiscriminantAnalysis(n_components=k, solver='svd')
    lda.fit(y, train_target)
    y = lda.transform(y)
    embeddings = lda.transform(embeddings)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(y, train_target)
    test_acc = knn.score(embeddings, test_target)
    # print("test accuracy by DPDR", test_acc)


def main():
    train_set, test_set, train_target, test_target = get_data()
    print('training set shape:', train_set.shape, 'test set shape:', test_set.shape)
    # print(train_target.shape, test_target.shape)
    loss_list = []
    ratio_list = []
    acc_list = []
    d = 0
    for i in range(50):
        d = d + 1
        loss, ratio, acc = pca_classifier(train_set, test_set, train_target, test_target, d)
        loss_list.append(loss)
        ratio_list.append(ratio)
        acc_list.append(acc)
    plt.plot(ratio_list, loss_list)
    plt.xlabel("Energy Ratio")
    plt.ylabel("ETE")
    plt.show()
    plt.plot(ratio_list, acc_list)
    plt.xlabel("Energy Ratio")
    plt.ylabel("Accuracy")
    plt.show()
    # mda_classifier(train_set, test_set, train_target, test_target, 39)
    # dpdr_classifier(train_set, test_set, train_target, test_target, 39)


if __name__ == '__main__':
    main()



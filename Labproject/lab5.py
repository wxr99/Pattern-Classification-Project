import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data as data
import matplotlib.colors as colors
from sklearn.neighbors import KNeighborsClassifier


class Sythetic_Dataset(data.Dataset):
    '''
    a man made dataset following g distribution
    '''
    def __init__(self, datanum1, datanum2):
        self.datanum = int(datanum1) + int(datanum2)
        self.classnum = 2
        '''
        parameter setting
        '''
        mean1 = np.array([-1, 0])
        mean2 = np.array([1, 0])
        cov = np.array([[1, 0], [0, 1]])
        '''
        create data
        '''
        ini_point = np.random.multivariate_normal(mean1, cov, int(datanum1))
        ini_label = 0 * np.ones(int(datanum1))
        ini_point = np.append(ini_point, np.random.multivariate_normal(mean2, cov, int(datanum2)), 0)
        ini_label = np.append(ini_label, 1 * np.ones(int(datanum2)), 0)
        self.point = ini_point
        self.label = ini_label

    def __getitem__(self, index):
        point, label = self.point[index], self.label[index]
        return point, label

    def __len__(self):
        return len(self.label)


def main():

    # color = np.array(["red", "green", "black", "orange", "purple", "beige", "cyan", "magenta"])
    # plt.scatter(train_set.point[:, 0], train_set.point[:, 1], c=train_set.label, cmap=colors.ListedColormap(color))
    # plt.show()
    for i in range(10):
        train_set = Sythetic_Dataset(datanum1=50-i, datanum2=40+i)
        validation_set = Sythetic_Dataset(datanum1=i, datanum2=10-i)
        train_ratio = 50-i/90
        test_ratio = i/10
        test_set = Sythetic_Dataset(datanum1=10, datanum2=10)
        best_k = 1
        k = 1
        best_val_acc = 0
        best_test_acc = 0
        print(i+1, "th test: ", "train ratio:", train_ratio, "test ratio:", test_ratio)
        while k*k <= train_set.datanum:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_set.point, train_set.label)
            validation_acc = knn.score(validation_set.point, validation_set.label)
            test_acc = knn.score(test_set.point, test_set.label)
            print("k=", k,
                  "acc for validation set=", validation_acc,
                  "acc for test set=", knn.score(test_set.point, test_set.label))
            k += 2


if __name__ == '__main__':
    main()
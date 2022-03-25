import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
max_iteration = 1000


def split1(iris):
    x = iris.data
    y = iris.target
    seed1 = np.random.random_integers(50, 99, 50)
    seed2 = np.random.random_integers(100, 149, 50)
    x1 = x[seed1]
    x2 = x[seed2]
    y1 = y[seed1]
    y2 = y[seed2] - np.ones(50) * 3
    x_train = np.vstack((x1[0:25], -1*x2[0:25]))
    y_train = np.vstack((y1[0:25], y2[0:25])).reshape(50)
    x_test = np.vstack((x1[25:50], -1*x2[25:50]))
    y_test = np.vstack((y1[25:50], y2[25:50])).reshape(50)
    return x_train, y_train, x_test, y_test


def batch_perceptron(x_train, lr, threshold):
    w = 2*np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    for i in range(max_iteration):
        score = np.dot(x_train, w)
        error_index = np.where( score <= 0)
        w += lr*np.sum(x_train[error_index], axis=0)
        if np.absolute(lr*np.sum(x_train[error_index], axis=0)).sum() < threshold:
            break
    return w


def single_sample_perceptron(x_train):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    restart = True
    loop = 0
    while restart:
        loop += 1
        if loop > max_iteration:
            print("Exceed max iterations!")
            break
        for i, sample in enumerate(x_train):
            score = np.dot(sample, w)
            if i == len(x_train)-1:
                restart = False
            if score <= 0:
                w += sample
                break
    return w


def margin_single_sample_perceptron(x_train, lr, margin):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    restart = True
    loop = 0
    while restart:
        loop += 1
        if loop > max_iteration:
            print("Exceed max iterations!")
            break
        for i, sample in enumerate(x_train):
            score = np.dot(sample, w)
            if i == len(x_train)-1:
                restart = False
            if score <= margin:
                w += lr*sample
                break
    return w


def batch_incremental_perceptron(x_train, lr):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    restart = True
    loop = 0
    while restart:
        loop += 1
        if loop > max_iteration:
            print("Exceed max iterations!")
            break
        score = np.dot(x_train, w)
        error_index = np.where(score <= 0)
        if len(error_index[0]) == 0:
            break
        w += lr * np.sum(x_train[error_index], axis=0)
    return w


def balanced_window(x_train):
    w1 = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    w2 = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    for i, sample in enumerate(x_train):
        score1 = np.dot(sample, w1)
        score2 = np.dot(sample, w2)
        if abs(score1 - score2) / (score1 - score2) == 1:
            w1 = 0.5*w1
            w2 = 1.5*w2
        if abs(score1 - score2) / (score1 - score2) == -1:
            w1 = 1.5*w1
            w2 = 0.5*w2
    return w1, w2


def margin_relaxation_single_sample_perceptron(x_train, lr, margin):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    restart = True
    loop = 0
    while restart:
        loop += 1
        if loop > max_iteration:
            print("Exceed max iterations!")
            break
        for i, sample in enumerate(x_train):
            score = np.dot(sample, w)
            if i == len(x_train) - 1:
                restart = False
            if score <= margin:
                k = lr * (margin-score) / (np.linalg.norm(sample, ord=2) * np.linalg.norm(sample, ord=2))
                w += k * sample
                break
    return w


def LMS(x_train, threshold, margin):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    b = margin * np.ones(len(x_train))
    for i in range(max_iteration):
        score = np.dot(x_train, w)
        k = 0.01 / (i+1) * (margin - score)
        w += np.dot(k, x_train)
        if np.absolute(np.dot(k, x_train)).sum() < threshold:
            break
    return w


def Ho_Kashyap(x_train, margin, threshold):
    w = 2 * np.random.random(len(x_train[0])) - np.ones(len(x_train[0]))
    b = margin * np.ones(len(x_train))
    for i in range(max_iteration):
        e = np.dot(x_train, w) - b
        e_plus = 0.5 * (e + np.absolute(e))
        b = b + 2 * 1.1/(i+1) * e_plus
        w = np.dot(np.linalg.pinv(x_train), b)
        if np.absolute(e).sum() < threshold:
            print(i)
            break
        if i == max_iteration - 1:
            print("Exceed max iterations!")
    return w


def Sim_Ho_Kashyap(x_train, margin, threshold):
    b = margin * np.ones(len(x_train))
    w = np.dot(np.linalg.pinv(x_train), b)
    for i in range(max_iteration):
        e = np.dot(x_train, w) - b
        e_plus = 0.5 * (e + np.absolute(e))
        b = b + 2 * 1.1/(i+1) * e_plus
        w = np.dot(np.linalg.pinv(x_train), b)
        if np.absolute(e).sum() < threshold:
            print(i)
            break
        if i == max_iteration - 1:
            print("Exceed max iterations!")
    return w


def train(iter):
    acc = np.zeros(iter)
    for i in range(iter):
        x_train, y_train, x_test, y_test = split1(iris)
        # w = batch_perceptron(x_train, lr=0.1, threshold=0.5)
        # w = single_sample_perceptron(x_train)
        # w = margin_single_sample_perceptron(x_train, lr=0.1, margin=0)
        # w = batch_incremental_perceptron(x_train, lr=0.1)
        # w = balanced_window(x_train)
        # note that in margin_relaxation_single_sample_perceptron: lr>1
        # w = margin_relaxation_single_sample_perceptron(x_train, lr=1.5, margin=5)
        # w = LMS(x_train, threshold=0.5, margin=10)
        # w = Ho_Kashyap(x_train, margin=5, threshold=5)
        w = Sim_Ho_Kashyap(x_train, margin=5, threshold=5)
        acc[i] = 1 - np.sum(np.dot(x_test, w) <= 0) / len(x_test)
    mean = np.mean(acc)
    var = np.var(acc)
    print("mean accuracy:", mean, "variance:", var)


def main():
    train(iter=1)


if __name__ == '__main__':
    main()
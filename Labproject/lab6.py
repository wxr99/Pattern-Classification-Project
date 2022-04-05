import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

max_iteration = 1000
threshold = 0.1
raw_data = np.array([[-7.82,-4.58,-3.97],
                 [-6.68,3.16,2.71],
                 [4.36,-2.19,2.09],
                 [6.72,0.88,2.80],
                 [-8.64,3.06,3.50],
                 [-6.87,0.57,-5.45],
                 [4.47,-2.62,5.76],
                 [6.73,-2.01,4.18],
                 [-7.71,2.34,-6.33],
                 [-6.91,-0.49,-5.68],
                 [6.18,2.81,5.82],
                 [6.72,-0.93,-4.04],
                 [-6.25,-1.22,1.13],
                 [-6.94,-1.22,1.13],
                 [8.09,0.20,2.25],
                 [6.81,0.17,-4.15],
                 [-5.19,4.24,4.04],
                 [-6.38,-1.74,1.43],
                 [4.08,1.30,5.33],
                 [6.27,0.93,-2.78]] ,dtype=float)


def init_center(k, data):
    # seed = np.random.random_integers(0, 19, k)
    # center = data[seed]
    # center = np.array([[1,1,1], [-1,1,-1]],dtype=float)
    # center = np.array([[0, 0, 0], [1, 1, -1]], dtype=float)
    # center = np.array([[0, 0, 0], [1, 1, 1], [-1, 0, 2]], dtype=float)
    center = np.array([[-0.1, 0, 0.1], [0, -0.1, 0.1], [-0.1, -0.1, 0.1]], dtype=float)
    print("Starting Points:\n", center)
    return center


def get_partition(center, data):
    distance = np.ones((len(data), len(center)))
    for i in range(len(data)):
        for j in range(len(center)):
            # distance[i][j] = np.dot((data[i]-center[j]), (data[i]-center[j]))
            distance[i][j] = 1 - np.exp(-10 * np.dot((data[i]-center[j]), (data[i]-center[j])))
    index = np.argmin(distance, axis=1)
    return index


def get_membership(center, data, b):
    distance = np.ones((len(data), len(center)))
    membership = np.ones((len(data), len(center)))
    for i in range(len(data)):
        for j in range(len(center)):
            # distance[i][j] = np.dot((data[i]-center[j]), (data[i]-center[j]))
            distance[i][j] = 1 - np.exp(-0.1 * np.dot((data[i] - center[j]), (data[i] - center[j])))
            membership[i][j] = distance[i][j] ** (-1 / (b - 1))
    for i in range(len(data)):
        sum = membership[i].sum()
        for j in range(len(center)):
            membership[i][j] = membership[i][j] / sum
    return membership


def get_center(data, membership, b):
    center = np.ones((len(membership[0]), len(data[0])))
    for i in range(len(data)):
        for j in range(len(center)):
            membership[i][j] = membership[i][j]**b
    for i in range(len(center)):
        sum1 = membership[:, i].sum()
        sum2 = np.zeros(len(data[0]))
        for j in range(len(data)):
            sum2 = sum2 + membership[j][i] * data[j]
        center[i] = sum2 / sum1
    return center


def refresh_center(index, data, k):
    center = np.ones((k, len(data[0])))
    for i in range(k):
        cluster = data[np.where(index == i)]
        center[i] = np.average(cluster, axis=0)
    return center


def k_means(data, k):
    center = init_center(k, data)
    iteration = 0
    for i in range(max_iteration):
        iteration += 1
        index = get_partition(center, data)
        new_center = refresh_center(index, data, k)
        if np.absolute(center - new_center).sum() < threshold:
            break
        center = new_center
    return center, index, iteration


def fcm(data, c, b):
    center = init_center(c, data)
    iteration = 0
    for i in range(max_iteration):
        iteration += 1
        membership = get_membership(center, data, b)
        new_center = get_center(data, membership, b)
        if abs((center - new_center).max()) < threshold:
            center = new_center
            break
        center = new_center
    index = np.argmax(membership, axis=1)
    return center, index, iteration


def main():
    # center, label, iteration = k_means(data=raw_data, k=2)
    center, label, iteration = fcm(data=raw_data, c=3, b=2)
    print("center:\n", center, "\nlabel:\n", label, "\niteration:\n", iteration)
    color = np.array(["red", "green", "black", "orange", "purple", "beige", "cyan", "magenta"])
    fig = plt.figure()
    ax = Axes3D(fig)
    x = raw_data[:, 0]
    y = raw_data[:, 1]
    z = raw_data[:, 2]
    ax.scatter(x, y, z, c=label, cmap=colors.ListedColormap(color))
    ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    plt.show()


if __name__ == '__main__':
    main()
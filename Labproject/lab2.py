import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *


raw_data_2= np.array([[-0.4,-0.31,0.38,-0.15,-0.35,0.17,-0.011,-0.27,-0.065,-0.12],
                  [0.58,0.27,0.055,0.53,0.47,0.69,0.55,0.61,0.49,0.054],
                  [0.089,-0.04,-0.035,0.011,0.034,0.1,-0.18,0.12,0.0012,-0.063]],dtype=np.float)
raw_data_3= np.array([[0.83,1.1,-0.44,0.047,0.28,-0.39,0.34,-0.3,1.1,0.18],
                  [1.6,1.6,-0.41,-0.45,0.35,-0.48,-0.079,-0.22,1.2,-0.11],
                  [-0.014,0.48,0.32,1.4,3.1,0.11,0.14,2.2,-0.46,-0.49]],dtype=np.float)
data_2 = np.transpose(raw_data_2)
data_3 = np.transpose(raw_data_3)
w_sample = np.array([1.0,2.0,-1.5],dtype=np.float)

def means(data):
    mean = np.average(data, axis=0)
    return mean


def within_scatter(data, mean):
    total = np.zeros([len(data[0]), len(data[0])])
    for x in data:
        # print(x,mean)
        total = total + np.outer((x - mean), (x - mean))
    return total


def between_scatter(mean1, mean2):
    total = np.outer((mean1, mean2), (mean1, mean2))
    return total


def get_projection(data1, data2, mean1, mean2):
    S_w = within_scatter(data1, mean1) + within_scatter(data2, mean2)
    return np.dot(np.linalg.inv(S_w), (mean1-mean2))


def get_variance(data, mean):
    total = 0
    for x in data:
        total += (x-mean)*(x-mean)
    return total/len(data)


def get_decision(mean1, var1, mean2, var2):
    x = Symbol('x')
    s = solve(0.5*(x-mean1)*(x-mean1)/var1-0.5*(x-mean2)*(x-mean2)/var2+0.5*ln(var1/var2), x)
    return s


def main():
    global data_2, data_3, raw_data_2, raw_data_3, w_sample
    print(data_2)
    mean_2 = means(data_2)
    mean_3 = means(data_3)
    w = get_projection(data_2,data_3,mean_2,mean_3)
    new_data_2 = np.dot(data_2,w)
    new_data_3 = np.dot(data_3,w)
    sample_data_2 = np.dot(data_2, w_sample)
    sample_data_3 = np.dot(data_3, w_sample)
    print("===========================================================")
    print("Problem (a) (b) (c) : get a projection")
    print("w:", w)
    print("new_data_2:", new_data_2)
    print("new_data_3:", new_data_3)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(raw_data_2[0], raw_data_2[1], raw_data_2[2], c='r', label='cat1')
    ax.scatter(raw_data_3[0], raw_data_3[1], raw_data_3[2], c='g', label='cat2')
    ax.legend(loc='best')
    ax.quiver(0,0,0,4*w[0],4*w[1],4*w[2],arrow_length_ratio=0.1)
    ax.quiver(0,0,0,0.8*w_sample[0],0.8*w_sample[1],0.8*w_sample[2],arrow_length_ratio=0.1)
    plt.show()
    print("===========================================================")
    print("Problem (d) (e) (f): get a projection")
    print("w_sample:", w_sample)
    print("sample_data_2:", sample_data_2)
    print("sample_data_3:", sample_data_3)
    new_mean_2 = means(new_data_2)
    new_mean_3 = means(new_data_3)
    var_2 = get_variance(new_data_2, new_mean_2)
    var_3 = get_variance(new_data_3, new_mean_3)
    decision_point1 = get_decision(new_mean_2,var_2,new_mean_3,var_3)[1]
    print("decision_point1:", decision_point1)

    y_axis1 = np.ones(10)
    y_axis2 = np.zeros(10)

    fig = plt.figure()
    plt.scatter(new_data_2, y_axis1, c='r', label='cat1')
    plt.scatter(new_data_3, y_axis2, c='g', label='cat2')
    plt.axvline(decision_point1)
    plt.show()

    sample_mean2 = means(sample_data_2)
    sample_mean3 = means(sample_data_3)
    sample_var2 = get_variance(sample_data_2, sample_mean2)
    sample_var3 = get_variance(sample_data_3, sample_mean3)
    decision_point2 = get_decision(sample_mean2,sample_var2,sample_mean3,sample_var3)[0]
    print("decision_point2:", decision_point2)

    fig = plt.figure()
    plt.scatter(sample_data_2, y_axis1, c='r', label='cat1')
    plt.scatter(sample_data_3, y_axis2, c='g', label='cat2')
    plt.axvline(decision_point2)
    plt.show()


if __name__ == '__main__':

    main()
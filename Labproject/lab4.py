import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(-1, 1, 100)
bias = np.zeros(100)
variance = np.zeros(100)
y_hat = np.zeros((100, 100))
y = np.zeros((100, 100))


def function_1():
    print("\nfixed function1:\n")
    for i in range(100):
        gaussian_blur = np.random.normal(0, 0.1, 100)
        y[i] = x ** 2 + gaussian_blur
        y_hat[i] = 0.5
    y_average = np.average(y_hat, axis=1)
    for i in range(100):
        bias[i] = np.dot((y_hat[i] - y[i]), (y_hat[i] - y[i]))
        variance[i] = np.dot((y_hat[i] - y_average), (y_hat[i] - y_average))
    n = np.linspace(1, 100, 100)
    plt.scatter(n, bias, c="red", marker='x', s=5, label='bias')
    plt.scatter(n, variance, c="green", s=5, label='variance')
    plt.legend()
    plt.show()
    plt.hist(bias, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    plt.hist(variance, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    # print("bias:\n", bias)
    # print("variance:\n", variance)


def function_2():
    print("\nfixed function2:\n")
    for i in range(100):
        gaussian_blur = np.random.normal(0, 0.1, 100)
        y[i] = x ** 2 + gaussian_blur
        y_hat[i] = 1
    y_average = np.average(y_hat, axis=1)
    for i in range(100):
        bias[i] = np.dot((y_hat[i] - y[i]), (y_hat[i] - y[i]))
        variance[i] = np.dot((y_hat[i] - y_average), (y_hat[i] - y_average))
    n = np.linspace(1, 100, 100)
    plt.scatter(n, bias, c="red", marker='x', s=5, label='bias')
    plt.scatter(n, variance, c="green", s=5, label='variance')
    plt.legend()
    plt.show()
    plt.hist(bias, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    plt.hist(variance, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    # print("bias:\n", bias)
    # print("variance:\n", variance)


def function_3():
    print("\nLinear Regression:\n")
    for i in range(100):
        gaussian_blur = np.random.normal(0, 0.1, 100)
        y[i] = x ** 2 + gaussian_blur
        model_3 = LinearRegression().fit(np.reshape(x, (-1, 1)), y[i])
        y_hat[i] = model_3.predict(np.reshape(x, (-1, 1)))
    y_average = np.average(y_hat, axis=1)
    for i in range(100):
        bias[i] = np.dot((y_hat[i] - y[i]), (y_hat[i] - y[i]))
        variance[i] = np.dot((y_hat[i] - y_average), (y_hat[i] - y_average))
    n = np.linspace(1, 100, 100)
    plt.scatter(n, bias, c="red", marker='x', s=5, label='bias')
    plt.scatter(n, variance, c="green", s=5, label='variance')
    plt.legend()
    plt.show()
    plt.hist(bias, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    plt.hist(variance, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    # print("bias:\n", bias)
    # print("variance:\n", variance)


def function_4():
    print("\nPolynomial Regression:\n")
    poly = PolynomialFeatures(degree=3)
    poly.fit(np.reshape(x, (-1, 1)))
    x_2 = poly.transform(np.reshape(x, (-1, 1)))
    for i in range(100):
        gaussian_blur = np.random.normal(0, 0.1, 100)
        y[i] = x ** 2 + gaussian_blur
        model_4 = LinearRegression().fit(x_2, y[i])
        y_hat[i] = model_4.predict(x_2)
    y_average = np.average(y_hat, axis=1)
    for i in range(100):
        bias[i] = np.dot((y_hat[i] - y[i]), (y_hat[i] - y[i]))
        variance[i] = np.dot((y_hat[i] - y_average), (y_hat[i] - y_average))
    n = np.linspace(1, 100, 100)
    plt.scatter(n, bias, c="red", marker='x', s=5, label='bias')
    plt.scatter(n, variance, c="green", s=5, label='variance')
    plt.legend()
    plt.show()
    plt.hist(bias, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    plt.hist(variance, bins=20, color='steelblue', edgecolor='black')
    plt.show()
    # print("bias:\n", bias)
    # print("variance:\n", variance)


def main():
    # gaussian_blur = np.random.normal(0, 0.1, 100)
    # data = x ** 2 + gaussian_blur
    # plt.scatter(x, data, c="red", marker='x', s=5)
    # plt.show()
    # function_1()
    # function_2()
    # function_3()
    function_4()


if __name__ == '__main__':
    main()
# 模式识别实验一

### 数据

![](C:\Users\Blank%20Wang\AppData\Roaming\marktext\images\2022-03-19-15-17-27-image.png)

## Problem 1

### (a) 编写程序，对实验数据中的第一类的三个特征进行计算，求解最大似然估计$\mu$和$\sigma$

#### Code

```python
def maximum_likelihood_mean(mean_value,fixed_var,feature_index):

    global data_1,data_2,data_3

    data =data_1[feature_index]
    result_buffer= np.zeros_like(mean_value)

    for index, data_value in enumerate(data):
        probability = Gaussian_density_probability(mean_value,fixed_var,data_value)
        result_buffer = np.vstack((result_buffer,probability))
    result_buffer = result_buffer[1:,:]

    log_likelihood= log_likelihood_function(result_buffer)

    # plt.plot(mean_value,log_likelihood,'bo')
    # plt.xlabel('mean value')
    # plt.ylabel('log likelihood')
    # plt.title('Log likelihood value for mean space')
    # plt.show()
    MLE_mean = mean_value[np.argmax(log_likelihood)]

    return MLE_mean

def maximum_likelihood_variance(MLE_mean,var_value,feature_index):

    global data_1,data_2,data_3

    data = data_1[feature_index]
    mean = MLE_mean
    result_buffer = np.zeros_like(var_value)

    for index, data_value in enumerate(data):
        probability = Gaussian_density_probability(mean,var_value,data_value)
        result_buffer = np.vstack((result_buffer, probability))

    result_buffer = result_buffer[1:, :]
    #print(result_buffer)
    log_likelihood = log_likelihood_function(result_buffer)
    MLE_variance = var_value[np.argmax(log_likelihood)]

    return MLE_variance
```

#### Result

```
|===========================================================
Problem (a)
Feature index :  1
Mean :  -0.07999999999999996
Variance :  0.904
Feature index :  2
Mean :  -0.6
Variance :  4.2
Feature index :  3
Mean :  -0.92
Variance :  4.5440000000000005
===========================================================
```

### (b) 处理二维数据，处理第一类中的任意两个特征的组合

#### Code

```python
def maximum_likelihood_mean_2d(mean_value_1,mean_value_2,cov_mat,data):

    mean_value = Mean_matrix_2d(mean_value_1,mean_value_2)
    result_buffer = np.zeros((mean_value.shape[0]))
    for index,mean in enumerate(mean_value):
        probability = multi_dimension_Gaussian_density_probability(mean.reshape((2,1)),cov_mat,data,2)
        prob_diag = np.diag(probability)
        result_buffer[index] = log_likelihood_function(prob_diag)

    # x= mean_value[:,0]
    # y= mean_value[:,1]
    # fig =plt.figure(figsize=(10,5))
    # ax= fig.add_subplot(111,projection='3d')
    # ax.plot(x,y,result_buffer)
    # ax.set_xlabel('mean space 1')
    # ax.set_ylabel('mean space 2')
    # ax.set_zlabel('log likelihood')
    # plt.show()
    MLE_mean = mean_value[np.argmax(result_buffer)]

    return MLE_mean

def maximum_likelihood_var_2d(mean_value,data):

    global var_value1,var_value2,var_value3
    # mean_value= mean_value.reshape((mean_value.size,1))
    # temp= np.zeros_like(mean_value)
    # temp1= np.hstack((mean_value,temp))
    # # print(temp1)
    # temp = np.zeros_like(mean_value)
    Cov_mat = Var_matrix_2d(var_value1,var_value2,var_value3)
    result_buffer = np.zeros((var_value1.shape[0]*var_value2.shape[0]*var_value3.shape[0]))
    for index,cov in enumerate(Cov_mat):
        probability = multi_dimension_Gaussian_density_probability(mean_value.reshape((2,1)),cov,data,2)
        prob_diag = np.diag(probability)
        #print(prob_diag)
        result_buffer[index] = log_likelihood_function(prob_diag)
        # if log_likelihood_function(prob_diag)>0:
            # print("$$$$$$$$$$$$$")
            # print(probability)
            # print(prob_diag)
    # print(result_buffer)
    #print(result_buffer)
    #print(np.argmax(result_buffer))
    MLE_var = Cov_mat[np.argmax(result_buffer)]

    return MLE_var
```

#### Result

```
===========================================================
Problem (b)
MLE mean for feature 1,2 :  [-0.08 -0.6 ]
MLE mean for feature 1,3 :  [-0.08 -0.92]
MLE mean for feature 2,3 :  [-0.6  -0.92]
MLE Covariance matrix for feature 1,2 :
[[0.90666667 0.56666667]
 [0.56666667 4.2       ]]
MLE Covariance matrix for feature 1,3 :
[[0.90666667 0.394     ]
 [0.394      4.54      ]]
MLE Covariance matrix for feature 2,3 :
[[4.2        0.73333333]
 [0.73333333 4.54333333]]
===========================================================
```

### (c) 处理三维数据，处理第一类中的三个特征的组合

#### Code

```python
def maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data):

    mean_value = Mean_matrix_2d(mean_value1,mean_value2)
    result_buffer = np.zeros((mean_value3.shape[0],4))
    #Cov_mat = Covariance_matrix_3d(data_test1, data_test2, data_test3)
    Cov_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
    for index, mean3 in enumerate(mean_value3):
        temp = np.ones(mean_value1.shape[0]*mean_value2.shape[0])*(mean3)
        temp = temp.reshape((mean_value1.shape[0]*mean_value2.shape[0],1))
        mean_value_stack = np.hstack((mean_value,temp))
        MLE_mean = MLE_mean_stacker(mean_value_stack,Cov_mat,data)
        result_buffer[index] =MLE_mean

    temp = np.argmax(result_buffer,axis=0)
    #print(temp,'d')
    #print(temp)
    MLE_mean= result_buffer[temp[3]]
    #print(MLE_mean)
    MLE_mean= MLE_mean[0:3]
    return MLE_mean

def maximum_likelihood_var_3d(MLE_mean,data):

    global var_value_31,var_value_32,var_value_33,var_value_34,var_value_35,var_value_36

    mean_value = MLE_mean
    result_buffer = np.zeros((var_value_31.shape[0]*var_value_32.shape[0]*var_value_33.shape[0]*var_value_34.shape[0]*var_value_35.shape[0]*var_value_36.shape[0]))
    Cov_mat = Var_matrix_3d(var_value_31,var_value_32,var_value_33,var_value_34,var_value_35,var_value_36)
    for index, cov in enumerate(Cov_mat):
        probability = multi_dimension_Gaussian_density_probability(mean_value.reshape((3,1)),cov,data,3)
        prob_diag = np.diag(probability)
        #print(prob_diag)
        result_buffer[index] = log_likelihood_function(prob_diag)

    MLE_var = Cov_mat[np.argmax(result_buffer)]

    return MLE_var
```

#### Result

```
===========================================================
Problem (c)
Mean :  [-0.08 -0.6  -0.92]
Covariance matrix:
[[0.906  0.568  0.39  ]
 [0.568  4.2008 0.73  ]
 [0.39   0.73   4.54  ]]
Sample covariance matrix
[[0.90617729 0.56778177 0.3940801 ]
 [0.56778177 4.20071481 0.7337023 ]
 [0.3940801  0.7337023  4.541949  ]]
===========================================================
```

### (d) 在这三维高斯模型可分离的条件下，编写程序估计类别二中的均值和协方差矩阵中的三个参数

#### Result

```
===========================================================
Problem (d)
Mean :  [-0.12  0.44  0.  ]
Covariance matrix:
[[0.054  0.     0.    ]
 [0.     0.046  0.    ]
 [0.     0.     0.0074]]
Sample Covariance matrix:
[[ 0.05392584 -0.01465126 -0.00517993]
 [-0.01465126  0.04597009  0.00850987]
 [-0.00517993  0.00850987  0.00726551]]
===========================================================
```

### (e) 比较前4种方式计算出来的均值的异同，并加以解释

#### Result

```
===========================================================
Problem (e)
dataset 1 mean vector:  [-0.0709 -0.6047 -0.911 ]
MLE mean of dataset 1:  [-0.08 -0.6  -0.92]
Difference:  [0.0091 0.0047 0.009 ]
dataset 2 mean vector:  [-0.1126   0.4299   0.00372]
MLE mean of dataset 2:  [-0.12  0.44  0.  ]
Difference:  [0.0074  0.0101  0.00372]
dataset 3 mean vector:  [0.2747 0.3001 0.6786]
MLE mean of dataset 3:  [0.28 0.32 0.68]
Difference:  [0.0053 0.0199 0.0014]
===========================================================
```

相比一维最大似然和二维最大似然中的均值估计，三维高斯可分离模型的精度要差一些。

#### Explanation

相比一维最大似然和二维最大似然，三维高斯可分离模型的假设空间变大，并且假设空间的增大并非线性的。所以导致同样的数据量，三维高斯可分离模型的精度差很大。

### (f) 比较前 4 种方式计算出来的方差的异同，并加以解释

#### Result

```
===========================================================
Problem (f)
dataset 1 Cov matrix:
[[0.90617729 0.56778177 0.3940801 ]
 [0.56778177 4.20071481 0.7337023 ]
 [0.3940801  0.7337023  4.541949  ]]
MLE Cov matrix of dataset 1:
[[0.906  0.568  0.39  ]
 [0.568  4.2008 0.73  ]
 [0.39   0.73   4.54  ]]
Difference:
[[1.7729e-04 2.1823e-04 4.0801e-03]
 [2.1823e-04 8.5190e-05 3.7023e-03]
 [4.0801e-03 3.7023e-03 1.9490e-03]]
-----------------------
dataset 2 Cov matrix:
[[ 0.05392584 -0.01465126 -0.00517993]
 [-0.01465126  0.04597009  0.00850987]
 [-0.00517993  0.00850987  0.00726551]]
MLE Cov matrix of dataset 2:
[[ 0.054  -0.014  -0.0052]
 [-0.014   0.046   0.0084]
 [-0.0052  0.0084  0.0074]]
Difference:
[[7.416000e-05 6.512600e-04 2.007200e-05]
 [6.512600e-04 2.991000e-05 1.098720e-04]
 [2.007200e-05 1.098720e-04 1.344944e-04]]
-----------------------
dataset 3 Cov matrix:
[[ 0.30186081  0.40474153 -0.18042342]
 [ 0.40474153  0.64496409 -0.20130386]
 [-0.18042342 -0.20130386  1.26214164]]
MLE Cov matrix of dataset 3:
[[ 0.32  0.42 -0.2 ]
 [ 0.42  0.64 -0.23]
 [-0.2  -0.23  1.28]]
Difference:
[[0.01813919 0.01525847 0.01957658]
 [0.01525847 0.00496409 0.02869614]
 [0.01957658 0.02869614 0.01785836]]
===========================================================
```

#### Explanation

相比一维最大似然和二维最大似然，三维高斯可分离模型的假设空间对协方差的估计基于可分离假设，规定了协方差矩阵以对角形式存在。而直接的最大似然解法的精度则极为依赖对于参数的搜索精度，极大增加了计算量类似的方法在高维数据中几乎是不可完成的，而分离假设为问题提供了相对简单有效的解法，实际的分类问题中往往对参数的具体假设并不影响最终的分类结果。

## Problem 2

### (a) 编写用 FISHER 线性判别方法，对三维数据求最优方向 w 的通用程序

Code

```python
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
```

### (b) 对表格中的类别 W2 和 W3，计算最优方向 w

### (c) 画出表示最优方向 w 的直线，并且标记出投影后的点在直线上的位置

### (d) 在这个子空间中，对每种分布用一维高斯函数拟合，并且求分类决策面

### (e) (b)中得到的分类器的训练误差是什么？

### (f) 为了比较，使用非最优方向 w=(1.0,2.0,-1.5)重复(d)(e)两个步骤。在这个非最优子空间 中，训练误差是什么。

#### Result

```
===========================================================
Problem (a) (b) (c) : get a projection
w: [-0.38324607  0.21374852 -0.07673687]
new_data_2: [ 0.27044299  0.17958786 -0.13119155  0.16992952  0.23198888  0.07466096
  0.13559003  0.22465461  0.12955569  0.06236637]
new_data_3: [ 0.02497771 -0.11640675  0.05643558 -0.22163101 -0.2703812   0.03842562
 -0.15793296 -0.10087196 -0.1297735  -0.05489557]
===========================================================
Problem (d) (e) (f): get a projection
w_sample: [ 1.   2.  -1.5]
sample_data_2: [0.6265 0.29   0.5425 0.8935 0.539  1.4    1.359  0.77   0.9132 0.0825]
sample_data_3: [ 4.051  3.58  -1.74  -2.953 -3.67  -1.515 -0.028 -4.04   4.19   0.695]
decision_point1: 0.0204784253784852
decision_point2: -0.0624673904703719
```

![](C:\Users\Blank%20Wang\AppData\Roaming\marktext\images\2022-03-19-17-34-17-image.png)

##### Fisher LDA Projection

![](C:\Users\Blank%20Wang\AppData\Roaming\marktext\images\2022-03-19-17-34-52-image.png)

**Error Rate: 20%**

##### Casual Projection

![](C:\Users\Blank%20Wang\AppData\Roaming\marktext\images\2022-03-19-17-36-44-image.png)

**Error Rate: 25%**

# 模式识别实验二

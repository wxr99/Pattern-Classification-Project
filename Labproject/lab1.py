import numpy as np
from numpy import pi,exp,sqrt,log
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (12,8)

Gaussian_constant = 1/sqrt(2*pi)
data_11= np.array([[0.42,-0.087,0.58],
                   [-0.2,-3.3,-3.4],
                   [1.3,-0.32,1.7],
                   [0.39,0.71,0.23],
                   [-1.6,-5.3,-0.15],
                   [-0.029,0.89,-4.7],
                   [-0.23,1.9,2.2],
                   [0.27,-0.3,0.87],
                   [-1.9,0.76,-2.1],
                   [0.87,-1.0,-2.6]],dtype=np.float)

data_1= np.array([[0.42,-0.2,1.3,0.39,-1.6,-0.029,-0.23,0.27,-1.9,0.87],
                   [-0.087,-3.3,-0.32,0.71,-5.3,0.89,1.9,-0.3,0.76,-1.0],
                   [0.58,-3.4,1.7,0.23,-0.15,-4.7,2.2,-0.87,-2.1,-2.6]],dtype= np.float)

data_2= np.array([[-0.4,-0.31,0.38,-0.15,-0.35,0.17,-0.011,-0.27,-0.065,-0.12],
                  [0.58,0.27,0.055,0.53,0.47,0.69,0.55,0.61,0.49,0.054],
                  [0.089,-0.04,-0.035,0.011,0.034,0.1,-0.18,0.12,0.0012,-0.063]],dtype=np.float)

data_3= np.array([[0.83,1.1,-0.44,0.047,0.28,-0.39,0.34,-0.3,1.1,0.18],
                  [1.6,1.6,-0.41,-0.45,0.35,-0.48,-0.079,-0.22,1.2,-0.11],
                  [-0.014,0.48,0.32,1.4,3.1,0.11,0.14,2.2,-0.46,-0.49]],dtype=np.float)

mean_resolution_1 = 10
mean_resolution_2 = 10
mean_resolution_3 = 10
var_resolution_1 = 5
var_resolution_2 = 5
var_resolution_3 = 5
var_resolution_4 = 5
var_resolution_5 = 5
var_resolution_6 = 5

var_value_34 = np.linspace(4.5, 4.6, num=var_resolution_4, endpoint=False)  # [3,3]
var_value_35 = np.linspace(4.2, 4.201, num=var_resolution_5, endpoint=False)  # [2,2]
var_value_36 = np.linspace(0.9, 0.91, num=var_resolution_6, endpoint=False)  # [1,1]

var_value_31 = np.linspace(0.35, 0.45, num=var_resolution_1, endpoint=False)  # [1,3]
var_value_32 = np.linspace(0.73, 0.77, num=var_resolution_2, endpoint=False)  # [2,3]
var_value_33 = np.linspace(0.55, 0.58, num=var_resolution_3, endpoint=False)  # [1,2]

def Gaussian_density_probability(mean, variance, data):

    sigma = sqrt(variance)
    temp = data-mean
    temp = temp/sigma
    Gaussian_constant = 1 / sqrt(2 * pi)
    probability = (Gaussian_constant/sigma)*exp(-(1/2)*(np.power(temp,2)))

    return probability

def multi_dimension_Gaussian_density_probability(mean,cov_matrix,data,dimension):

    det = np.linalg.det(cov_matrix)
    if det < 0:
        probability = np.zeros((data.shape[1],data.shape[1]))
    elif det > 0:
        cov_inverse = np.linalg.inv(cov_matrix)
        temp = data- mean
        constant = 1/((sqrt(2*pi)**dimension)*sqrt(det))
        probability =constant*exp((-1/2)*np.dot(np.dot(temp.T,cov_inverse),temp))

    return probability

def log_likelihood_function(probability):
    log_probability = log(probability)
    log_probability_sum = np.sum(log_probability,axis=0)
    return log_probability_sum

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

def MLE_mean_stacker(mean_value,cov_mat,data):

    result_buffer = np.zeros((mean_value.shape[0]))
    for index,mean in enumerate(mean_value):
        probability = multi_dimension_Gaussian_density_probability(mean.reshape((3,1)),cov_mat,data,3)
        prob_diag = np.diag(probability)
        result_buffer[index] = log_likelihood_function(prob_diag)
    MLE_mean = mean_value[np.argmax(result_buffer)]
    MLE_mean = np.append(MLE_mean,[np.max(result_buffer)],axis=0)
    return MLE_mean


def Covariance(x,y):
    x_mean, y_mean = x.mean(),y.mean()
    covariance = np.sum((x-x_mean)*(y-y_mean))/x.size

    return covariance

def Covariance_matrix_2d(x,y):

    Cov_mat= np.array([[Covariance(x,x),Covariance(x,y)],
                       [Covariance(x,y),Covariance(y,y)]])
    return Cov_mat

def Covariance_matrix_3d(x,y,z):

    Cov_mat = np.array([[Covariance(x,x),Covariance(x,y),Covariance(x,z)],
                        [Covariance(x,y),Covariance(y,y),Covariance(y,z)],
                        [Covariance(x,z),Covariance(y,z),Covariance(z,z)]])
    return Cov_mat

def Mean_matrix_2d(mean_value1,mean_value2):

    mean_value1_temp = np.tile(mean_value1, mean_value2.shape[0])
    mean_value1_temp = mean_value1_temp.reshape((mean_value1_temp.shape[0],1))
    mean_value2_temp = np.repeat(mean_value2, mean_value1.shape[0])
    mean_value2_temp = mean_value2_temp.reshape((mean_value2_temp.shape[0],1))
    mean_value = np.hstack((mean_value1_temp,mean_value2_temp))
    return mean_value

def Var_matrix_2d(var_value1,var_value2,var_value3):

    var_value1_temp = np.tile(var_value1, var_value2.shape[0])
    var_value1_temp = var_value1_temp.reshape((var_value1_temp.shape[0],1))
    var_value2_temp = np.repeat(var_value2, var_value1.shape[0])
    var_value2_temp = var_value2_temp.reshape((var_value2_temp.shape[0],1))
    var_value = np.hstack((var_value1_temp,var_value2_temp))

    result_buffer = np.zeros((2,2))
    for _, cov in enumerate(var_value3):
        Cov_matrix = np.array([[0,cov],[cov,0]])
        for _, vars in enumerate(var_value):
            Diag_temp = np.diag(vars)
            result_buffer = np.vstack((result_buffer, Cov_matrix + Diag_temp))

    result_buffer= result_buffer.reshape((-1,2,2))
    Var_matrix= result_buffer[1:]
    return Var_matrix

def Var_matrix_3d(var_value1,var_value2,var_value3,var_value4,var_value5,var_value6):

    var_value1_temp = np.tile(var_value1, var_value2.shape[0])
    var_value1_temp = var_value1_temp.reshape((var_value1_temp.shape[0], 1))
    var_value2_temp = np.repeat(var_value2, var_value1.shape[0])
    var_value2_temp = var_value2_temp.reshape((var_value2_temp.shape[0], 1))
    var_value3_temp = np.concatenate((var_value1_temp, var_value2_temp),axis=1)

    var_value4_temp = np.repeat(var_value3,var_value3_temp.shape[0])
    var_value4_temp = var_value4_temp.reshape((var_value4_temp.shape[0],1))
    var_value5_temp = np.tile(var_value3_temp,(var_value3.shape[0],1))
    var_value5_temp = var_value5_temp.reshape((var_value4_temp.shape[0], -1))
    var_value6_temp= np.concatenate((var_value4_temp,var_value5_temp),axis=1)

    var_value7_temp = np.repeat(var_value4,var_value6_temp.shape[0])
    var_value7_temp = var_value7_temp.reshape((var_value7_temp.shape[0],1))
    var_value8_temp = np.tile(var_value6_temp,(var_value4.shape[0],1))
    var_value8_temp = var_value8_temp.reshape((var_value7_temp.shape[0], -1))
    var_value9_temp= np.concatenate((var_value7_temp,var_value8_temp),axis=1)

    var_value10_temp = np.repeat(var_value5,var_value9_temp.shape[0])
    var_value10_temp = var_value10_temp.reshape((var_value10_temp.shape[0],1))
    var_value11_temp = np.tile(var_value9_temp,(var_value5.shape[0],1))
    var_value11_temp = var_value11_temp.reshape((var_value10_temp.shape[0], -1))
    var_value12_temp= np.concatenate((var_value10_temp,var_value11_temp),axis=1)

    var_value13_temp = np.repeat(var_value6,var_value12_temp.shape[0])
    var_value13_temp = var_value13_temp.reshape((var_value13_temp.shape[0],1))
    var_value14_temp = np.tile(var_value12_temp,(var_value6.shape[0],1))
    var_value14_temp = var_value14_temp.reshape((var_value13_temp.shape[0], -1))
    var_value15_temp= np.concatenate((var_value13_temp,var_value14_temp),axis=1)

    #print(var_value15_temp.shape)
    result_buffer = np.zeros((1,3,3))
    for _, cov_source in enumerate(var_value15_temp):
        Cov = np.diag(cov_source[0:3])
        temp = np.array([[[0,cov_source[3],cov_source[4]],
                         [cov_source[3],0,cov_source[5]],
                         [cov_source[4],cov_source[5],0]]])
        result_buffer = np.vstack((result_buffer, Cov + temp))

    result_buffer = result_buffer[1:]
    return  result_buffer


def main():
    # Data
    global data_1,data_2,data_3
    # for mean grid_search resolution
    global mean_resolution_1,mean_resolution_2,mean_resolution_3
    # for variance grid_search resolution
    global var_resolution_4,var_resolution_5,var_resolution_6
    # for covariance grid_serach resolution
    global var_resolution_1, var_resolution_2, var_resolution_3

    # Grid mean value array for 1,2,3 dimension
    global mean_value1, mean_value2, mean_value3
    # Grid variance value array for 1,2 dimension
    global var_value1,var_value2,var_value3
    # Grid variance value array for 1,2,3 dimension
    global var_value_31,var_value_32,var_value_33,var_value_34,var_value_35,var_value_36

    mean_resolution_1 = 50
    mean_resolution_2 = 50
    mean_resolution_3 = 50
    var_resolution_1 = 10
    var_resolution_2 = 10
    var_resolution_3 = 10

    mean_value1 = np.linspace(-1,1,num=mean_resolution_1,endpoint=False)
    mean_value2 = np.linspace(-1,1,num=mean_resolution_2,endpoint=False)
    mean_value3 = np.linspace(-1,1,num=mean_resolution_3,endpoint=False)

    var_value1 = np.linspace(4, 4.5, num=var_resolution_1, endpoint=False)
    var_value2 = np.linspace(4, 4.5, num=var_resolution_2, endpoint=False)
    var_value3 = np.linspace(0.3, 1, num=var_resolution_3, endpoint=False)


#################################################################################
#for testing maximum_likelihood for each mean, variance

    mean_value = np.vstack((mean_value1,mean_value2,mean_value3))
    var_value = np.vstack((var_value1,var_value2,var_value3))

##################################################################################
#Data_setting for (a),(b),(c)

    data = data_1
###################################################################################
#(a)
    print("===========================================================")
    print("Problem (a)")

    mean_resolution_1 = 100
    mean_value1 = np.linspace(-1,1,num=mean_resolution_1,endpoint=False)

    var_resolution_1=50
    var_resolution_2=50
    var_resolution_3=50

    var_value1 = np.linspace(0.8, 1.2, num=var_resolution_1, endpoint=False)
    var_value2 = np.linspace(4, 4.5, num=var_resolution_2, endpoint=False)
    var_value3 = np.linspace(4.5, 4.7, num=var_resolution_3, endpoint=False)
    var_value = np.vstack((var_value1,var_value2,var_value3))

    for index in range(data.shape[0]):
        MLE_mean =maximum_likelihood_mean(mean_value1,1,index)
        MLE_variance = maximum_likelihood_variance(MLE_mean,var_value[index],index)
        print("Feature index : ",index+1)
        print("Mean : ",MLE_mean)
        print("Variance : ", MLE_variance)
    print("===========================================================")

###################################################################################
#(b)
    print("===========================================================")
    print("Problem (b)")
    Cov_mat1 = Covariance_matrix_2d(data[0],data[1])
    Cov_mat2 = Covariance_matrix_2d(data[0],data[2])
    Cov_mat3 = Covariance_matrix_2d(data[1],data[2])

    MLE_mean1 = maximum_likelihood_mean_2d(mean_value1,mean_value2,Cov_mat1,data[0:2])
    MLE_mean2 = maximum_likelihood_mean_2d(mean_value1,mean_value3,Cov_mat2,np.vstack((data[0],data[2])))
    MLE_mean3 = maximum_likelihood_mean_2d(mean_value2,mean_value3,Cov_mat3,data[1:])
    print("MLE mean for feature 1,2 : ",MLE_mean1)
    print("MLE mean for feature 1,3 : ",MLE_mean2)
    print("MLE mean for feature 2,3 : ",MLE_mean3)

    var_resolution_1 = 30
    var_resolution_2 = 30
    var_resolution_3 = 30

    var_value1 = np.linspace(0.8, 1.2, num=var_resolution_1, endpoint=False)
    var_value2 = np.linspace(4, 4.5, num=var_resolution_2, endpoint=False)
    var_value3 = np.linspace(0.5, 0.7, num=var_resolution_3, endpoint=False)
    MLE_variance1 = maximum_likelihood_var_2d(MLE_mean1,data[0:2])
    print("MLE Covariance matrix for feature 1,2 : ")
    print(MLE_variance1)

    var_value1 = np.linspace(0.8, 1.2, num=var_resolution_1, endpoint=False)
    var_value2 = np.linspace(4.5, 4.7, num=var_resolution_2, endpoint=False)
    var_value3 = np.linspace(0.37, 0.41, num=var_resolution_3, endpoint=False)
    MLE_variance2 = maximum_likelihood_var_2d(MLE_mean2,np.vstack((data[0],data[2])))
    print("MLE Covariance matrix for feature 1,3 : ")
    print(MLE_variance2)

    var_value1 = np.linspace(4.1, 4.3, num=var_resolution_1, endpoint=False)
    var_value2 = np.linspace(4.5, 4.6, num=var_resolution_2, endpoint=False)
    var_value3 = np.linspace(0.7, 0.75, num=var_resolution_3, endpoint=False)
    MLE_variance3 = maximum_likelihood_var_2d(MLE_mean3,data[1:])
    print("MLE Covariance matrix for feature 2,3 : ")
    print(MLE_variance3)

    print("===========================================================")

###################################################################################
#(c)

    print("===========================================================")
    print("Problem (c)")
    MLE_mean_3d = maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data)
    print("Mean : ",MLE_mean_3d)
    print("Covariance matrix:")
    print(maximum_likelihood_var_3d(MLE_mean_3d,data))
    print("Sample covariance matrix")
    print(Covariance_matrix_3d(data[0],data[1],data[2]))
    print("===========================================================")

###################################################################################
#(d)
    print("===========================================================")
    print("Problem (d)")
    #Converting data
    data = data_2

    #Converting mean
    mean_resolution_1 = 50
    mean_resolution_2 = 50
    mean_resolution_3 = 50

    mean_value1 = np.linspace(-1,1,num=mean_resolution_1,endpoint=False)
    mean_value2 = np.linspace(-1,1,num=mean_resolution_2,endpoint=False)
    mean_value3 = np.linspace(-1,1,num=mean_resolution_3,endpoint=False)

    #Estimating mean
    MLE_mean_3d = maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data)
    print("Mean : ",MLE_mean_3d)

    #Converting Variance grid
    var_resolution_1 = 5
    var_resolution_2 = 5
    var_resolution_3 = 5
    var_resolution_4 = 5
    var_resolution_5 = 5
    var_resolution_6 = 5

    var_value_34 = np.linspace(0.005, 0.008, num=var_resolution_4, endpoint=False)  # [3,3]
    var_value_35 = np.linspace(0.04, 0.05, num=var_resolution_5, endpoint=False)  # [2,2]
    var_value_36 = np.linspace(0.05, 0.06, num=var_resolution_6, endpoint=False)  # [1,1]

    var_value_31 = np.linspace(0., 0., num=var_resolution_1, endpoint=False)  # [1,3]
    var_value_32 = np.linspace(0., 0., num=var_resolution_2, endpoint=False)  # [2,3]
    var_value_33 = np.linspace(0., 0., num=var_resolution_3, endpoint=False)  # [1,2]

    MLE_variance_3d = maximum_likelihood_var_3d(MLE_mean_3d,data)
    print("Covariance matrix:")
    print(MLE_variance_3d)
    print("Sample Covariance matrix:")
    print(Covariance_matrix_3d(data[0],data[1],data[2]))
    print("===========================================================")

####################################################################################
#(e)
    print("===========================================================")
    print("Problem (e)")

    # Mean of each dataset
    mean_1 = np.mean(data_1,axis=1)
    mean_2 = np.mean(data_2,axis=1)
    mean_3 = np.mean(data_3,axis=1)
    MLE_mean_1 = maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data_1)
    MLE_mean_2 = maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data_2)
    MLE_mean_3 = maximum_likelihood_mean_3d(mean_value1,mean_value2,mean_value3,data_3)
    print("dataset 1 mean vector: ",mean_1)
    print("MLE mean of dataset 1: ",MLE_mean_1)
    print("Difference: ",np.abs(mean_1 - MLE_mean_1))
    print("dataset 2 mean vector: ",mean_2)
    print("MLE mean of dataset 2: ",MLE_mean_2)
    print("Difference: ",np.abs(mean_2 - MLE_mean_2))
    print("dataset 3 mean vector: ",mean_3)
    print("MLE mean of dataset 3: ",MLE_mean_3)
    print("Difference: ",np.abs(mean_3 - MLE_mean_3))

    print("===========================================================")

####################################################################################
#(f)
    print("===========================================================")
    print("Problem (f)")

    #Covariance matrix of each dataset

    Cov_1 = Covariance_matrix_3d(data_1[0],data_1[1],data_1[2])
    Cov_2 = Covariance_matrix_3d(data_2[0],data_2[1],data_2[2])
    Cov_3 = Covariance_matrix_3d(data_3[0],data_3[1],data_3[2])
    # Changing grid range for data 1
    var_value_34 = np.linspace(4.5, 4.6, num=var_resolution_4, endpoint=False)  # [3,3]
    var_value_35 = np.linspace(4.2, 4.201, num=var_resolution_5, endpoint=False)  # [2,2]
    var_value_36 = np.linspace(0.9, 0.91, num=var_resolution_6, endpoint=False)  # [1,1]
    var_value_31 = np.linspace(0.35, 0.45, num=var_resolution_1, endpoint=False)  # [1,3]
    var_value_32 = np.linspace(0.73, 0.77, num=var_resolution_2, endpoint=False)  # [2,3]
    var_value_33 = np.linspace(0.55, 0.58, num=var_resolution_3, endpoint=False)  # [1,2]
    MLE_Cov_1 = maximum_likelihood_var_3d(MLE_mean_1,data_1)

    # Changing grid range for data 2
    var_value_34 = np.linspace(0.005, 0.008, num=var_resolution_4, endpoint=False)  # [3,3]
    var_value_35 = np.linspace(0.04, 0.05, num=var_resolution_5, endpoint=False)  # [2,2]
    var_value_36 = np.linspace(0.05, 0.06, num=var_resolution_6, endpoint=False)  # [1,1]
    var_value_31 = np.linspace(-0.01, -0.004, num=var_resolution_1, endpoint=False)  # [1,3]
    var_value_32 = np.linspace(0.006, 0.01, num=var_resolution_2, endpoint=False)  # [2,3]
    var_value_33 = np.linspace(-0.02, -0.01, num=var_resolution_3, endpoint=False)  # [1,2]
    MLE_Cov_2 = maximum_likelihood_var_3d(MLE_mean_2,data_2)

    # Changing grid range for data 3
    var_value_34 = np.linspace(1.2, 1.3, num=var_resolution_4, endpoint=False)  # [3,3]
    var_value_35 = np.linspace(0.4, 0.8, num=var_resolution_5, endpoint=False)  # [2,2]
    var_value_36 = np.linspace(0.2, 0.5, num=var_resolution_6, endpoint=False)  # [1,1]
    var_value_31 = np.linspace(-0.2, -0.1, num=var_resolution_1, endpoint=False)  # [1,3]
    var_value_32 = np.linspace(-0.25, -0.15, num=var_resolution_2, endpoint=False)  # [2,3]
    var_value_33 = np.linspace(0.3, 0.5, num=var_resolution_3, endpoint=False)  # [1,2]
    MLE_Cov_3 = maximum_likelihood_var_3d(MLE_mean_3,data_3)

    print("dataset 1 Cov matrix: ")
    print(Cov_1)
    print("MLE Cov matrix of dataset 1: ")
    print(MLE_Cov_1)
    print("Difference: ")
    print(np.abs(Cov_1 - MLE_Cov_1))
    print("-----------------------")
    print("dataset 2 Cov matrix: ")
    print(Cov_2)
    print("MLE Cov matrix of dataset 2: ")
    print(MLE_Cov_2)
    print("Difference: ")
    print(np.abs(Cov_2 - MLE_Cov_2))
    print("-----------------------")
    print("dataset 3 Cov matrix: ")
    print(Cov_3)
    print("MLE Cov matrix of dataset 3: ")
    print(MLE_Cov_3)
    print("Difference: ")
    print(np.abs(Cov_3 - MLE_Cov_3))

    print("===========================================================")


if __name__ == '__main__':

    main()
#!/usr/bin/env python
# coding: utf-8

# In[1]:



import random
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
#from matplotlib import pyplot as plt

np.random.seed(42)

##### WE ARE IMPLEMENTING SCALER INSIDE get_features FUNCTION
class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        raise NotImplementedError
    def __call__(self,features, is_train=False):
        raise NotImplementedError

def get_features(csv_path,is_train=False,scaler=None):
    #import data frame into pandas from csv
    df=pd.read_csv(csv_path)

    #add extra column for 1 in dataframe
    df.insert(0,'new',1)
    #dataframe to numpy array calculator
    x=df.to_numpy()
    x=np.vstack(x)

    
    #delete entry number of data that was in earlier csv
    x=np.delete(x,1,1)
    
    #deleting MODIS column
    x=np.delete(x,9,1)
    #deleting 6.0NRT column
    x=np.delete(x,10,1)
    
    
    #declaring numpy array for w
    p=x.shape[0]
   
    count=0

    for i in range(0,p):
        if x[i][8]=='Terra':
               x[i][8]=1
        if x[i][8] == 'Aqua':
               x[i][8]=1.1
        if x[i][12]=='D':
               x[i][12]=1
        if x[i][12]=='N':  
               x[i][12]=1.01
        x[i][6] = 1

    
    x=np.delete(x,11,1)
    
    for i in range(0,p):
        x[i][0]=1
    
    l1=len(x[0])
    min1=np.min(x[:,1])
    max1=np.max(x[:,1])
    p=x.shape[0]
    for i in range(0,p):
        if(max1-min1):
            x[i][1]=(x[i][1]-min1)/(max1-min1)
   
    min=np.min(x[:,2])
    max=np.max(x[:,2])
    p=x.shape[0]
    for i in range(0,p):
        if(max-min):
            x[i][2]=(x[i][2]-min)/(max-min)
    
    min3=np.min(x[:,3])
    max3=np.max(x[:,3])
    p=x.shape[0]
    for i in range(0,p):
        if max3-min3:
            x[i][3]=(x[i][3]-min3)/(max3-min3)
    
    min6=np.min(x[:,6])
    max6=np.max(x[:,6])
    p=x.shape[0]
    for i in range(0,p):
        if max6-min6:
            x[i][6]=(x[i][6]-min6)/(max6-min6)
    
    min7=np.min(x[:,7])
    max7=np.max(x[:,7])
    p=x.shape[0]
    for i in range(0,p):
        if max7-min7:
            x[i][7]=(x[i][7]-min7)/(max7-min7)
    
    min10=np.min(x[:,10])
    max10=np.max(x[:,10])
    p=x.shape[0]
    for i in range(0,p):
        if max10-min10:
            x[i][10]=(x[i][10]-min10)/(max10-min10)
  
    min9=np.min(x[:,9])
    max9=np.max(x[:,9])
    p=x.shape[0]
    for i in range(0,p):
        if max9-min9:
            x[i][9]=(x[i][9]-min9)/(max9-min9)
      
    return x

    raise NotImplementedError



def basis1(x):
    p=x.shape[0]
    for i in range(0,p-1):
        x[i][3]=x[i][3]**2+x[i][3]+2
        x[i][4]=(2*x[i][4]**2)+x[i][4]+2
        x[i][5]=(x[i][5]**2)+x[i][5]+x[i][5]+3
        if x[i][7]>200 and x[i][7]<800:
            x[i][7]=(x[i][7]**2)+x[i][7]+2
        else :
            x[i][7]=x[i][7]/4
        x[i][8]=x[i][8]*x[i][8]+2
        x[i][9]=x[i][9]**2
    return x
 
def basis2(x):
    p=x.shape[0]
    for i in range(0,p-1):
        x[i][3]=x[i][3]**2+(1)/(1+pow(2.303,-x[i][3]))
        x[i][4]=x[i][4]**2+(2)/(1+pow(2.303,-x[i][4]))
        x[i][5]=x[i][5]**2+(3)/(1+pow(2.303,-x[i][5]))
        x[i][6]=x[i][6]**2+(2)/(1+pow(2.303,-x[i][6]))
        x[i][7]=x[i][7]**2+(2)/(1+pow(2.303,-x[i][7]))
        x[i][8]=x[i][8]**2+(3)/(1+pow(2.303,-x[i][8]))
        x[i][9]=x[i][9]**2+(1)/(1+pow(2.303,-x[i][9]))
    return x
 
    
def get_targets(csv_path):
    
    df=pd.read_csv(csv_path)
    x=df.to_numpy()
    y=np.zeros(len(x),dtype='f')
    for i in range(0,x.shape[0]):
        y[i]=x[i][13]
    return y

    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    raise NotImplementedError
     

def analytical_solution(feature_matrix, targets, C=0.0):
        C=0.002
        Feature_Transpose=np.transpose(feature_matrix)
        Feature_Square=np.dot(Feature_Transpose,feature_matrix)
        
        Feature_Square1=Feature_Square.astype('float64')
        b = np.identity(Feature_Square1.shape[0], dtype = float)
        b1=b*C
        c=np.add(Feature_Square1,b1)
        c=c.astype('float64')
        c1=np.linalg.inv(c)
        
        c2=np.dot(c1,Feature_Transpose)
        
        c3=np.dot(c2,targets)
        return c3

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''
    l1=len(feature_matrix)
    pred=np.zeros(l1, dtype = 'f')
    for i in range(l1):
        sum=0
        for j in range(len(weights)):
            sum+=(weights[j]*feature_matrix[i][j])
        pred[i]=sum
    return pred
        
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

    raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''
    mse=0
    n=len(targets)
    for i in range(n):
        sum=0
        for j in range(len(weights)):
            sum+=round((weights[j]*feature_matrix[i][j]),10)
            
        diff=sum-targets[i]
        
        mse+=round((diff)*(diff),10)/len(feature_matrix)
    return mse


    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    l2=0
    for i in weights:
        l2+=(i*i)
    return round(l2,10)


    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''
    total_loss=round(mse_loss(feature_matrix, weights, targets)+(C*l2_regularizer(weights)),10)

    return total_loss
    

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''
    grad = np.zeros(len(weights),dtype='f')
    
    for i in range(len(weights)):
        instance_sum=0.0
        for j in range(len(targets)):
            wTx=0.0
            for k in range(len(weights)):
                    wTx+=round((weights[k]*feature_matrix[j][k]),10)
            diff=(wTx-targets[j])
            diff*=feature_matrix[j][i]
            instance_sum+=round(diff,10)
        
        instance_sum/=len(feature_matrix)
        l2=C*weights[i]  # C represents 2*C
        grad[i]=round(instance_sum,10)+l2
        
    return grad
        

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    raise NotImplementedError






def sample_random_batch(feature_matrix, targets, batch_size):
      
        number_of_rows = feature_matrix.shape[0]
 
        #produce random index among whole row
        random_indices = np.random.choice(number_of_rows, batch_size, replace=False)

        #store those rows in a variable rows
        sampled_feature_matrix = feature_matrix[random_indices, :]
        
        # sampled_targets[] contains value of y0,y1....yk as a vector of k*1
        sampled_targets=np.zeros(batch_size,dtype='f')

        for i in range(len(random_indices)):
            sampled_targets[i]=targets[random_indices[i]]
        
        #sorting the matrices sampled_feature_matrix and sampled_targets
        #sampled_feature_matrix.sort()
        #sampled_targets.sort()
        
        return (sampled_feature_matrix,sampled_targets)
    
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''
    w=np.zeros(n, dtype = 'f')
    return w
    '''
    Arguments
    n: int
    '''
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''
    
    for i in range(len(weights)):
        weights[i]-=(lr*gradients[i])
        
    return weights

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    raise NotImplementedError

def early_stopping(lf_dev,min_error):
    # allowed to modify argument list as per your need
    b=3
    if(lf_dev - min_error > b): # here b is hyper parameter
        return True
    else:
        return False
    
    # return True or False
    raise NotImplementedError
    

def plot_trainsize_losses(Xtrain_dev,YMSE):
    plt.clf()
    plt.plot(Xtrain_dev, YMSE)
    plt.savefig("mygraph.png")
    plt.clf()
   
    #raise NotImplementedError


def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr,
                        C,
                        batch_size,
                        max_steps,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''
    x=train_feature_matrix
    dx=dev_feature_matrix
    y=train_targets
    dy=dev_targets 
    
    w=initialize_weights(len(x[0]))
    
    dev_loss = mse_loss(dx, w, dy)
    train_loss = mse_loss(x, w, y)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    
    lf_old=math.inf
    lf=0
    min_error=math.inf
    opt_w=w
    
    k=0
    while(k<max_steps):
        (new_x,new_y)=sample_random_batch(x, y, batch_size)
        while(True):
            grad=compute_gradients(new_x, w, new_y, C)
            w=update_weights(w, grad, lr)
            
            lf=round(loss_fn(new_x,w,new_y,0),10)
            lf_dev=round(loss_fn(dx,w,dy,0),10)
            if(lf_dev < min_error):
                min_error=lf_dev
                opt_w=w
            print("Train loss: ",lf)
            print("Dev loss: ",lf_dev)
            if(abs((round(lf_old,3))-(round(lf,3))) <=2 ):  #convergence criteria
                break
                  
            k+=1
            lf_old=lf
            
            if(k>=max_steps):
                break
             
            #Early Stopping
            if(early_stopping(lf_dev,min_error)==True):
                break
            
        print("Train loss: ",round(loss_fn(new_x,w,new_y,0.5),10))
        print("Dev loss: ",round(loss_fn(dx,w,dy,0.5),10))


    return opt_w

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    #scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('DATA/train.csv',True,scaler=None), get_targets('DATA/train.csv')
    dev_features, dev_targets = get_features('DATA/dev.csv',False,scaler=None), get_targets('DATA/dev.csv')
    
    ## BASIS FUNCTION
    train_features=basis1(train_features)
    dev_features=basis1(dev_features)
    
    
    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

      
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.008,
                        C=0.14,
                        batch_size=500,
                        max_steps=200000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
      
    Xtrain_dev = np.array([100,200,500,1000,2000,4000])
    YMSE = np.array([mse_loss(dev_features[0:100], gradient_descent_soln, dev_targets[0:100]), 
                     mse_loss(dev_features[0:200], gradient_descent_soln, dev_targets[0:200]), 
                     mse_loss(dev_features[0:500], gradient_descent_soln, dev_targets[0:500]), 
                     mse_loss(dev_features[0:1000], gradient_descent_soln, dev_targets[0:1000]),
                     mse_loss(dev_features[0:2000], gradient_descent_soln, dev_targets[0:2000]),
                     mse_loss(dev_features[0:4000], gradient_descent_soln, dev_targets[0:4000])])
    Xtrain_dev = Xtrain_dev.tolist()
    YMSE = YMSE.tolist()
    plot_trainsize_losses(Xtrain_dev,YMSE)
    





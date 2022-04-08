# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 21:19:10 2021

@author: ZML
"""


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#%% import data
plt.close()
name = ['time', 'Amb', 'FreeB', 'FreeBS', 'R1', 'RSl', 'FixB', 'T7', 'Nut',
        'L1', 'Med', 'Sh', 'L2', 'R2', 'Motor', 'R1B', 'LSl',
        'XPos', 'YPos', 'ZPos', 'Feed', 'XLoad', 'YLoad', 'ZLoad', 'XMotor', 'YMotor', 'ZMotor']

Xtrain = pd.read_csv('Xtrain_edit.csv', names=name,  index_col=False).iloc[:, [2,5,6,9,13]].values
Ytrain = pd.read_csv('Ytrain.csv', names=name, index_col=False).iloc[:, 12].values


#%% data processing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.67, shuffle=False)

scaler = MinMaxScaler().fit(Xtrain)
sc_Xtrain = scaler.transform(Xtrain)
sc_Ytrain = np.ones(shape=(Xtrain.shape[0],1))


#%% Draw RBF
# x=np.arange(-1,1+0.01,0.01)[:,np.newaxis]
# gamma=50
# Y=np.exp(-gamma*np.power(x,2))
# Y=np.zeros(shape=[x.shape[0],1])
# for i in range(x.shape[0]):
#     _diff=np.power(x[i,0]-0, 2)+np.power(x[i,1]-1, 2)
#     Y[i,0]=np.exp(-gamma*_diff)
    
    

# RBF_transform(15,2500,x)


# def RBF_transform(n,gamma, x):
#     range_up=x[0]
#     range_down=x[-1]
#     n=(range_up-range_down)/(n-1)
#     C = np.arange(range_down, range_up+n, n)
#     plt.figure(5)
#     Y=np.zeros(shape=x.shape)
#     for i in range(len(C)):
#         Y = np.exp(-np.square(x-C[i])*gamma)


#         plt.plot(x,Y)
#         plt.show()

# RBF_transform(3, gamma, x)

# %% Setup NeuralNetwork

from keras.layers import Layer
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma
        
        
    def build(self, input_shape):        
           
        range_down = -1
        range_up = 2
        if(self.units==1):
            self.uuuu=np.zeros([1, self.units])
        else:
            n=(range_up-range_down)/(self.units-1)
            self.uuuu = np.arange(range_down,range_up+n,n)
        print(self.uuuu)
        
        super(RBFLayer, self).build(input_shape)


    def call(self, inputs):
               
        inputs=inputs.numpy()
        
        rbf_diff=np.zeros([self.units, inputs.shape[0]])
        for i in range(inputs.shape[0]):
            for j in range(self.units):
                diff=np.power((inputs[i,:]-self.uuuu[j]),2)
                rbf_diff[j,i]=np.sum(diff)
        
        rbf_pow = -self.gamma*rbf_diff
        rbf = np.exp(rbf_pow).T        
        
        rbf_diff = tf.convert_to_tensor(rbf_diff, dtype=tf.float32)
        return rbf


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

#%% training

model = tf.keras.Sequential()
model.add(RBFLayer(6, gamma=3, input_shape=(5,)))
model.add(tf.keras.layers.Dense(1, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.Zeros()))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.3,
    decay_steps=10,
    decay_rate=0.7)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='mean_absolute_error', optimizer=opt, run_eagerly=True)


history = model.fit(sc_Xtrain, sc_Ytrain, epochs=50, verbose=1, batch_size=8, shuffle=False)






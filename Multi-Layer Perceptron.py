# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:17:38 2021

@author: Hanan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:12:14 2021

@author: Hanan
"""

# Note that this is a 4-layer perceptron (1 input, 1 output & 2-hidden layers)

# Defining Functions

import numpy as np
from decimal import Decimal

def sigmoid(x):      # sigmoid function
    e=2.718281828
    z=1/(1+(e)**(-x))
    return z

def sigmoid_diff(x):      # sigmoid differentiated function
    e=2.718281828
    z=-((e)**(-x))/((1+(e)**(-x))*(1+(e)**(-x)))
    return z

def signal(x):      # returns binary output for a function
    if x>=0:
        return 1
    else:
        return -1
    
def vector_sigmoid(x):    # vector signal of a array
    z=x.shape
    for i in range(z[0]):
        for j in range(z[1]):
            x[i][j]=sigmoid(x[i][j])
    return x

def vector_sigmoid_diff(x):    # vector sigmoid differential of a list
    z=x.shape
    for i in range(z[0]):
        for j in range(z[1]):
            x[i][j]=sigmoid_diff(x[i][j])
    return x

def vector_error(x,y):
    t=Decimal(0)
    a=y.shape[0]
    b=y.shape[2]
    for i in range(a):
        for j in range(b):
            t+=(Decimal(float(x[i][j]))-Decimal(float(y[i][j])))*(Decimal(float(x[i][j]))-Decimal(float(y[i][j])))
    return t/(2*Decimal(a))


# Main Program


y_initial=[[[0.0611,0.2860,0.7464]],[[0.5102,0.7464,0.0860]],[[0.0004,0.6916,0.5006]],[[0.9430,0.4476,0.2648]],[[0.1399,0.1610,0.2477]],[[0.6423,0.3229,0.8567]],[[0.6492, 0.0007, 0.6422]],[[0.1818, 0.5078, 0.9046]],[[0.7382, 0.2647, 0.1916]],[[0.3879, 0.1307, 0.8656]],[[0.1903, 0.6523, 0.782]],[[0.8401, 0.449, 0.2719]],[[0.0029, 0.3264, 0.2476]],[[0.7088, 0.9342, 0.2763]],[[0.1283, 0.1882, 0.7253]],[[0.8882, 0.3077, 0.8931]],[[0.2225, 0.9182, 0.782]],[[0.1957, 0.8423, 0.3085]],[[0.9991, 0.5914, 0.3933]],[[0.2299, 0.1524, 0.7353]]]   # Inputs
d_initial=[[[0.4831]],[[0.5965]],[[0.5318]],[[0.6843]],[[0.2872]],[[0.7663]],[[0.5666]],[[0.6601]],[[0.5427]],[[0.5836]],[[0.695]],[[0.679]],[[0.2956]],[[0.7742]],[[0.4662]],[[0.8093]],[[0.7581]],[[0.5826]],[[0.7938]],[[0.5012]]]
n=0.001   # Learning rate
e=Decimal(0)     # Error range

cases=int(input("Enter number of testcases >>> "))
x0=int(input("Enter number parameters >>> "))
x1=int(input("Enter number of nodes in layer 1 >>> "))
x2=int(input("Enter number of nodes in layer 2 >>> "))
x3=int(input("Enter number outputs >>> "))

y0=np.array(y_initial)
d=np.array(d_initial)

w1=np.random.rand(x1,x0)   # W(1)
w2=np.random.rand(x2,x1)   # W(2)
w3=np.random.rand(x3,x2)   # W(3)

y1=np.ones((cases,1,x1))   # Y(1)
y2=np.ones((cases,1,x2))   # Y(2)
y3=np.ones((cases,1,x3))   # Outputs

epoch=0 
vector_error_previous=Decimal(0) 
while (vector_error(d,y3)-vector_error_previous)*(vector_error(d,y3)-vector_error_previous)>e*e:     
    
    epoch+=1
    vector_error_previous=vector_error(d,y3)
    
    for case1 in range(cases):
        y1[case1]=vector_sigmoid(np.dot(y0[case1],w1.transpose()))
        y2[case1]=vector_sigmoid(np.dot(y1[case1],w2.transpose()))
        y3[case1]=vector_sigmoid(np.dot(y2[case1],w3.transpose()))
        
        delta3=d[case1]-y3[case1]*vector_sigmoid_diff(np.dot(y2[case1],w3.transpose()))
        w3=w3+(n*np.dot(delta3.transpose(),y2[case1]))
        
        delta2=-(np.dot(delta3,w3))*vector_sigmoid_diff(np.dot(y1[case1],w2.transpose()))
        w2=w2+(n*np.dot(delta2.transpose(),y1[case1]))
        
        delta1=-(np.dot(delta2,w2))*vector_sigmoid_diff(np.dot(y0[case1],w1.transpose()))
        w1=w1+(n*np.dot(delta1.transpose(),y0[case1]))
        
        y1[case1]=vector_sigmoid(np.dot(y0[case1],w1.transpose()))
        y2[case1]=vector_sigmoid(np.dot(y1[case1],w2.transpose()))
        y3[case1]=vector_sigmoid(np.dot(y2[case1],w3.transpose()))

print()        
print("Weights 1 :",w1)
print()
print("Weights 2 :",w2)
print()
print("Weights 3 :",w3)
print()
print("Number of epochs :",epoch)
print()
print("Root square Error :",vector_error_previous)
        
    
tester=int(input("Number of testcases >>> "))
list1=[]

for tester1 in range(tester):
    list2=[]
    for tester2 in range(x0):
        print("Value of parameter",tester2+1,"in case",tester1+1)
        input_x=float(input(">>> "))
        list2=list2+[input_x]
    list1=list1+[[list2]]
    
y1=np.ones((tester,1,x1))   # Y(1)
y2=np.ones((tester,1,x2))   # Y(2)
y3=np.ones((tester,1,x3))   # Outputs

for tester3 in range(len(list1)):
    y1[tester3]=vector_sigmoid(np.dot(list1[tester3],w1.transpose()))
    y2[tester3]=vector_sigmoid(np.dot(y1[tester3],w2.transpose()))
    y3[tester3]=vector_sigmoid(np.dot(y2[tester3],w3.transpose()))
print("y1 :",y1)    
print("y2 :",y2)  
print("Outputs :",y3)





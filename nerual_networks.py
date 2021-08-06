import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_prop(inp, w1, w2, w3, b1, b2, b3):
    a1 = sigmoid(np.dot(w1, inp) + b1)
    a2 = sigmoid(np.dot(w2, a1) + b2)
    a3 = sigmoid(np.dot(w3, a2) + b3)
    return {'a1': a1, 'a2': a2, 'a3': a3}

def cost(y, a):
    return (- np.dot(y.T, np.log(a)) - np.dot((1-y).T, np.log(1-a)))

def backward_prop(inp, w1, w2, w3, b1, b2, b3, y):
    l = len(y)
    dic = forward_prop(inp, w1, w2, w3, b1, b2, b3)
    dz3 = dic['a3'] - y
    db3 = np.sum(dz3)
    dw3 = np.dot(dz3, dic['a2'].T)
    dz2 = np.dot(dic['a2'].T,1-dic['a2'])*np.dot(w3.T, dz3)
    db2 = np.sum(dz2)
    dw2 = np.dot(dz2, dic['a1'].T)
    dz1 = np.dot(dic['a1'].T,1-dic['a1'])*np.dot(w2.T, dz2)
    db1 = np.sum(dz1)
    dw1 = np.dot(dz1, inp.T)
    return {'dw3':dw3/l, 'dw2':dw2/l, 'dw1':dw1/l, 'db3':db3/l, 'db2':db2/l, 'db1':db1/l}

def gradient_descent(inp, w1, w2, w3, b1, b2, b3, y, lr, num_ep):
    for _ in range(num_ep):
        a = forward_prop(inp, w1, w2, w3, b1, b2, b3)['a3']
        c = cost(y, a)
        dic = backward_prop(inp, w1, w2, w3, b1, b2, b3, y)
        w1 -= lr*dic['dw1']
        w2 -= lr*dic['dw2']
        w3 -= lr*dic['dw3']
        b1 -= lr*dic['db1']
        b2 -= lr*dic['db2']
        b3 -= lr*dic['db3']
    return {'w3':w3, 'w2':w2, 'w1':w1, 'b3':b3, 'b2':b2, 'b1':b1}

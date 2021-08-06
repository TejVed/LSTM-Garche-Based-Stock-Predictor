import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_prop(inp, w1, w2, w3, b1, b2, b3):
    a1 = sigmoid(np.dot(w1.T, inp) + b1)
    a2 = sigmoid(np.dot(w2.T, a1) + b2)
    a3 = 2*(((np.dot(w3.T, a2) + b3) > 0).astype(float)) - 1.0
    return {'a1': a1, 'a2': a2, 'a3': a3}

def backward_prop(inp, y, w1, w2, w3, b1, b2, b3, lr):
    dic = forward_prop(inp, w1, w2, w3, b1, b2, b3)
    t = (dic['a3'] - y.astype(float))[0]
    w1 -= t*inp*lr
    w2 -= t*dic['a1']*lr
    w3 -= t*dic['a2']*lr
    b1 -= t*lr
    b2 -= t*lr
    b3 -= t*lr
    return (w1, w2, w3, b1, b2, b3)

def training(x, y, num_epochs, lr):
    w1 = np.random.randn(3,3)
    w2 = np.random.randn(3,3)
    w3 = np.random.randn(3,1)
    b1 = np.random.randn(3,1)
    b2 = np.random.randn(3,1)
    b3 = np.random.randn(1,1)
    for _ in range(num_epochs):
        count = 0
        for i in range(len(x)):
            (w1, w2, w3, b1, b2, b3) = backward_prop(np.reshape(x[i].T,(3,1)), y[i], w1, w2, w3, b1, b2, b3, lr)
            a3 = forward_prop(x[i], w1, w2, w3, b1, b2, b3)['a3']
            if a3[0][0]==float(y[i]):
                count+=100
        print("EPOCH" + str(_) + "  :  " + str(count/len(x)))
    return (w1, w2, w3, b1, b2, b3)

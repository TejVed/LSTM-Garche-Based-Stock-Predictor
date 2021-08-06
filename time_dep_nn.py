import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mserr(a, y, n):
    return ((a-y)**2)[0][0]/n

def forward_prop(inp, w1, w2, b1, b2):
    a1 = sigmoid(np.dot(w1.T, inp) + b1)
    a2 = sigmoid(np.dot(w2.T, a1) + b2)
    return {'a1': a1, 'a2': a2}

def backward_prop(inp, y, w1, w2, b1, b2, w1_prev, w2_prev, b1_prev, b2_prev, n_p, lr, alpha):
    dic = forward_prop(inp, w1, w2, b1, b2)
    a1 = dic['a1']
    a2 = dic['a2']
    dz2 = 2*a2*(1-a2)*(a2-y)/n_p
    dw2 = np.dot(a1,dz2.T)
    db2 = dz2
    dz1 = w2*dz2*a1*(1-a1)
    dw1 = np.dot(inp,dz1.T)
    db1 = dz1
    temp = w1
    w1 -= lr*dw1 + alpha * (w1_prev - w1)
    w1_prev = temp
    temp = w2
    w2 -= lr*dw2 + alpha * (w2_prev - w2)
    w2_prev = temp
    temp = b1
    b1 -= lr*db1 + alpha * (b1_prev - b1)
    b1_prev = temp
    temp = b2
    b2 -= lr*db2 + alpha * (b2_prev - b2)
    b2_prev = temp
    return (w1, w2, b1, b2, w1_prev, w2_prev, b1_prev, b2_prev)

def training(x, num_epochs, n_p, n1, lr, alpha):
    w1 = np.random.randn(n_p, n1)
    w2 = np.random.randn(n1,1)
    b1 = np.random.randn(n1,1)
    b2 = np.random.randn(1,1)
    w1_prev = w1
    w2_prev = w2
    b1_prev = b1
    b2_prev = b2
    for _ in range(num_epochs):
        mse = 0
        for i in range(n_p, len(x)):
            inp = x[i-n_p:i]
            (w1, w2, b1, b2, w1_prev, w2_prev, b1_prev, b2_prev) = backward_prop(inp, x[i], w1, w2, b1, b2, w1_prev, w2_prev, b1_prev, b2_prev, n_p, lr, alpha)
            a2 = forward_prop(inp, w1, w2, b1, b2)['a2']
            mse += mserr(x[i], a2, len(x)-n_p)
        print("EPOCH" + str(_) + "  :  " + str(mse))
    return (w1, w2, b1, b2)

def testing(x, y, n_p, w1, w2, b1, b2):
    arr = np.concatenate((x,y))
    ans = np.zeros((len(y),1))
    mse = 0
    for i in range(len(x), len(arr)):
        t = forward_prop(arr[i-n_p:i], w1, w2, b1, b2)['a2']
        mse += mserr(t, arr[i], len(y))
        ans[i - len(x)][0] = t[0][0]
    print("Output is ")
    print(ans)
    print("MSE is ")
    print(mse)
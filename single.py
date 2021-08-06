import numpy as np

def forward_prop(inp, w, b):
    return (2*(((np.dot(w.T, inp) + b) > 0).astype(float)) - 1.0)

def backward_prop(inp, y, w, b, lr):
    a = forward_prop(inp, w, b)
    w -= ((a-y)[0])*inp*lr
    b -= ((a-y)[0])*lr
    return (w, b)

def training(x, y, num_epochs, lr):
    w = np.random.randn(3,1)
    b = np.random.randn(1,1)
    print(w)
    print(b)
    for _ in range(num_epochs):
        count = 0
        for i in range(len(x)):
            (w, b) = backward_prop(np.reshape(x[i].T,(3,1)), y[i], w, b, lr)
            a = forward_prop(x[i], w, b)
            if a[0][0]==float(y[i]):
                count+=100
        print("EPOCH" + str(_) + "  :  " + str(count/len(x)))
    return (w, b)

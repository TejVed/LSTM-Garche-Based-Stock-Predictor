import random
import numpy as np
from arch import arch_model

def create_set(n, alpha, beta, omega):
    series = [random.randn(), random.randn()]
    std_dev = [1,1]
    for i in range(n):
        std_dev.append(np.sqrt(omega + alpha*series[-1]**2 + beta*std_dev[-1]**2))
        series.append(std_dev[-1] * random.randn())
    return (series, std_dev)

def create_model():
    series, std_dev = create_set(1000, 0.1, 0.2, 0.3)
    model = arch_model(series, p=1, q=1)
    model = model.fit()
    return model
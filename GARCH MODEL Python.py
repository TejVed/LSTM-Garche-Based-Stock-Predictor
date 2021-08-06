# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 20:48:35 2021

@author: Hanan
"""

from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from arch.__future__ import reindexing


n = 1000
omega = 0.5
alpha_1 = 0.1
alpha_2 = 0.2
beta_1 = 0.3
beta_2 = 0.4

test_size = int(0.1*n)

series = [gauss(0,1),gauss(0,1)]
vols = [1, 1]

for _ in range(n):
    new_vol = np.sqrt(omega + alpha_1*series[-1]**2 + alpha_2*series[-2]**2 + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = gauss(0,1) * new_vol
    vols.append(new_vol)
    series.append(new_val)


train, test = series[:-test_size],series[-test_size:]
model = arch_model(train,p=2,q=2)
model_fit = model.fit()
model_fit.summary()

live_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    live_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
plt.figure(figsize=(100,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(live_predictions)
plt.title('Live Prediction',fontsize=20)
plt.legend(['True Volatility','Predicted Volatility'],fontsize=16)
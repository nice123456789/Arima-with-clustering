# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:04:09 2020

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



def slope(a):
	x_s=0
	y_s=0
	xy_s=0
	x2_s=0
	i=0
	n =len(a)
	while i<len(a):
		x_s+=i
		y_s+=a[i]
		xy_s+=a[i]*i
		x2_s=i*i
		i=i+1
	ans = (y_s*x2_s-x_s*xy_s)//(n*x2_s-(x_s)**2)
	return int(ans)

        

def ar(train,test,df):
    stepwise_model.fit(train)
    future_forecast = stepwise_model.predict(n_periods=200)
    future_forecast = pd.DataFrame(future_forecast,index = test.index)
    plt.figure(figsize=(20,3))
    plt.plot(test, label="Actual Values")
    plt.plot(future_forecast, label="Predicted Values")
    plt.legend()
    plt.show()
    rmse =np.sqrt(mean_squared_error(test,future_forecast))
    mae=np.sqrt(mean_absolute_error(test,future_forecast))
    return rmse,mae
    
series = pd.read_csv('wind dataset-2007.csv', header=3);
series = series[:2000]
train_index = int(0.9 * len(series))
df=series['wind_sp']
train = series[:train_index]
test = series[train_index:]
trn=train['wind_sp']
tst=test['wind_sp']
arr= trn
arr4 = []
arr3=[]
i=0
while i<arr.size:
	arr4.append(arr[i])
	i=i+1
k=20
i=0
mapp = {}
maplist = {}
slopeTimeSeries = {}
sprms={}
smae={}
while i<len(arr4):
	temp = []
	for j in range(0,k):
		if i+j<len(arr4):
			temp.append(arr4[i+j])
	sl = slope(temp) 
	if sl in mapp:
		mapp[sl]=mapp[sl]+1
	else:
		mapp[sl]=1
		maplist[sl]=[]
		slopeTimeSeries[sl]=[]
	   
	maplist[sl].append(i)
	for x in temp:
	   slopeTimeSeries[sl].append(x)
	i=i+k
for x in mapp:
	print(x)
	for y in maplist[x]:
		print(y,end=" ")
	print("\n")


stepwise_model = auto_arima(df, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
for x in slopeTimeSeries:
    plt.plot(slopeTimeSeries[x],label="Cluster "+str(x))
i=0
for x in slopeTimeSeries:
    point=ar(slopeTimeSeries[x],tst,df)
    sprms[i]=point[0]
    smae[i]=point[1]
    i=i+1
for x in sprms:
    print(sprms[x],"   ",smae[x])

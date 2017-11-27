# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
get_ipython().system(u'ls -F --color ')
get_ipython().magic(u'cd PSA')
get_ipython().system(u'ls -F --color ')
bus1=np.load("bus1.npy")
import numpy as np
bus1=np.load("bus1.npy")
model=Sequential()
model.add(Dense(200,input_dim=24*7,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(24,activation='relu'))
get_ipython().magic(u'pinfo model.compile')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
bus1[:168]
bus1[:24*7]
bus1[24*7:24*8]
x=np.array((168,1000))
x[0]
x=np.zeros((168,1000))
x.shape
x[0]
x.shape
len(x[0])
x=np.zeros((1000,168))
y=np.zeros((1000,24))
for i in range(1000):
    x[i]=bus1[24*i:24*i+168]
    y[i]=bus1[24*i+168:24*i+168+24]
    
x[0]
x[1]
model.fit(x,y,epochs=100,batch_size=10)
24*999+168
len(bus1)
x=np.zeros((2000,168))
y=np.zeros((2000,24))
for i in range(2000):
    x[i]=bus1[24*i:24*i+168]
    y[i]=bus1[24*i+168:24*i+168+24]
    
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
get_ipython().magic(u'pinfo model.compile')
get_ipython().magic(u'pinfo model.optimizers')
get_ipython().magic(u'pinfo model.compile')
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='mean_squared_logarithmic_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
x
y
x.shape
y.shape
model
model.loss
model.losses
model.loss_weights()
model.loss_weights
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='hinge',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.compile(loss='cosine_proximity',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=1000,batch_size=20)
model.fit(x,y,epochs=1000,batch_size=50)
model.fit(x,y,epochs=1000,batch_size=10)
model.fit(x,y,epochs=100,batch_size=10)
x
y
model=Sequential()
model.add(Dense(200,input_dim=24*7,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(24,activation='relu'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=100,batch_size=10)
model.evaluate(x,y)
bus1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(bus1)
bus1.reshape(-1,1)
scaler.fit(bus1.reshape(-1,1))
scaler.transform(bus1.reshape(-1,1))
bus1scaled=scaler.transform(bus1.reshape(-1,1))
bus1scaled.shape
x=np.zeros((2000,168))
y=np.zeros((2000,24))
for i in range(2000):
    x[i]=bus1[24*i:24*i+168]
    y[i]=bus1[24*i+168:24*i+168+24]
    
x=np.zeros((2000,168))
y=np.zeros((2000,24))
for i in range(2000):
    x[i]=bus1scaled[24*i:24*i+168]
    y[i]=bus1scaled[24*i+168:24*i+168+24]
    
bus1scaled.reshape((52584,1))
bus1scaled.reshape((52584))
bus1scaled=bus1scaled.reshape((52584))
for i in range(2000):
    x[i]=bus1scaled[24*i:24*i+168]
    y[i]=bus1scaled[24*i+168:24*i+168+24]
    
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
get_ipython().system(u'clear ')
get_ipython().system(u'clear ')
x
y
model.compile(loss="mean_absolute_error",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.evaluate(x,y)
model.compile(loss="mean_absolute_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.evaluate(x,y)
model.compile(loss="mean_absolute_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_absolute_error",optimizer="radadelta",metrics=["accuracy"])
model.compile(loss="mean_absolute_error",optimizer="adadelta",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model=Sequential()
model.add(Dense(30,input_dim=24,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(6,activation='relu'))
x=np.zeros((10000,24))
y=np.zeros((10000,6))
for i in range(10000):
    x[i]=bus1scaled[6*i:6*i+24]
    y[i]=bus1scaled[6*i+24:6*i+24+6]
    
i
bus1scaled[6*i+24:6*i+24+6]
bus1scaled[6*i+24-6:6*i+24]
6*i+24
x=np.zeros((8000,24))
y=np.zeros((8000,6))
for i in range(8000):
    x[i]=bus1scaled[6*i:6*i+24]
    y[i]=bus1scaled[6*i+24:6*i+24+6]
    
x
bus1scaled
y
model.compile(loss="mean_squared_error",optimizer="adadelta",metrics=["accuracy"]
)
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_absolute_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
model.compile(loss="mean_absolute_error",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model=Sequential()
model.add(Dense(400,input_dim=2*24*7,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(24,activation='relu'))
x=np.zeros((2000,168*2))
y=np.zeros((2000,6))
y=np.zeros((2000,24))
for i in range(2000):
    x[i]=bus1scaled[24*i:24*i+24*7*2]
    y[i]=bus1scaled[24*i+24*7*2:24*i+24*7*2+24]
    
model.compile(loss="mean_absolute_error",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_square_error",optimizer="rmsprop",metrics=["accuracy"])
model.compile(loss="mean_squared_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.evaluate(x,y)
x.shape
y.shape
model=Sequential()
model.add(Dense(400,input_dim=2*24*7,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(24,activation='relu'))
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model=Sequential()
model.add(Dense(2*24*7,input_dim=2*24*7,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(24,activation='relu'))
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model=Sequential()
model.add(Dense(400,input_dim=2*24*7,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(24,activation='relu'))
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_squared_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
model.compile(loss="mean_squared_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model=Sequential()
model.add(Dense(10,input_dim=6,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='relu'))
x=np.zeros((10000,6))
y=np.zeros(10000)
for i in range(10000):
    x[i]=bus1scaled[6*i:6*i+6]
    y[i]=bus1scaled[6*i+6]
    
52584/6
x=np.zeros((8500,6))
y=np.zeros(8500)
for i in range(8500):
    x[i]=bus1scaled[6*i:6*i+6]
    y[i]=bus1scaled[6*i+6]
    
model.compile(loss="mean_squared_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
y=np.zeros((8500,1))
y[0]
for i in range(8500):
    x[i]=bus1scaled[6*i:6*i+6]
    y[i]=bus1scaled[6*i+6]
    
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
get_ipython().system(u'clear ')
model=Sequential()
model.add(Dense(50,input_dim=24,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(24,activation='relu'))
x=np.zeros((2000,24))
y=np.zeros((2000,24))
for i in range(8500):
    x[i]=bus1scaled[24*i:24*i+24]
    y[i]=bus1scaled[24*i+24:24*i+48]
    
for i in range(2000):
    x[i]=bus1scaled[24*i:24*i+24]
    y[i]=bus1scaled[24*i+24:24*i+48]
    
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.compile(loss="mean_absolute_error",optimizer="adamax",metrics=["accuracy"])
model.f
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model.evaluate(x,y)
model=Sequential()
model.add(Dense(200,input_dim=24*5,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(24*5,activation='relu'))
x=np.zeros((2000,24*5))
y=np.zeros((2000,24*5))
len(bus1)
len(bus1)/120
x=np.zeros((400,24*5))
y=np.zeros((1000,24*5))
x=np.zeros((1000,24*5))
for i in range(1000):
    x[i]=bus1scaled[24*5*i:24*5*i+24*5]
    y[i]=bus1scaled[24*5*i+24*5:24*5*i+24*10]
    
x=np.zeros((400,24*5))
y=np.zeros((400,24*5))
for i in range(1000):
    x[i]=bus1scaled[24*5*i:24*5*i+24*5]
    y[i]=bus1scaled[24*5*i+24*5:24*5*i+24*10]
    
for i in range(400):
    x[i]=bus1scaled[24*5*i:24*5*i+24*5]
    y[i]=bus1scaled[24*5*i+24*5:24*5*i+24*10]
    
x
model.compile(loss="mean_absolute_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
for i in range(400):
    x[i]=bus1scaled[120*i:120*i+120]
    y[i]=bus1scaled[120*i+120:120*i+240]
    
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
len(bus1)/168
model=Sequential()
model.add(Dense(10,input_dim=6,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(6,activation='relu'))
len(bus1)/6
x=np.zeros((8000,6))
y=np.zeros((8000,6))
for i in range(8000):
    x[i]=bus1scaled[6*i:6*i+6]
    y[i]=bus1scaled[6*i+6:6*i+12]
    
model.compile(loss="mean_absolute_error",optimizer="adagrad",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="armsprop",metrics=["accuracy"])
model.compile(loss="mean_absolute_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model=Sequential()
model.add(Dense(20,input_dim=12,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(12,activation='relu'))
x=np.zeros((4000,12))
y=np.zeros((4000,12))
for i in range(4000):
    x[i]=bus1scaled[12*i:12*i+12]
    y[i]=bus1scaled[12*i+12:12*i+24]
    
model.compile(loss="mean_absolute_error",optimizer="rmsprop",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="ada,",metrics=["accuracy"])
model.compile(loss="mean_absolute_error",optimizer="adam,",metrics=["accuracy"])
model.compile(loss="mean_absolute_error",optimizer="adam",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model.compile(loss="mean_absolute_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=1)
model=Sequential()
from keras import LSTM
from keras.layers import LSTM
model.add(LSTM(4,input_shape=(1,24)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
x=np.zeros((2000,24))
y=np.zeros(2000)
for i in range(2000):
    x[i]=bus1scaled[24*i:24*i+24]
    y[i]=bus1scaled[24*i+24]
    
model.fit(x,y,epochs=100,batch_size=1)
x[0]
tx=np.reshape(x,(x.shape[0],1,x.shape[1]))
tx
tx.shape
model.fit(tx,y,epochs=100,batch_size=1)
model.predict(tx)
model.predict(tx)[:,0]
math.sqrt(mean_squared_error(y[0],model.predict(tx)[:,0])
)
import math
math.sqrt(mean_squared_error(y[0],model.predict(tx)[:,0])
)
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y[0],model.predict(tx)[:,0])
)
y[0]
model.predict(tx)[:,0]
len(model.predict(tx)[:,0])
len(model.predict(tx)[0,:])
len(model.predict(tx)[0])
model.predict(tx)[0]
model.predict(tx)[1]
model.predict(tx)[2]
math.sqrt(mean_squared_error(y[0],model.predict(tx)[0]))
y[0]-model.predict(tx)[0]
model.predict(tx)
model.predict(tx)
y
plot(model.predict(tx))
from pylab import *
plot(model.predict(tx))
plot(y)
show()
plot(model.predict(tx))
plot(y)
show()
model.fit(tx,y,epochs=100,batch_size=1)
plot(model.predict(tx))
plot(y)
show()
model.predict(tx).shape
y.shape
y=y.reshape((2000,1))
y.shape
mean_squared_error(y,model.predict(tx))
math.sqrt(mean_squared_error(y,model.predict(tx)))
tp=scaler.inverse_transform(model.predict(tx))
tp
ty=scaler.inverse_transform(ty)
ty=scaler.inverse_transform(y)
ty
math.sqrt(mean_squared_error(ty,tp))
math.sqrt(mean_squared_error(y,model.predict(tx)))
mean(y)
mean(ty)
32./4850.
model.fit(tx,y,epochs=100,batch_size=10)
tp=scaler.inverse_transform(model.predict(tx))
tp
ty=scaler.inverse_transform(y)
math.sqrt(mean_squared_error(ty,tp))
x=np.zeros((1000,24))
y=y.reshape((1000,1))
y=np.zeros(1000)
for i in range(1000):
    x[i]=bus1scaled[24*i:24*i+24]
    y[i]=bus1scaled[24*i+24]
    
tx=np.reshape(x,(x.shape[0],1,x.shape[1]))
model.fit(tx,y,epochs=100,batch_size=10)
model.fit(tx,y,epochs=100,batch_size=1)
model.fit(tx,y,epochs=100,batch_size=10)
tp=scaler.inverse_transform(model.predict(tx))
ty=scaler.inverse_transform(y)
y=y.reshape((1000,1))
ty=scaler.inverse_transform(y)
math.sqrt(mean_squared_error(ty,tp))
testx=np.zeros((1000,24))
testy=np.zeros(1000)
for i in range(1000,2000):
    testx[i]=bus1scaled[24*i:24*i+24]
    testy[i]=bus1scaled[24*i+24]
    
for i in range(1000,2000):
    testx[i-1000]=bus1scaled[24*i:24*i+24]
    testy[i-1000]=bus1scaled[24*i+24]
    
testx=np.reshape(testx,(testx.shape[0],1,testx.shape[1]))
model.predict(testx)
testp=model.predict(testx)
testp=scaler.inverse_transform(testp)
testy
testy=scaler.inverse_transform(testy)
testy=testy.reshape((-1,1))
testy=scaler.inverse_transform(testy)
testy
math.sqrt(mean_squared_error(testy,testp))
model.fit(tx,y,epochs=100,batch_size=1)
model.evaluate(testx,testy)
plot(testx,testy)
plot(testx)
testx
plot(testy)
plot(testp)
show()
ty
plot(tx)
plot(y)
plot(model.predict(tx))
show()
plot(testy)
plot(testp)
show()
plot(y)
plot(tx)
plot(model.predict(tx))
show()
testy=scaler.inverse_transform(testy)
testy
testy=np.zeros(1000)
testy=model.predict(testx)
testy
testy=scaler.inverse_transform(testy)
testy
plot(testy)
plot(testp)
show()
testp
plot(testy)
plot(testp)
show()

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

from pylab import *

#this function runs the model on the input bus (min size=48000 hours, can be modified within)
#input = bus name, optionally number of epochs (iterations) and batch_size (defaults are fine)
def modelAndPredict(bus, num_epochs=20, batchsize=1):

	#scales the input data to (0,1)
	scaler=MinMaxScaler()
	scaler.fit(bus)
	scaledbus=scaler.transform(bus)

	#creates a sequential LSTM layered model in Keras
	model=Sequential()
	model.add(LSTM(4,input_shape=(1,24)))
	model.add(Dense(1))
	model.compile(loss="mean_squared_error",optimizer="adam")

	#create the arrays to hold train/test datasets
	trainx=np.zeros((100,24))
	testx=np.zeros((100,24))
	trainy=np.zeros(100)
	testy=np.zeros(100)
	
	#setup the data in the arrays
	for i in range(100):
		trainx[i]=scaledbus[24*i:24*i+24].reshape(24)
		trainy[i]=scaledbus[24*i+24]

	for i in range(100,200):
		testx[i-100]=scaledbus[24*i:24*i+24].reshape(24)
		testy[i-100]=scaledbus[24*i+24]

	#modify the numpy array for the LSTM model (required by the function)
	trainx=np.reshape(trainx,(trainx.shape[0],1,trainx.shape[1]))
	testx=np.reshape(testx,(testx.shape[0],1,testx.shape[1]))

	#run the model on the training dataset
	model.fit(trainx,trainy,epochs=num_epochs,batch_size=batchsize)

	#predictions to be made on test data:
	tpred=model.predict(testx)
	tpred_scaled=scaler.inverse_transform(tpred.reshape((-1,1)))
	texact_scaled=scaler.inverse_transform(testy.reshape((-1,1)))
	
	plot(tpred_scaled.reshape((len(tpred),1)),label="Predicted Power")
	plot(texact_scaled.reshape((len(tpred),1)),label="Measured Power")
	title("Load Prediction on Test Dataset")
	xlabel("Index")
	ylabel("Power (in MW)")
	legend()
	show()

	tpred=model.predict(trainx)
	tpred_scaled=scaler.inverse_transform(tpred.reshape((-1,1)))
	texact_scaled=scaler.inverse_transform(trainy.reshape((-1,1)))
	plot(tpred_scaled.reshape((len(tpred),1)),label="Predicted Power")
	plot(texact_scaled.reshape((len(tpred),1)),label="Measured Power")
	title("Load Prediction on Test Dataset")
	xlabel("Index")
	ylabel("Power (in MW)")
	legend()
	show()
	
	testScore=math.sqrt(mean_squared_error(texact_scaled,tpred_scaled))
	print("Test Score: %.2f percent accuracy, on average, on the test data" % (100-testScore*100/np.mean(bus)))	
	

#Load the hourly power log for each bus
#Reshaping is required for the LSTM model to work properly
bus1=np.load("bus1.npy").reshape((-1,1))
bus2=np.load("bus2.npy").reshape((-1,1))
bus3=np.load("bus3.npy").reshape((-1,1))
bus4=np.load("bus4.npy").reshape((-1,1))


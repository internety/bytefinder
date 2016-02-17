import time

from keras.models import Sequential, model_from_json
from keras.layers import containers
from keras.layers.core import TimeDistributedDense, Activation, Dropout, AutoEncoder
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import EarlyStopping

import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt

import theano
theano.config.mode = 'FAST_RUN'
theano.config.floatX = 'float32'

#
# Model I/O
#

# Save model
def saveModel(model):
	t = int(time.time())
	open('models/%s_meta.json' % t, 'w').write(model.to_json())
	model.save_weights('models/%s_data.h5' % t, overwrite=True)
	return None

# Load model
def loadModel(name):
	model = model_from_json(open('models/%s_meta.json' % name).read())
	model.load_weights('models/%s_data.h5' % name)
	return model

# Use model
def runModel(inMatrix, model):
	return model.predict(inMatrix)


def trainEncoder(inMatrix):

	print("Compiling Encoder...")
	inSize = inMatrix.shape[2]
	encoder = containers.Sequential([LSTM(output_dim=inSize/2, input_dim = inSize, return_sequences=True), \
	                                 Dropout(0.8), \
                                     LSTM(output_dim=inSize/4, input_dim = inSize/2, return_sequences=True), \
                                     Dropout(0.4), \
                                     LSTM(output_dim=inSize/8, input_dim = inSize/4, return_sequences=True), \
                                     Dropout(0.2), \
                                     LSTM(output_dim=inSize/16, input_dim = inSize/8, return_sequences=True)])
	decoder = containers.Sequential([LSTM(output_dim=inSize/8, input_dim = inSize/16, return_sequences=True), \
	                                 LSTM(output_dim=inSize/4, input_dim = inSize/8, return_sequences=True), \
	                                 LSTM(output_dim=inSize/2, input_dim = inSize/4, return_sequences=True), \
                                     LSTM(output_dim=inSize, input_dim = inSize/2, return_sequences=True)])
									 
	autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
	
	model = Sequential()
	model.add(autoencoder)
	model.compile(loss='mse', optimizer='rmsprop')
	
	# Train model
	print("Training Encoder...")
	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.fit(inMatrix, inMatrix, batch_size=30, validation_split=0.15, callbacks=[early_stopping], verbose=1)

	autoencoder.output_reconstruction = False
	return model

#
# Train model
#

def trainModel(inMatrix, targMatrix):
	assert inMatrix.shape[:-1] == targMatrix.shape[:-1]
	
	targMatrix = targMatrix[:,:,:1]
	
	autoencoder = trainEncoder(inMatrix)
	inMatrix = autoencoder.predict(inMatrix)
	
	# Compile model
	print("Compiling Model...")
	inSize, outSize = inMatrix.shape[2], targMatrix.shape[2]
	
	model = Sequential()
	model.add(LSTM(output_dim=inSize//3, input_dim=inSize, return_sequences=True))
	model.add(LSTM(output_dim=outSize, input_dim=inSize//3, return_sequences=True))
	
	model.compile(loss="mse", optimizer="rmsprop")
	
	# Train model
	print("Training Model...")
	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.fit(inMatrix, targMatrix, batch_size=30, validation_split=0.15, callbacks=[early_stopping], verbose=1)
	
	# Save model
	print("Saving Model...")
	saveModel(model)

	return model
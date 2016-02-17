import time

from keras.models import Sequential, model_from_json
from keras.layers import containers
from keras.layers.core import TimeDistributedDense, Activation, Dropout, AutoEncoder, Dense
from keras.layers.embeddings import Embedding
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

#
# Train model
#

def trainModel(inMatrix, targMatrix):
	assert inMatrix.shape[:-1] == targMatrix.shape[:-1]
	
	targMatrix = targMatrix[:,:,:1]
	
	# Compile model
	print("Compiling Model...")
	
	model = Sequential()
	model.add(GRU(input_dim=inMatrix.shape[2], output_dim=inMatrix.shape[2]/2, return_sequences=True))
	model.add(TimeDistributedDense(input_dim=inMatrix.shape[2]/2, output_dim=targMatrix.shape[2]))
	model.compile(loss='mse', optimizer='adam')
	
	# Train model
	print("Training Model...")
	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.fit(inMatrix, targMatrix, batch_size=30, validation_split=0.15, callbacks=[early_stopping], verbose=1)
	
	# Save model
	print("Saving Model...")
	saveModel(model)

	return model
import time

from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

import numpy as np
np.random.seed(1)

import theano
theano.config.mode = 'FAST_RUN'
theano.config.floatX = 'float32'

# Save model
def save(model):
	t = int(time.time())
	open('models/%s_meta.json' % t, 'w').write(model.to_json())
	model.save_weights('models/%s_data.h5' % t, overwrite=True)
	return None

# Load model
def load(name):
	model = model_from_json(open('models/%s_meta.json' % name).read())
	model.load_weights('models/%s_data.h5' % name)
	return model

# Use model
def run(inMatrix, model):
	return model.predict(inMatrix)

# Train model
def train(inMatrix, targMatrix):

	print("Compiling Model...")
	model = Sequential()
	model.add(LSTM(input_dim=inMatrix.shape[2], output_dim=4, return_sequences=False))
	model.add(Dropout(0.5))
	model.add(Dense(input_dim=1, output_dim=targMatrix.shape[1]))
	model.add(Activation('softmax'))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	print("Training Model...")
	model.fit(inMatrix, targMatrix, batch_size=30, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1)
	
	# Save model
	print("Saving Model...")
	save(model)

	return model
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import time, os

# Third-party Libraries
import numpy as np
np.random.seed(1)

import theano
theano.config.mode = 'FAST_RUN'
theano.config.floatX = 'float32'

from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

###############################################################################

# Save model
def save(model, classes):

	print("Saving Model...")
	t = int(time.time())
	os.makedirs('models/%s' % t)
	open('models/%s/meta.json' % t, 'w').write(model.to_json())
	model.save_weights('models/%s/data.h5' % t, overwrite=True)
	open('models/%s/classes.txt' % t, 'w').write(', '.join(classes))
	return None

# Load model
def load(name):

	print("Loading Model...")
	model = model_from_json(open('models/%s/meta.json' % name).read())
	model.load_weights('models/%s/data.h5' % name)
	classes = open('models/%s/classes.txt' % name).read().split(', ')
	return model, classes

# Use model
def run(inMatrix, model):
	return model.predict(inMatrix)

# Train model
def train(inMatrix, targMatrix):

	print("Compiling Model...")
	model = Sequential()
	model.add(GaussianNoise(sigma=0.1, input_shape=inMatrix.shape[1:]))
	model.add(LSTM(input_dim=inMatrix.shape[2], output_dim=1, return_sequences=False))
	model.add(Dense(input_dim=1, output_dim=targMatrix.shape[1]))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print("Training Model...")
	model.fit(inMatrix, targMatrix, batch_size=30, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1)

	return model
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import time, os

# Third-party Libraries
import numpy as np
np.random.seed(1)

import tensorflow

from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.noise import GaussianNoise
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D
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

def build(inShape, targShape):

	print("Building Model...")
	model = Sequential()
	model.add(LSTM(input_shape=inShape[1:], output_dim=targShape[-1], return_sequences=True))
	return model

# Train model
def train(model, inMatrix, targMatrix):

	print("Compiling Model...")
	model.compile(loss='mse', optimizer='rmsprop')
	print("Training Model...")
	model.fit(inMatrix, targMatrix, batch_size=30, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1)

	return model
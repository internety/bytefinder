#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import time, os

# Third-party Libraries
import numpy as np
np.random.seed(1)

from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense
from keras.layers.recurrent import LSTM
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

# Load model
def load(name):

	print("Loading Model...")
	model = model_from_json(open('models/%s/meta.json' % name).read())
	model.load_weights('models/%s/data.h5' % name)
	classes = open('models/%s/classes.txt' % name).read().split(', ')
	return model, classes

# Use model
def run(model, inMatrix):

	print("Running Model...")
	return model.predict({'input':inMatrix})['output']

# Build model
def build(inShape, targShape):

	print("Building Model...")
	input = Input(inShape[1:])
	forward = LSTM(output_dim=targShape[-1], return_sequences=True)(input)
	backward = LSTM(output_dim=targShape[-1], return_sequences=True, go_backwards=True)(input)
	output = TimeDistributed(Dense(targShape[-1], activation='softmax'))(forward, backward)
	return Model(input=input, output=output)

# Train model
def train(model, inMatrix, targMatrix):

	print("Compiling Model...")
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print("Training Model...")
	model.fit(inMatrix, targMatrix, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
	return model
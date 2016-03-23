#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Local Libraries
import modeler
import data

###############################################################################

def main():
	retrain = True
	if retrain:
		input, target, classes = data.sample('data')
		model = modeler.build(input.shape, target.shape)
		modeler.train(model, input, target)
		modeler.save(model, classes)
	else:
		model, classes = modeler.load(sorted(os.listdir('models'))[-1])

	data.backtest(input, model.predict(target))

if __name__ == "__main__":
	main()
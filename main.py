#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import os

# Local Libraries
import modeler
import data

###############################################################################

def main():

	if not os.path.exists('data'):
		os.makedirs('data')

	if next(os.walk('data'))[1]:
		input, target, classes = data.sample('data')

		retrain = True
		if retrain:
			model = modeler.build(input.shape, target.shape)
			modeler.train(model, input, target)
			modeler.save(model, classes)
		else:
			model, classes = modeler.load(sorted(os.listdir('models'))[-1])
			output = modeler.run(model, input)
			data.backtest(classes, input, output)
	else:
		print("""\nNo data found.\nPut subfolders of files by class, within the 'data' folder.""")

if __name__ == "__main__":
	main()
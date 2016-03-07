#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Local Libraries
import modeler
import data

###############################################################################

def test(model, d):
	print('Backtesting...')
	with open('log.csv', 'a+') as log:
		for file in os.listdir(d):
			if not file.startswith('.'):
				with open(d + '/' + file) as f:
					s = f.read()
					result = modeler.run(data.preprocess(s, min(2000, len(s))), model)
					log.write('\n'+file+'\t')
					for i in xrange(result.shape[1]):
						log.write('%s\t' % result[0,i])


def main():
	
	retrain = True
	if retrain:
		input, target, classes = data.sample('data')
		model = modeler.build(input.shape, target.shape)
		modeler.train(model, input, target)
		modeler.save(model, classes)
	else:
		model, classes = modeler.load(sorted(os.listdir('models'))[-1])

	test(model, 'data/good')

if __name__ == "__main__":
	main()
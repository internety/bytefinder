#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Local Libraries
import modeler
import data

###############################################################################

def test(model, classes):
	d = 'data/test'
	with open('log.csv', 'a+') as log:
		for file in os.listdir(d):
			if not file.startswith('.'):
				with open(d + '/' + file) as f:
					s = f.read()
					result = modeler.run(data.preprocess(s, min(500, len(s))), model)
					log.write(file+'\n')
					for i in xrange(len(classes)):
						log.write('\t%s:\t%s\n' % (classes[i], result[0,i]))


def main():
	
	retrain = True

	if retrain:
		input, target, classes = data.sample('data/train')
		model = modeler.train(input, target)
		modeler.save(model, classes)

	else:
		model, classes = modeler.load(sorted(os.listdir('models'))[-1])

	test(model, classes)
	

if __name__ == "__main__":
	main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Local Libraries
import modeler
import data

###############################################################################

class colors:
    normal = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    blink_1 = '\033[5m'
    blink_2 = '\033[6m'
    fg_k = '\033[90m'
    fg_r = '\033[91m'
    fg_g = '\033[92m'
    fg_y = '\033[93m'
    fg_b = '\033[94m'
    fg_m = '\033[95m'
    fg_c = '\033[96m'
    fg_w = '\033[97m'
    bg_k = '\033[100m'
    bg_r = '\033[101m'
    bg_g = '\033[102m'
    bg_y = '\033[103m'
    bg_b = '\033[104m'
    bg_m = '\033[105m'
    bg_c = '\033[106m'
    bg_w = '\033[107m'

def test(model):
	print('Backtesting...')
	with open('log.csv', 'a+') as log:

		# For each sub-file/directory within dname
		for root, dirs, files in os.walk(dname):
			print('\tOpening %s...' % root)

			# For file in each directory
			for fname in files:
				if not fname.startswith('.'):
					try:
						with open('data/' + fname) as f:
							print("\t\tReading %s..." % fname[:40])
							fstring = f.read()
							result = modeler.run(data.preprocess(s, min(2000, len(fstring))), model)
							log.write('\n'+file+'\t')
							for i in xrange(result.shape[1]):
								log.write('%s\t' % result[0,i])
					except IOError:
						print("\t\tCould not read %s..." % fname)


def main():
	retrain = True
	if retrain:
		input, target, classes = data.sample('data')
		model = modeler.build(input.shape, target.shape)
		modeler.train(model, input, target)
		modeler.save(model, classes)
	else:
		model, classes = modeler.load(sorted(os.listdir('models'))[-1])

	test(model)

if __name__ == "__main__":
	main()
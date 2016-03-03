#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Third-party Libraries
import numpy as np

###############################################################################

def makeMatrix(string_list):
	result = []
	for s in string_list:
		fmat = np.fromstring(s, dtype=np.dtype('uint8'))
		one_hot = np.zeros((fmat.size, 256))
		one_hot[np.arange(fmat.size), fmat] = 1
		result.append(one_hot)
	return np.array(result)

def sample(dname):
	ncat = 100
	fsamps = 5 # Samples per file
	window = 160 # Timesteps per sample

	inList, targList = [], []
	classes = list(set('/'.join([x[0] for x in os.walk(dname)]).split('/')))
	classes.remove(dname)

	for root, dirs, files in os.walk(dname):
		
		if root != dname:
			print("Opening %s..." % root)
			target = np.zeros(len(classes))
			target[[classes.index(x) for x in root.split('/')[1:]]] = 1
			for fname in files[:ncat]:
				try:
					with open(root+'/'+fname) as f:
						if not fname.startswith('.'):
							print("\tReading %s..." % fname)
							fstring = f.read()
							if len(fstring) > window:
								for _ in xrange(fsamps):
									i = random.randint(0, len(fstring)-window)
									inList.append(fstring[i:i+window])
									targList.append(target)
				except IOError:
					print("\tCould not read %s..." % fname)

	print("Making Matrix...")
	inMatrix = makeMatrix(inList)
	targMatrix = np.array(targList)

	print("Input shape:\t(%s %s %s)" % inMatrix.shape)
	print("Target shape:\t(%s %s)" % targMatrix.shape)
	
	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p])
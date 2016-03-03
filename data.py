#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Third-party Libraries
import numpy as np

###############################################################################

# Given a file string 'fstring', string is selected (eg. in sentences) with length 'window'
def preprocess(fstring, window=100):

	i = random.randint(0, len(fstring)-window)
	fstring = fstring[i:i+window]
	
	fmat = np.fromstring(fstring, dtype=np.dtype('uint8'))
	result = np.zeros((1, fmat.size, 256))
	result[0, np.arange(fmat.size), fmat] = 1

	return result

# 
def sample(dname):
	ncat = 100		# Files per category
	fsamps = 5 		# Samples per file
	window = 160 	# Timesteps per sample

	inList, targList, classes = [], [], []

	classes = [root[root.rindex('/')+1:] for root, dirs, files in os.walk(dname) if [x for x in files if not x.startswith('.')]]

	for root, dirs, files in os.walk(dname):
			print('Opening %s...' % root)

			target = np.zeros(len(classes))
			target[[classes.index(x) for x in root.split('/')[1:] if x in classes]] = 1
			for fname in files[:ncat]:
				try:
					with open(root+'/'+fname) as f:
						if not fname.startswith('.'):
							print("\tReading %s..." % fname)
							fstring = f.read()
							if len(fstring) > window:
								for _ in xrange(fsamps):
									inList.append(preprocess(fstring, window))
									targList.append(target)
							
				except IOError:
					print("\tCould not read %s..." % fname)

	print("Making Matrix...")
	inMatrix = np.vstack(inList)
	targMatrix = np.array(targList)

	print(inMatrix.shape)
	print(targMatrix.shape)
	
	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p], classes)
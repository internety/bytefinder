#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os
random.seed(1)

# Third-party Libraries
import numpy as np
np.random.seed(1)

###############################################################################

def backtest(input, output):

	# For each sequence in input
	for sequence in xrange(input.shape[0]):

		print('-'*40)
		print(output[sequence])

		fstring = mat2str(input[sequence])
		print(fstring)
	return

# Given a file string 's',
# sample and output a numpy matrix with shape (1, len(s), 256)
def str2mat(s):
	smat = np.fromstring(s, dtype=np.dtype('uint8'))
	result = np.zeros((1, smat.size, 256))
	result[0, np.arange(smat.size), smat] = 1
	return result

def mat2str(smat):
	return (np.where(smat)[-1]).tostring().replace('\x00','')

# Sample a directory and all subdirectories
def sample(dname):
	ncat = 90		# Files per category
	fsamps = 10 	# Samples per file
	window = 200 	# Timesteps per sample

	inList, targList, classes = [], [], []
	classes = [root.split('/')[-1] for root, dirs, files in os.walk(dname) if '/' in root]

	# For each sub-file/directory within dname
	for root, dirs, files in os.walk(dname):
			print('Opening %s...' % root)
			target = np.array([1 if x in root.split('/')[1:] else 0 for x in classes])

			# For file in each directory
			random.shuffle(files)
			for fname in files[:ncat]:
				if not fname.startswith('.'):
					try:
						with open(root+'/'+fname) as f:
							print("\tReading %s..." % fname[:40])
							fstring = f.read()
							if len(fstring) > window:
								for _ in xrange(fsamps):
									i = random.randint(0, len(fstring)-window)
									inList.append(str2mat(fstring[i:i+window]))
									targList.append(target)
					except IOError:
						print("\tCould not read %s..." % fname)

	inMatrix, targMatrix = np.vstack(inList), np.array(targList)

	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p], classes)
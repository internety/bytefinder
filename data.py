#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os

# Third-party Libraries
import numpy as np

###############################################################################

class colors:
	normal = '\033[0m'
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

def backtest(input, output):

	# For each sequence in input
	for sequence in xrange(input.shape[0]):
		
		print('-'*40)
		cat_colors = [colors.fg_b, colors.fg_g, colors.fg_r]
		fstring = mat2str(input[sequence])
		fcat = output[sequence]

		# For each timestep in sequence
		for timestep in xrange(len(fstring)):
			char = fstring[timestep]
			cat = fcat[timestep]
			print(cat_colors[np.argmax(cat)] + char + colors.normal, end="")
		print()
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
			target = np.tile([1 if x in root.split('/')[1:] else 0 for x in classes], (window, 1))

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
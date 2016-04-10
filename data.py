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

class colors:
	normal = '\033[0m'
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

def backtest(classes, input, output):

	# Print classification labels
	cat_colors = [colors.fg_r, colors.fg_g, colors.fg_y, colors.fg_b, colors.fg_m, colors.fg_c, colors.fg_w]
	for i in xrange(len(classes)):
		print(cat_colors[i] + classes[i] + colors.normal, end=' ')
	print()

	# For each sequence in input
	for sequence in xrange(input.shape[0]):
		
		print('-'*40)
		fstring = mat2str(input[sequence])
		fcat = output[sequence]

		# For each timestep in sequence
		for timestep in xrange(len(fstring)):
			char = fstring[timestep]
			cat = fcat[timestep]
			print(cat_colors[np.argmax(cat)] + char + colors.normal, end='')
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
def sample(dname, window=100, size=10000):

	inList, targList = [], []

	# Calculate number of samples per directory,
	# based on number of siblings
	classes = []
	ncat = {dname:size}
	for root, dirs, files in os.walk(dname):
		classes.append(root.split('/')[-1])
		for d in dirs:
			ncat[root+'/'+d] = ncat[root]/len(dirs)
	classes.remove(dname)

	# For each sub-file/directory within dname
	for root, dirs, files in os.walk(dname):
		files = filter(lambda x: not x.startswith('.'), files)
		files = filter(lambda x: os.path.getsize(root+'/'+x)>window, files)

		if files:
			print('Opening %s...' % root)
			target = np.tile([1 if x in root.split('/') else 0 for x in classes], (1, window, 1))
	
			total_size = sum([os.path.getsize(root+'/'+file) for file in files])
			file_pdf = [os.path.getsize(root+'/'+file)/float(total_size) for file in files]
			for _ in xrange(ncat[root]):
				fpath = root+'/'+np.random.choice(files, p=file_pdf)
				with open(fpath) as f:
					f.seek(random.randint(0, os.path.getsize(fpath)-window))
					inList.append(str2mat(f.read(window)))
					targList.append(target)

	inMatrix, targMatrix = np.vstack(inList), np.vstack(targList)

	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p], classes)
import csv, random
import numpy as np


def readstr(s):
	fmat = np.fromstring(s, dtype=np.dtype('uint8'))
	result = np.zeros((fmat.size, 256))
	result[np.arange(fmat.size), fmat] = 1
	return np.expand_dims(result, axis=0)

# File I/O
def readfile(fname):
	print "Reading %s..." % fname
	try:
		with open(fname) as f:
			s = f.read()
			return readstr(s)
	except IOError:
		return np.empty((0,0,0))

def readfiles():
	
	with open('classes.csv', 'rb') as csvfile:
		csvlist = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
		flist = csvlist[1:]
		classes = csvlist[0][1:]
	random.shuffle(flist)
	files = [(x[0], np.array(x[1:], dtype=float)) for x in flist]

	fsamps = 500 # Samples per file
	window = 160 # Timesteps per sample
	inMatrix, targMatrix = [], []
	for f in files:
		fmat = readfile(f[0])
		if fmat.shape[1] > window:
			result = []
			for x in xrange(fsamps):
				i = random.randint(0, fmat.shape[1]-window)
				result.append(fmat[:, i:i+window])
			fmat = np.concatenate(result, axis=0)
			inMatrix.append(fmat)
			tmat = np.tile(f[1], (fmat.shape[0], 1))
			targMatrix.append(tmat)
	
	inMatrix = np.concatenate(inMatrix, axis=0)
	targMatrix = np.concatenate(targMatrix, axis=0)
	
	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p])
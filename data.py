import csv, random
import numpy as np

# File I/O
def loadfile(fname):
	bits = 8
	try:
		fmat = np.fromfile(fname, dtype=np.dtype('uint%s' % bits))
		result = np.zeros((fmat.size, 2**bits))
		result[np.arange(fmat.size), fmat] = 1
		return np.expand_dims(result, axis=0)
	except IOError:
		return np.empty((0,0,0))

def getfiles():
	with open('classes.csv', 'rb') as csvfile:
		flist = list(csv.reader(csvfile, delimiter=',', quotechar='"'))[1:]
	random.shuffle(flist)
	return [(x[0], np.array(x[1:], dtype=float)) for x in flist]

def readfiles():

	fsamps = 100			# Samples per file
	window = 160
	
	inMatrix, targMatrix = [], []				# Extract training data
	for f in getfiles():
		print f[0]
		fmat = loadfile(f[0])					# Load file
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
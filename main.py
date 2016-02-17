import csv, random
import numpy as np

import lstm

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
	
# Data preparation
def sample(mat, w=100, n=5):
	assert mat.shape[1] > w
	result = []
	for x in xrange(n):
		i = random.randint(0, mat.shape[1]-w)
		result.append(mat[:, i:i+w])
	return np.concatenate(result, axis=0)

def getfiles():
	with open('classes.csv', 'rb') as csvfile:
		flist = list(csv.reader(csvfile, delimiter=',', quotechar='"'))[1:]
	random.shuffle(flist)
	return [(x[0], np.array(x[1:], dtype=float)) for x in flist]

def readfiles():

	fsamps = 5			# Samples per file
	nsamps = 1000		# Samples in total
	window = 100
	
	inMatrix, targMatrix = [], []				# Extract training data
	for f in getfiles()[:nsamps//fsamps]:
		print f[0]
		fmat = loadfile(f[0])					# Load file
		if fmat.shape[1] > window:
			fmat = sample(fmat, window, fsamps)	# Sample
			inMatrix.append(fmat)
			
			tmat = np.tile(f[1], (fmat.shape[0], fmat.shape[1], 1))
			targMatrix.append(tmat)
	
	# Turn lists into matrices (online concatenation is slow)
	inMatrix = np.concatenate(inMatrix, axis=0)
	targMatrix = np.concatenate(targMatrix, axis=0)
	
	# Shuffle matrices
	p = np.random.permutation(inMatrix.shape[0])
	return (inMatrix[p], targMatrix[p])

#Main
def main():

	# Get model, either through training or loading
	if True:
		inMatrix, targMatrix = readfiles()
		model = lstm.trainModel(inMatrix, targMatrix)
	else:
		model = lstm.loadModel('')
		
	# Backtest model
	for f in getfiles():
		print 'Opening ' + f[0] + '...'
		fmat = loadfile(f[0])
		result = lstm.runModel(fmat, model)
		print np.mean(result, axis=1)
		print np.max(result, axis=1)
		print np.min(result, axis=1)
		print

if __name__ == "__main__":
	main()
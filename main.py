import csv, random
import numpy as np

import modeler
import data

#Main
def main():

	#input = np.concatenate((np.ones((1000, 100, 256)),np.zeros((1000, 100, 256))), axis=0)
	#target = np.concatenate((np.ones((1000, 1)),np.zeros((1000, 1))), axis=0)
	input, target = data.readfiles()
	print input.shape
	print target.shape
	model = modeler.train(input, target)
	#model = modeler.load('1456868768')
	for i in xrange(input.shape[0]):
		print 'Target: ' + str(target[i])
		result = modeler.run(np.expand_dims(input[i], axis=0), model)
		print result

if __name__ == "__main__":
	main()
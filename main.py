import csv, random
import numpy as np

import modeler
import data

def main():

	input, target = data.readfiles()
	print input.shape
	print target.shape
	model = modeler.train(input, target)
	test = raw_input('Enter text (enter to cancel): ')
	while test:
		m = data.readstr(test)
		result = modeler.run(m, model)
		print result
		test = raw_input('Enter text (enter to cancel): ')

if __name__ == "__main__":
	main()
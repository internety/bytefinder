import csv, random
import numpy as np

import modeler
import data

def main():

	input, target, classes = data.readfiles()
	model = modeler.train(input, target)

	test = raw_input('\nEnter text (enter to cancel): ')
	while test:
		m = data.readstr(test)
		result = modeler.run(m, model)
		for i in xrange(len(classes)):
			print '%s:\t%s%%' % (classes[i], round(result[0,i]*100,2))
		test = raw_input('\nEnter text (enter to cancel): ')

if __name__ == "__main__":
	main()
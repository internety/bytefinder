#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random

# Local Libraries
import modeler
import data

###############################################################################

def main():
	input, target = data.sample('data')
	model = modeler.train(input, target)

	test = raw_input('\nInput text (enter to cancel): ')
	while test:
		m = data.makeMatrix([test])
		result = modeler.run(m, model)
		print(result)
		test = raw_input('\nInput text (enter to cancel): ')

if __name__ == "__main__":
	main()
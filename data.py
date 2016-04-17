#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import random, os, re
random.seed(1)

# Third-party Libraries
import numpy as np
np.random.seed(1)

###############################################################################

def backtest(classes, input, output):
	for sequence in xrange(input.shape[0]):
		print(output[sequence].shape)
		name = random.randint(100000,999999)
		cats = '\n'.join(["<input type='radio' name='cat_select' onclick='myFunction()'>"+x for x in classes])
		doc = """<!doctype html>
		<html>
		<head>
		<title>
		</title>
		</head>
		<body>
		<form>
		%s
		</form>
		<script type="text/javascript" src="%s_meta.js">
		</script>
		<script language="javascript" type="text/javascript">
		function myFunction(){var form=document.forms[0];for(i=0;i<form.length;i++){if(form[i].checked){var cat=i;}}var x=document.getElementById("text");var text=x.textContent||x.innerText;var newText='';for(var a=0;a<text.length;a++){var letter=text.charAt(a);if(a>
		0 && a<=colors.length){color=Math.round(Math.min(Math.max(colors[a-1][cat],0.0),1.0)*255);hex=color.toString(16).toUpperCase().repeat(3);}else{hex='000000'}newText+=letter.fontcolor(hex);}x.innerHTML=newText;}</script>
		<div id="text">
		%s</div>
		</body>
		</html>""" % (cats, name, mat2str(input[sequence]))
		form = '['+','.join(['%0.2f' for x in xrange(output.shape[-1])])+'],'
		last_elem = '['+','.join(['0.0' for x in xrange(output.shape[-1])])+']'
		np.savetxt('results/%s_meta.js' % name, output[sequence],fmt=form,delimiter=',',header='var colors=[',footer=last_elem+']',comments='')
		with open('results/%s.html' % name, 'w+') as f:
			f.write(doc)

# Given a file string 's',
# sample and output a numpy matrix with shape (1, len(s), 256)
def str2mat(s):
	smat = np.fromstring(s, dtype=np.dtype('uint8'))
	result = np.zeros((1, smat.size, 256))
	result[0, np.arange(smat.size), smat] = 1
	return result

def mat2str(smat):
	return (np.where(smat)[-1]).tostring().replace('\x00','')

def clean(dname):

	print('Cleaning %s...' % dname)
	for root, dirs, files in os.walk(dname):
		for file in files:
			fname = root+'/'+file
			with open(fname, 'r') as f:
				text = re.sub('\s+', ' ', f.read())
			with open(fname, 'w') as f:
				f.write(text)

# Sample a directory and all subdirectories
def sample(dname, window=500, size=4000):

	print('Sampling...')
	ncat = {dname:size}  # Samples per category, based on directory tree
	nfile = []           # Samples per file, based on relative filesize
	classes = []         # Named classes, based on folder names
	for root, dirs, files in os.walk(dname):
		for d in dirs:
			ncat[root+'/'+d] = ncat[root]/len(dirs)
		if '/' in root:
			classes.append(root.split('/')[-1])

		# Remove hidden and overly small files
		files = filter(lambda x: not x.startswith('.'), files)
		files = filter(lambda x: os.path.getsize(root+'/'+x)>window, files)

		# Calculate size of category
		catsize = sum([os.path.getsize(root+'/'+file) for file in files])
		for file in files:
			fpath = root+'/'+file
			ntimes = int(round(ncat[root]*os.path.getsize(fpath)/float(catsize)))
			if ntimes:
				nfile.append((fpath, ntimes))

	nsamples = sum(x[1] for x in nfile)
	inMatrix = np.empty((nsamples, window, 256), dtype=np.dtype('float32'))
	targMatrix = np.empty((nsamples, window, len(classes)), dtype=np.dtype('float32'))
	
	i = 0
	for file in nfile:
		target = np.tile([1 if x in file[0].split('/') else 0 for x in classes], (1, window, 1))
		for _ in xrange(file[1]):
			if not i % 1000:
				print('%0.2f%% Done' % (100.0*i/nsamples))
			with open(file[0]) as f:
				f.seek(random.randint(0, os.path.getsize(file[0])-window))
				inMatrix[i] = str2mat(f.read(window))
				targMatrix[i] = target
				i += 1

 	return (inMatrix, targMatrix, classes)
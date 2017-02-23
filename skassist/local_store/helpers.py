# -*- coding: utf-8 -*-
from os import makedirs
from os.path import isdir
import pickle
import bz2 # bz2.BZ2File
import gzip # gzip.GzipFile

# DONE: [OPT] Integrate compression with backwards compatibility.

# _______________________________________________________________________Helpers
def saveToFile(obj, path):
    with gzip.GzipFile(path, 'wb') as file:
        pickle.dump(obj, file)

def loadFromFile(path):
	try:
	    with gzip.GzipFile(path, 'rb') as file:
	        return pickle.load(file)
	except OSError:
		with open(path, 'rb') as file:
			return pickle.load(file)

def makeDirectoryPath(path):
	"""Create directory at :obj:`path` and create intermediate directories as 
	required. (`tzot@stackoverflow <http://stackoverflow.com/a/600612/5304427>`_)

	"""
	try:
		makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and isdir(path):
			pass
		else:
			raise
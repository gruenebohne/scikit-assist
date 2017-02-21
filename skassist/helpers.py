# -*- coding: utf-8 -*-

import pickle
import bz2 # bz2.BZ2File
import gzip # gzip.GzipFile

# TODO: [OPT] Integrate compression with backwards compatibility.

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

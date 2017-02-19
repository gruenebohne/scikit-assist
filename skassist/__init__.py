# -*- coding: utf-8 -*-
"""scikit-assistant is a module that supports the management and execution of data 
science experiments. Every experiment is defined by a dataset and a validation 
strategy. A dataset must be provided by a pandas DataFrame. The validation strategy 
is specified by a list of lists, each containing up to three indices into the 
dataset (training, testing and validation set). 

Optionally, a list of columns that are to be used as features during model 
training can be provided. If not provided, all columns but the target will be
used as features.

Example:
	A new experiment with a DataFrame ``data`` can be specified as follows::
		
		from scikit-assistant import Library

		cross_val = [
			[tr1,te1,va1],	# 1. validation split
			[tr2,te2,va2],	# 2. validation split
			[tr3,te3,va3]	# 3. validation split
		]

		features = ['col1', 'col2', 'col3']

		lib = Library()

		lib.add('name_of_experiment', data, cross_val, features, 
			'Some meaningful desription.') 

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

.. todo::
	* automatic reload on file change
	* automatic result_computation on prediction change
	* transparent file compression
	* DBMS integration (MongoDB, MySQL)

"""

# expose the library class
from .library import Library
from .experiment import Experiment
from .model import Model
from .files import LocalFiles
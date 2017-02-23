# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris


def load_iris_data():
	data = load_iris()

	features = ['f1','f2','f3','f4']
	target = 'target'

	data = pd.DataFrame(
		np.hstack((
			data['data'], np.transpose([data['target']])
		)), 
		columns=features+target
	)

	kf = KFold(n_splits=10, shuffle=True)
	splits = [(tr, te) for tr, te in kf.split(data)]

	return data, splits, features, target
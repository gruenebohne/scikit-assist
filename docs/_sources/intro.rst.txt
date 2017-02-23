.. scikit-assist documentation master file, created by Tillmann Radmer

===================================================================================
`scikit-assist: a data science manager <https://radmerti.github.io/scikit-assist>`_
===================================================================================

scikit-assist is project for assisting data science experiments. It is manly written for scikit-learn, but can be used with other frameworks like TensorFlow. The main module is a library that keeps and manages the storage of all experiments. This way no experiment will ever get lost and is easily accessible through Python.

The storage is done in files on disk. Ideally, this would be replaced by a storage API that is flexible enough to allow different database targets. The advantage of using disk storage is that while the API is not complete changes to the library, liek copying or deleting files, can be done through the command line or file browser.

For more information, documentation and examples please head over to the  `project page <https://radmerti.github.io/scikit-assist>`_.

.. note::
	The module has no installer and is not PyPI compatible yet. So the easiest way
	to get going is to download or clone this repository and copy the `skassist`
	folder into your project folder.

Example
=======
Creating a new experiment based on the scikit-learn iris dataset, adding several models, evaluating them and calculating results is easy::
	
	from skassist import Library
	from skassist.examples import load_iris_data
	from sklearn.ensemble import RandomForestClassifier
	
	# Create a new experiment with the iris dataset
	data, splits, features, target = load_iris_data()
	lib = Library()
	lib.add('iris_example', data, splits, features, 'Iris data example experiment.')

	# Our model class
	class SimpleClassificationModel():
		def __init__(self, estimator):
			self.estimator = estimator

			# set mandatory variables used by SLIB
			self.name = self.__class__.__name__
			self.target = 'target'
			self.extra_features = []
			self.params = vars(estimator)
			self.params['classifier_name'] = estimator.__class__.__name__

		def fit(self, X, y):
			self.estimator.fit(X,y)

		def predict_proba(self, X):
			return self.estimator.predict_proba(X)

	# We want to know how many trees we need
	for n_trees in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
		rf = RandomForestClassifier(n_estimators=n_trees)
		lib.experiments[0].add(SimpleClassificationModel(rf))

	# run cross-validation
	lib.lib.experiments[0].eval()
.. scikit-assist documentation master file, created by Tillmann Radmer

===================================================================================
`scikit-assist: a data science manager <https://radmerti.github.io/scikit-assist>`_
===================================================================================

scikit-assist is project for assisting data science experiments. It is manly written for scikit-learn, but can be used with other frameworks like TensorFlow. The main module is a library that keeps and manages the storage of all experiments. This way no experiment will ever get lost and is easily accessible through Python.

The storage is done in files on disk. Ideally, this would be replaced by a storage API that is flexible enough to allow different database targets. The advantage of using disk storage is that while the API is not complete changes to the library, liek copying or deleting files, can be done through the command line or file browser.

For more information please head over to the project page `scikit-assist <https://radmerti.github.io/scikit-assist>`_.

from .files import LocalFiles
from .experiment import Experiment

# from concurrent.futures  ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from os import makedirs, listdir # remove, rmdir,
from os.path import join, isdir, exists, expanduser # isfile

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# data preparation
# from sklearn.cross_validation import StratifiedKFold


# _______________________________________________________________________Library
class Library(LocalFiles):
    """Manages the root folder of the library.

    This class supports the creation and deletion of new experiments

    "Properties created with the ``@property`` decorator should be documented
    in the property's getter method."

    Attributes:
        experiments (list): A list of experiments found in the library folder.
        path (str): Path to the root directory.

    """

    # __________________________________________________________________________
    def __init__(self, lib_folder=join(expanduser("~"),'Datasets','SLIB')):
        """Library constructor.

        The __init__ method tries to read the content of the library folder. When
        looking for experiments it only considers folder with a name string 
        starting in 'exp_'. Everything else is ignored.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            lib_folder (:obj:`str`, optional): Absolute/relative path to the root directory.

        """

        LocalFiles.__init__(self, lib_folder)

        if not exists(self.path):
            makedirs(self.path)

        # load experiments in the library folder
        self.experiments = []

        # only consider folders with matching name
        onlyfolders = [f for f in listdir(self.path) if isdir(join(self.path, f))]
        onlyexperiments = sorted([f for f in onlyfolders if 'exp_' in f])

        # Append Experiment objects to the list. The Experiment object only
        # loads the meta information on creation. Data and Model files are
        # loaded on-demand.
        for fname in onlyexperiments:
            self.experiments.append(
                Experiment(join(self.path, fname))
            )

    # __________________________________________________________________________
    # create new experiment
    def add(self, name, df, skf, features, description=''):
        self.experiments.append(
            Experiment.New(
                name,
                df,
                skf,
                features,
                self.path,
                description
            )
        )

    # __________________________________________________________________________
    # deletes the experiment at the given index from memory and permanent storage
    def delete(self, index):
        # delete the experiment folder on disk
        self.experiments[index].delete()
        # remove the experminet from the maintained list
        del self.experiments[index]


    # __________________________________________________________________________
    def __repr__(self):
        return "<Library path: {0}, experiments:{1}>".format(
            self.path, 
            self.experiments
        )

    # __________________________________________________________________________
    def __str__(self):
        return "<Library path: {0}, experiments:{1}>".format(
            self.path, 
            self.experiments
        )

# -*- coding: utf-8 -*-

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

    The constructor method tries to read the content of the library folder. When
    looking for experiments it only considers folder with a name string starting 
    in `exp_`. Everything else is ignored.

    Args:
        lib_folder (:obj:`str`, optional): Absolute/relative path to the root directory.

    Attributes:
        experiments (:obj:`list`): A list of experiments found in the library folder.

        path (:obj:`str`): Path to the root directory.

    .. todo::
        * Implement a :obj:`search` function for finding experiments (by name, date, dataset size, etc.)

    """

    # __________________________________________________________________________
    def __init__(self, lib_folder=join('.','library')):
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
    def add(self, name, df, skf, features, description=''):
        """Adds an experiment based on the dataset `df` to the library. A folder
        `'exp_'+name+timestamp` is created for the experiment in the library
        folder. It will hold the dataset and cross-validation masks, as well as
        sub-folders for each model.
        
        Args:
            name (:obj:`str`): 
                A name for the experiment. Will be used together with the 
                timestamp for storing the experiment.

            df (:obj:`pandas.DataFrame`): 
                The dataset as a Pandas DataFrame.

            skf (:obj:`numpy.ndarray`): 
                An array of indices, each being one cross-validation split.

            features (:obj:`list`): 
                A list of column names that are to be used as features during 
                training.

            description (:obj:`str`): 
                A descriptive string of the dataset, experiment or changes to 
                make finding stuff later easier.

        """
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
    def delete(self, index):
        """Deletes the experiment at the given index from memory and permanent 
        storage.

        Args:
            index (:obj:`int`): 
                Index of the experiment to delete. The index is sorted by name
                and timestamp.

        """
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

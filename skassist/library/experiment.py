# -*- coding: utf-8 -*-
from ..local_store import LocalFiles
from .model import Model
# from .helpers import saveToFile

from os import listdir, makedirs
from os.path import join, isdir
from datetime import datetime
from os.path import exists

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np



# ___________________________________________________________________ Experiment
class Experiment(LocalFiles):
    """Manages the root folder of an experiment.

    An experiments folder manages the dataset and cross-validation splits
    associated with it. Evaluation and result calcualtion can be initiated
    for all models with the evaluate()

    Attributes:
        experiments (:obj:`list`): A list of experiments found in the library folder.

        path (:obj:`str`): Path to the root directory.

    .. todo::
        * Implement :obj:`LibEstimator` base class that all models must inherit from. The base class ensures that the needed properties are implemented.
        * Offer optional arguments in :func:`Experiment.add` for the LibEstimator properties when a user doesn't want to inherit from the base class!?
    
    """

    # __________________________________________________________________________
    def __init__(self, experiment_folder):
        LocalFiles.__init__(self, experiment_folder)

        # load meta information
        self.meta = self.load('meta')

        # get only folders that contain 'model_'
        onlyfolders = self.list_folders()
        modelfolders = sorted([f for f in onlyfolders if 'model_' in f])

        # load the model dictionaries into an array
        self.models = []
        for mfolder in modelfolders:
            self.models.append(
                Model(join(self.path, mfolder))
            )

        self.meta['num_models'] = len(self.models)


    # __________________________________________________________________________
    @classmethod
    def New(cls, name, df, skf, features, lib_path, description=''):
        """Factory method for creating a new Experiment instance given a name,
        dataset, cross-validation mask and a list of features. The path to the
        library in which the experiment is created must be given.

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

            lib_path (:obj:`str`): 
                Path to the library in which the experiment is created.

            description (:obj:`str`): 
                A descriptive string of the dataset, experiment or changes to 
                make finding stuff later easier.

        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        path = join(lib_path, 'exp_{0}_{1}'.format(name, timestamp))

        # TODO: [REM] Not needed anymore as the functionality is now in LocalFiles.
        # create the directory if it doesn't exist
        # if not exists(path):
        #     makedirs(path)

        experiment_class = cls(path)
        experiment_class.save_new(df, 'data')
        experiment_class.save_new(skf, 'skf')

        # TODO: [REM] Not needed anymore as LocalFile.save_new is used instead now.
        # write the data and cross-validatioin object to the new directory
        # saveToFile(df, join(path, 'data'))
        # saveToFile(skf, join(path, 'skf'))

        # write meta information
        meta = {
            'created': timestamp,
            'name': name,
            'num_models': 0,
            'features': features,
            'description': description
        }
        experiment_class.save_new(meta, 'meta')
        # TODO: [REM] Not needed anymore as LocalFile.save_new is used instead now.
        # saveToFile(meta, join(path, 'meta'))

        # return class instance
        return experiment_class


    # __________________________________________________________________________
    def add(self, estimator):
        """Adds a model to the experiment. 
        
        Args:
            estimator (:class:`~skassist.base.BaseEstimator`): 
                A name for the experiment. Will be used together with the 
                timestamp for storing the experiment.

        """
        features = self.meta['features']
        if type(estimator.extra_features) is list:
            all_features = features + estimator.extra_features
        elif estimator.extra_features is not None:
            print('ERROR: extra_features must be of type list.')
            return
        else:
            all_features = features

        self.models.append(
            Model.New(
                estimator,
                estimator.name,
                self.meta['name'],
                estimator.target,
                all_features,
                self.path,
                estimator.params
            )
        )

        # update the meta information and save to disk
        self.meta['num_models'] += 1
        self.save('meta')


    # __________________________________________________________________________
    # deletes the model at the given index from memory and permanent storage
    def delete(self, index):
        # delete the model folder on disk
        self.models[index].delete()
        # remove the model from the maintained list
        del self.models[index]

        # update the meta information and save to disk
        self.meta['num_models'] -= 1
        self.save('meta')


    # __________________________________________________________________________
    # delete all models in this experiment
    def delete_all(self, index):
        for i, m in enumerate(self.models):
            self.delete(i)


    # __________________________________________________________________________
    def findone(self, boolean_func):
        """Return the first model matching :func:`~doc_definitions.boolean_func`.
        
        Args:
            boolean_func (:func:`~doc_definitions.boolean_func`):
                A function that takes an :class:`~skassist.Model` and 
                returns a boolean indicating a match.

        """
        return next(self.find(boolean_func))


    # __________________________________________________________________________
    def find(self, boolean_func):
        """Iterator function, yielding all models matching :func:`~definitions.boolean_func`.
        
        Args:
            boolean_func (:func:`~definitions.boolean_func`):
                A function that takes an :class:`~skassist.Experiment` and 
                returns a boolean indicating a match.
        
        """
        for model in self.models:
            if boolean_func(model):
                yield model


    # __________________________________________________________________________
    # fit, then evaluate all all models for any cross-validation split for
    # which no prediction is present
    def evaluate(self, splits_first=False, max_workers=1, verbose=1, te_split_idx=1):
        # load the dataset
        data = self.load('data')
        skf = self.load('skf')

        print('tr: {0}\nte: {1}\nall: {2}'.format(
            len(np.concatenate(skf[0][:te_split_idx])),
            len(skf[0][te_split_idx]),
            len(data)
        ))

        len_models = len(self.models)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_scores = []
            len_models = len(self.models)
            for i, model in enumerate(self.models):
                future_to_scores.append(
                    executor.submit(
                        model.evaluate, data, skf,
                        split_list=None, verbose=0, te_split_idx=te_split_idx
                    )
                )
            
            for k, future in enumerate(as_completed(future_to_scores)):
                try:
                    ret_model = future.result()
                except Exception as exc:
                    print('ERROR:{0}'.format(exc))
                else:
                    if verbose > 0:
                        print('{0:2}/{1:<2} {2}'.format(k+1, len_models,
                              ret_model.meta['name']), end='\n')

        # free RAM by dropping the dataset
        data = None
        skf = None
        self.drop('data')
        self.drop('skf')


    # __________________________________________________________________________
    # calculate result for all models in this experiment
    def calc_result(self, scoring_function, name, max_workers=1, verbose=1, te_split_idx=1):
        # Load the dataset from disk
        data = self.load('data')
        skf = self.load('skf')


        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_scores = []
            len_models = len(self.models)
            for i, model in enumerate(self.models):
                future_to_scores.append(
                    executor.submit(
                        model.calc_results, scoring_function, name, data, skf,
                        verbose=0, te_split_idx=te_split_idx
                    )
                )

            for j, future in enumerate(as_completed(future_to_scores)):
                try:
                    ret_model = future.result()
                except Exception as exc:
                    print('ERROR:{0}'.format(exc))
                else:
                    if verbose > 0:
                        print('{0:>3}/{1:<3} {2}'.format(j+1, len_models,
                              ret_model.meta['name']), end='\n')


        # # Invoke results update for all models
        # for j,model in enumerate(self.models):
        #     if verbose > 0:
        #         print('{0:2}/{1:<2} {2}'.format(j+1, len(self.models),
        #             model.meta['name']), end='\n')
        #     model.calc_results(scoring_function, name, data, skf, 
        #                        verbose=verbose,
        #                        te_split_idx=te_split_idx)

        # drop dataset
        self.drop('data')
        self.drop('skf')


   # ___________________________________________________________________________
    # Reset results with given name
    def reset_results(self, name, te_split_idx=1):
        for model in self.models:
            model.reset_results(name, te_split_idx=te_split_idx)
        # end for self.models


   # ___________________________________________________________________________
    # Reset predictions
    def reset_predictions(self):
        for model in self.models:
            model.reset_predictions()
        # end for self.models


    # __________________________________________________________________________
    def __repr__(self):
        return "<Experiment path: {0}, meta: {1}, models: {2}>".format(
            self.path, 
            self.meta, 
            self.models
        )

    def __str__(self):
        return "<Experiment path: {0}, meta: {1}, models: {2}>".format(
            self.path, 
            self.meta, 
            self.models
        )


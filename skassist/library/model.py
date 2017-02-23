# -*- coding: utf-8 -*-

from ..local_store import LocalFiles
# from .helpers import saveToFile

from datetime import datetime, timedelta
# from os import makedirs
from os.path import join, exists
from time import clock
import numpy as np



# _________________________________________________________________________Model
class Model(LocalFiles):
    """Manages the root folder of a model.

    The model class handles the model file as well as the predictions and results
    that are stored in seperate files. The evaluate() function runs the model for
    each cross-valudation split and makes and saves the predictions.

    The calc_results() can then be used to calculate a evaluation metric based on
    the predictions. The evaluation metric is defined by a function that is
    passed into calc_results().

    Attributes:
        meta (:obj:`dict`): A dictionary holding meta information about the model.
        
        path (:obj:`str`): Path to the root directory.

    """

    # __________________________________________________________________________
    def __init__(self, model_path):
        LocalFiles.__init__(self, model_path)

        # Load the meta file
        self.meta = self.load('meta')


    # __________________________________________________________________________
    # factory method to load a Model instance from a folder
    @classmethod
    def New(cls, estimator, name, experiment, target, features,
                 experiment_path, modelParams={}):
        """Factory method for creating a new Model instance. The model is created
        in its sub-folder inside the :obj:`experiment_path` folder.

        Args:
            estimator (:obj:`str`): 
                A name for the experiment. Will be used together with the 
                timestamp for storing the experiment.

            name (:obj:`str`): 
                A name for the experiment. Will be used together with the 
                timestamp for storing the experiment.

            experiment (:obj:`str`): 
                Name of the experiment. Useful if the model folder is lost & found.

            target (:obj:`str`): 
                Name of the target variable. Must be a columns in the dataset.

            features (:obj:`list`): 
                A list of column names that are to be used as features during 
                training.

            experiment_path (:obj:`str`): 
                Path to the library in which the experiment is created.

            modelParams (:obj:`dict`, optional): 
                A dictionary with tunable model parameters.

        """
        # create the meta information
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # save the path to this model
        path = join(experiment_path, 'model_{0}'.format(timestamp))

        # Adjust path second for second if a folder with the same
        # timestamp already exists.
        while exists(path):
            timestamp = (datetime.strptime(timestamp, '%Y%m%d%H%M%S')
                      + timedelta(seconds=1)).strftime('%Y%m%d%H%M%S')
            path = join(experiment_path, 'model_{0}'.format(timestamp))

        # TODO: [REM] Not needed anymore as the functionality is now in LocalFiles.
        # Now, create the directory. It shouldn't exist.
        # if not exists(path):
        #     makedirs(path)

        model_class = cls(path)
        model_class.save_new(estimator, 'estimator')

        meta = {
            'ID': timestamp,
            'created': timestamp,
            'experiment': experiment,
            'modified': timestamp,
            'name': name,
            'target': target,
            'features': features,
            'mParams': modelParams
        }
        model_class.save_new(meta, 'meta')

        # TODO: [REM] Not needed anymore as LocalFile.save_new is used instead now.
        # save to three different files for leaner updates
        # saveToFile(estimator, join(path, 'estimator'))
        # saveToFile(meta, join(path, 'meta'))

        return model_class


    # __________________________________________________________________________
    # 
    def reset_results(self, name, verbose=1, te_split_idx=1):
        """Reset results with given name.
        
        Args:
            name (:obj:`str`): Name of the result that should be deleted.

            verbose (:obj:`int`, optional): Level of output. 0 is no output.

            te_split_idx (:obj:`int`, optional): 
                Index of evaluation split that should be resetted.

        """
        name = name+'_{0}'.format(te_split_idx)
        ready_dict_name = 'ready_mask_{0}'.format(te_split_idx)
        if verbose > 0: 
            print('reset_results({0}):'.format(name))
        results = self.load('results')
        if results is not None:
            if name in results:
                results = results[name]
                for i in range(len(results[ready_dict_name])):
                    results[ready_dict_name][i] = False
                # end for results['ready_mask']
                for key, value in results.items():
                    if key != ready_dict_name:
                        results[key] = None
            # end if name in results
        # end if not None
        self.done('results')

    # __________________________________________________________________________
    def reset_predictions(self):
        """Resets all predictions for all test/validation splits.

        """
        predictions = self.load('predictions')
        if predictions is not None:
            for i in range(len(predictions['ready_mask'])):
                predictions['ready_mask'][i] = False
            # end for results['ready_mask']
            for key, value in predictions.items():
                if key != 'ready_mask':
                    predictions[key] = None
            # end if name in results
        # end if not None
        self.done('predictions')


    # __________________________________________________________________________
    def fit(self, df):
        """Fits the model on the whole dataset and save it.

        Args:
            df (:obj:`pandas.DataFrame`): 
                The DataFrame on which to train the model. Must contain all
                feature, "extra" feature and target columns that the model
                requires.

        """
        # load the estimator from disk
        estimator = self.load('estimator')

        # train the estimator on the whole dataset
        estimator.fit(
            df.loc[:, self.meta['features']],
            df.loc[:, self.meta['target']]
        )

        # save the mode to disk
        self.done('estimator')


    # __________________________________________________________________________
    def evaluate(self, df, skf, split_list=None, verbose=1, te_split_idx=1):
        """Cross-evaluates the model on the test datasets given by `te_split_idx`.
        `te_split_idx` indexes the split number to use for testing. All splits
        *bellow* the test index are used for training.
        
        .. note::
            The cross-validation indices in `skf` must index into the DataFrame `df`.

        Args:
            df (:obj:`pandas.DataFrame`): 
                The DataFrame on which to evaluate the model. Must contain all
                feature, "extra" feature and target columns that the model
                requires.

            skf (:obj:`numpy.ndarray`): 
                An array containing arrays of splits. E.g. an array with 10 arrays,
                each containing 3 splits for a 10-fold cross-validation with
                training, test and validation set.

            split_list (:obj:`list`): 
                A list of split indices to use for evaluation. This is usefull
                when computation time is a limiting factor and a reduced 
                evaluation for model selection is sufficient.

            verbose (:obj:`int`): Level of print output. 0 is no output.

            te_split_idx (:obj:`int`): 
                Index of split that the model is evaluated on.

        """
        print_ending = ''

        p_dict_name = 'probabilities_{0}'.format(te_split_idx)
        te_time_dict_name = 'test_times_{0}'.format(te_split_idx)
        tr_time_dict_name = 'train_times_{0}'.format(te_split_idx)
        ready_dict_name = 'ready_mask_{0}'.format(te_split_idx)

        if type(split_list)!=list:
            split_list = [i for i in range(len(skf))]
            print_ending = '\n'

        # load the estimator and results file from disk
        estimator = self.load('estimator')
        predictions = self.load('predictions')

        # Initialize predictions file if not done before.
        if predictions == None:
            self.local['predictions'] = {}
            predictions = self.local['predictions']

        if not p_dict_name in predictions \
                or predictions[p_dict_name] is None:
            predictions[p_dict_name] = [
                np.zeros(len(split[1]), dtype=np.float16) for split in skf
            ]
        if not te_time_dict_name in predictions \
                or predictions[te_time_dict_name] is None:
            predictions[te_time_dict_name] = np.zeros(len(skf), dtype=np.float32)
        if not tr_time_dict_name in predictions \
                or predictions[tr_time_dict_name] is None:
            predictions[tr_time_dict_name] = np.zeros(len(skf), dtype=np.float32)

        # create the mask indicating result readyness if not existant
        if not ready_dict_name in predictions or predictions[ready_dict_name] is None:
            predictions[ready_dict_name] = np.zeros(len(skf), dtype=np.bool)

        for i, split in enumerate(skf):
            
            if verbose > 0:
                if i == 0:
                    print('|', end='')

            if i in split_list:
                if verbose > 0:
                    print('{0:2}/{1:<2}:'.format(i+1, len(skf)), end='')

                # start the clock
                all_clock = clock()

                # only calculate anything if the fold is not yet ready as
                # inidcated by the ready_mask
                if not predictions[ready_dict_name][i]:
                    # create training and test split indices
                    tr_split = np.concatenate(split[:te_split_idx])
                    te_split = split[te_split_idx]

                    # start the clock
                    clock_start = clock()

                    # train the model
                    estimator.fit(
                        df.loc[
                            tr_split,
                            self.meta['features']
                        ],
                        df.loc[
                            tr_split,
                            self.meta['target']
                        ]
                    )

                    # save the training time
                    predictions[tr_time_dict_name][i] = clock() - clock_start

                    # reset the clock
                    clock_start = clock()

                    # make predictions for the test split and save locally
                    local_prediction = estimator.predict_proba(
                        df.loc[
                            te_split,
                            self.meta['features']
                        ]
                    )

                    # only one prediction for regression tasks
                    if len(np.shape(local_prediction))==1:
                        predictions[p_dict_name][i] = local_prediction
                    # two predictions for binary outcomes, take the second one
                    # e.g. "1", e.g. the positive outcome
                    elif len(np.shape(local_prediction))==2:
                        predictions[p_dict_name][i] = local_prediction[:,1]

                    # save the prediction time
                    predictions[te_time_dict_name][i] = clock() - clock_start

                    if verbose > 0:
                        print('{0:4.0f}'.format(predictions[tr_time_dict_name][i]+
                            predictions[te_time_dict_name][i]), end='')

                    # Set the mask indicating that this result is ready
                    predictions[ready_dict_name][i] = True

                    # Save the results to disk after computing each split such
                    # that results are available as soon as possible
                    self.save('predictions')

                else:
                    if verbose > 0:
                        print('{0:4.0f}'.format(predictions[tr_time_dict_name][i]+
                            predictions[te_time_dict_name][i]), end='')
               
                # end if not predictions['ready_mask']

                if verbose > 0:
                    print('|', end='')
            # end if i in split_list
        # end for enumerate(skf)
        if verbose > 0:
            print('', end=print_ending)

        # Drop the estimator from RAM. Not saving to disk.
        self.drop('estimator')
        # Also drop the results from RAM. All results have been written to disk.
        self.done('predictions')

        return self

    def scoring_function(self, model, y_true, y_predicted_probability):
        """The scoring function takes a model, the true labels and the prediction
        and calculates one or more scores. These are returned in a dictionary which
        :func:`~skassist.Model.calc_results` uses to commit them to permanent storage.

        Args:
            scoring_function (:func:`function`): 
                A python function for calculating the results given the true labels
                and the predictions. See :func:`~skassist.Model.scoring_function`.

            skf (:obj:`numpy.ndarray`): 
                An array containing arrays of splits. E.g. an array with 10 arrays,
                each containing 3 splits for a 10-fold cross-validation with
                training, test and validation set.

            df (:obj:`pandas.DataFrame`): 
                The DataFrame on which to evaluate the model. Must contain all
                feature, "extra" feature and target columns that the model
                requires.

        """
        raise NotImplemented('This function is only for documentation. DO NOT CALL!')
        return None

    # __________________________________________________________________________
    def calc_results(self, scoring_function, name, df, skf, verbose = 1, te_split_idx=1):
        """Update or create the results series `name` using the provided 
        :func:`~skassist.Model.scoring_function`. 

        .. note::
            The cross-validation indices in `skf` must index into the DataFrame `df`.
        
        Args:
            scoring_function (:func:`function`): 
                A python function that calculates results given a model, its
                predictions and the true labels. See :func:`~skassist.Model.scoring_function`.

            skf (:obj:`numpy.ndarray`): 
                An array containing arrays of splits. E.g. an array with 10 arrays,
                each containing 3 splits for a 10-fold cross-validation with
                training, test and validation set.

            df (:obj:`pandas.DataFrame`): 
                The DataFrame on which to evaluate the model. Must contain all
                feature, "extra" feature and target columns that the model
                requires.

            skf (:obj:`numpy.ndarray`): 
                An array containing arrays of splits. E.g. an array with 10 arrays,
                each containing 3 splits for a 10-fold cross-validation with
                training, test and validation set.

            verbose (:obj:`int`): Level of print output. 0 is no output.

            te_split_idx (:obj:`int`): Index of split that the model is evaluated on.

        """
        # Load the predictions and results file
        results = self.load('results')
        predictions = self.load('predictions')

        p_dict_name = 'probabilities_{0}'.format(te_split_idx)
        name = name+'_{0}'.format(te_split_idx)
        ready_dict_name = 'ready_mask_{0}'.format(te_split_idx)

        state = ''

        if predictions is not None:
            probabilities = predictions[p_dict_name]

            # Create dictionary in dictionary of results
            if results is None:
                self.local['results'] = {}
                results = self.local['results']
                results[name] = {}
            elif not name in results:
                results[name] = {}
            results = results[name]

            # Create the mask indicating result readyness if not existant.
            if not ready_dict_name in results:
                results[ready_dict_name] = np.zeros(len(skf), dtype=np.bool)

            for k, split in enumerate(skf):
                # start the clock
                start_clock = clock()

                # Only calculate anything if the fold is not yet ready as
                # inidcated by the ready_mask
                if not results[ready_dict_name][k]:
                    if probabilities is not None:
                        # Only calculate results if the model was trained and 
                        # tested on this split before.
                        te_split = split[te_split_idx]

                        if np.not_equal(
                                probabilities[k], 
                                np.zeros(len(te_split))
                                ).any():

                            # Update values in the results dictionary. NOTE THAT 
                            # the values returned by the scoring function MUST be 
                            # normalized to the number of cross-validation folds.
                            for key, value in scoring_function(self,
                                                               df.loc[te_split,:],
                                                               probabilities[k]
                                                               ).items():
                                if key in results:
                                    if results[key] is None:
                                        results[key] = np.zeros(
                                            (len(skf),len(value))
                                        )
                                    results[key][k] = value
                                else:
                                    results[key] = np.zeros((len(skf),len(value)))
                                    results[key][k] = value

                            # Set the mask indicating that this result is ready
                            results[ready_dict_name][k] = True
                            state = 'C'
                        else:
                            results[ready_dict_name][k] = False
                            state = 'N'
                        # end if np.not_equal(...)
                    # end if probabilites is not None
                    else:
                        state = 'P'
                # end if not ready_mask[k]
                else:
                    state = 'D'

                if verbose > 0:
                    if k == 0:
                        print('|', end='')
                    print(
                        '{0}({1}):{2:2.0f}|'.format(
                            k,
                            state,  
                            clock()-start_clock), 
                        end=''
                    )

            # end for enumerate(skf)

            if verbose > 0:
                print('')
        # end predictions is not None
        else:
            if verbose > 0:
                print('None')

        # Save the results to disk, then free the RAM.
        self.done('results')
        self.drop('predictions')#

        return self


    # __________________________________________________________________________
    def __repr__(self):
        return "<Model path: {0}, meta:{1}>".format(self.path, self.meta)

    # __________________________________________________________________________
    def __str__(self):
        return "<Model path: {0}, meta:{1}>".format(self.path, self.meta)

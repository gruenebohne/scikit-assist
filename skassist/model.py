from .files import LocalFiles
from .helpers import saveToFile

from datetime import datetime, timedelta
from os import makedirs
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

    "Properties created with the ``@property`` decorator should be documented
    in the property's getter method."

    Attributes:
        meta (dict): A dictionary holding meta information about the model.
        path (str): Path to the root directory.

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

        # Now, create the directory. It shouldn't exist.
        if not exists(path):
            makedirs(path)

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

        # save to three different files for leaner updates
        saveToFile(estimator, join(path, 'estimator'))
        saveToFile(meta, join(path, 'meta'))

        return cls(path)


    # __________________________________________________________________________
    # Reset results with given name
    def reset_results(self, name, verbose=1, te_split_idx=1):
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
    # Reset predictions
    # TODO: add tr/te/va set (e.g. te_split_idx)
    def reset_predictions(self):
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
    # Fits the model on the whole dataset and save it.
    def fit(self, df):
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
    # Cross-validate the model.
    # TODO: [CODE] Rewrite
    def evaluate(self, df, skf, split_list=None, verbose=1, te_split_idx=1):
        
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


    # __________________________________________________________________________
    # update this model's results
    # TODO: [CODE] Rewrite
    def calc_results(self, scoring_function, name, df, skf, verbose = 1, te_split_idx=1):
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

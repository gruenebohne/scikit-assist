# -*- coding: utf-8 -*-

# ______________________________________________________________________________
def boolean_func(experiment):
    """Function that returns True when an experiment matches and False otherwise.
    
    Args:
        experiment (:class:`~skassist.Experiment`): Experiment that is to be tested.

    """

# ______________________________________________________________________________
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
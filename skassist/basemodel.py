# ____________________________________________________________________ BaseModel
class BaseModel():
    """The base class that every model used for :class:`skassist` must inherit
    from. It defines the interface through which the module trains the model
    and makes predictions. It also defines a few mandatory properties that must
    be set. Those are used by the module to identify the model and pass the
    correct feature and target columns to the model.

    Attributes:
    	name (:obj:`str`): A name for the model.  

    	extra_features (:obj:`list`): 
			A list of additional features that are needed by the model and 
			are not in the :obj:`Experiment.features` property of the 
			experiment. The extra features must be columns in the dataset.

		target (:obj:`str`): Name of the target variable.

		params (:obj:`list`): List of tunable parameters of the model.
    
    """
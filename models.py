import numpy as np
import tensorboard

print(tensorboard.__version__)


class JaiNN:
    """
    A class providing convenience functions for a neural netowrk model
    """
    def __init__(self, utils):
        """
        Initializes this JaiNN object
        :param utils: the utils object being used for this session
        """
        self.utils = utils
        self.model = utils.get_gru_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=None, epochs=None):
        """
        Sets up the model params and begins the training process
        :param data: the data to use for training
        :param method_to_call: the method to call to generate plots
        :param param_range: the range to use for tuning parameters
        :param param_factor: the parameter multiplier to use on the range
        :param epochs: the number of epochs to train for
        :return: the best val_loss model found during training
        """
        if method_to_call is None:
            raise IOError("Select a valid curve generating function")
        if param_range is None:
            param_range = [0, 100]
        args = {
            'data': data,
            'param_range': param_range,
            'param_factor': param_factor,
            'get_model': self.utils.get_gru_model,
            'metric': 'categorical_crossentropy',
            'epochs': epochs
        }
        self.model = method_to_call(**args)


class JaiLR:
    """
    A class offering convenience functions for a logistic regression model
    """
    def __init__(self, utils):
        """
        Initializes this JaiLR object with the given utils object.
        :param utils: the utils object used for this session
        """
        self.utils = utils
        self.model = utils.get_logistic_reg_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=0.001, epochs=300, **kwargs):
        """
        Sets up the model params and begins the training process
        :param data: the data to use for training
        :param method_to_call: the method to call to generate plots
        :param param_range: the range to use for tuning parameters
        :param param_factor: the parameter multiplier to use on the range
        :param epochs: the number of epochs to train for
        :param kwargs: args to be passed on to utils
        :return: the best val_loss model found during training
        """
        if method_to_call is None:
            raise IOError("Select a valid curve generating function")
        if param_range is None:
            param_range = [0, 100]
        args = {
            'data': data,
            'param_range': param_range,
            'param_factor': param_factor,
            'get_model': self.utils.get_logistic_reg_model,
            'metric': 'categorical_crossentropy',
            'epochs': epochs
        }
        self.model = method_to_call(**args, **kwargs)


class JaiSVM:
    """
    A class providing convenience functions when using a support vector machine model
    """
    def __init__(self, utils):
        """
        Initializes this JaiSVM object with the given utils object
        :param utils: the utils object used for this session
        """
        self.utils = utils
        self.model = utils.get_svm_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=0.001, epochs=1000, **kwargs):
        """
        Sets up the model params and begins the training process
        :param data: the data to use for training
        :param method_to_call: the method to call to generate plots
        :param param_range: the range to use for tuning parameters
        :param param_factor: the parameter multiplier to use on the range
        :param epochs: the number of epochs to train for
        :param kwargs: args to be passed on to utils
        :return: the best val_loss model found during training
        """
        if method_to_call is None:
            raise IOError("Select a valid curve generating function")
        if param_range is None:
            param_range = [0, 100]
        args = {
            'data': data,
            'param_range': param_range,
            'param_factor': param_factor,
            'get_model': self.utils.get_svm_model,
            'metric': 'categorical_hinge',
            'epochs': epochs
        }
        self.model = method_to_call(**args, **kwargs)

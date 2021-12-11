import numpy as np
import tensorboard

print(tensorboard.__version__)


class JaiNN:
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_gru_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=0.001, epochs=300):
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
            'is_gru': True,
            'epochs': epochs
        }
        self.model = method_to_call(**args)

    def predict(self, data):
        d = [np.array(data[0])[None, ...], np.array(data[1])[None, ...]]
        return self.model.predict_on_batch(d)


class JaiLR:
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_logistic_reg_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=0.001, epochs=300):
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
        self.model = method_to_call(**args)

    def predict(self, data):
        d = np.array(data[0])[None, ...]
        return self.model.predict_on_batch(d)


class JaiSVM:
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_svm_model()

    def prepare_and_run(self, data, method_to_call=None, param_range=None, param_factor=0.001, epochs=300):
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
        self.model = method_to_call(**args)

    def predict(self, data):
        d = np.array(data[0])[None, ...]
        return self.model.predict_on_batch(d)

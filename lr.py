import numpy as np


# Logistic regression model. Work in progress.
# To be moved to models.py?
class JaiLR:
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_logistic_reg_model()

    def l2_tuning_curve(self, data):
        self.model = self.utils.l2_tuning_curve(
            data=data,
            get_model=self.utils.get_logistic_reg_model,
            metric="categorical_crossentropy",
            param_range=(0, 1000, 12),
            param_factor=0.001
        )

    def learning_rate_tuning_curve(self, data):
        self.model = self.utils.learning_rate_tuning_curve(
            data=data,
            get_model=self.utils.get_logistic_reg_model,
            metric="categorical_crossentropy",
            param_range=(0, 400, 4),
            param_factor=0.001
        )

    def learning_curve(self, data):
        self.model = self.utils.learning_curve(
            data=data,
            get_model=self.utils.get_logistic_reg_model,
            metric="categorical_crossentropy",
            param_range=(3, len(data[1][0])),
        )

    def loss_over_epochs(self, data, epochs):
        self.model = self.utils.loss_over_epochs(
            data=data,
            get_model=self.utils.get_logistic_reg_model,
            metric="categorical_crossentropy",
            epochs=epochs
        )

    def predict(self, data):
        d = np.array(data[0])[None, ...]
        return self.model.predict_on_batch(d)

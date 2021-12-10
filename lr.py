from __future__ import absolute_import, division, print_function
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import tensorflow as tf

# Logistic regression model. Work in progress. To be moved to
# models.py

training_steps = 1000
batch_size = 256
display_step = 50


class JaiLR:
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_logistic_reg_model()

    def get_learning_curve_reg(self, data, labels, epochs):
        train_data, test_data = data
        train_labels, test_labels = labels
        classes = len(self.utils.get_vocabulary())
        for lr in range(1, 1000, 2):
            lr = lr * 0.001
            tf.random.set_seed(638)
            model = self.utils.get_logistic_reg_model()
            history = model.fit(
                train_data[0],
                to_categorical(train_labels, classes),
                epochs=epochs,
                validation_split=0.2
            )

    def learning_curve(self, data, labels, epochs):
        train_data, test_data = data
        train_labels, test_labels = labels
        classes = len(self.utils.get_vocabulary())
        history = self.model.fit(
            train_data[0],
            to_categorical(train_labels, classes),
            epochs=epochs,
            validation_split=0.2
        )
        self.eval(
            data[1][0],
            to_categorical(labels[1], classes)
        )

        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Cross Entropy')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()

    def train(self, data, labels, epochs):
        return

    # Utility for running experiments.
    def eval(self, data, labels):
        _, accuracy = self.model.evaluate(data, labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    def predict(self, data):
        return self.model.predict(data[0])

from tensorflow import keras
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical as tc


class JaiSVM(keras.regularizers.Regularizer):
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_svm_model(self)

    # Complete this method if implementing own gaussian_kernel
    # Possibly find a way to use 'c' and 'sigma'
    def __call__(self, x):
        print("x in regularizer: {}".format(x))

    def run_experiment(self, data, labels, epochs):
        self.utils.init_feature_extractor()
        classes = len(self.utils.get_vocabulary())
        history = self.train(
            data[0][0],
            tc(labels[0], classes),
            epochs
        )
        self.eval(data[1][0], tc(labels[1], classes))

        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Cross Entropy')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()

    def train(self, data, labels, epochs):
        return self.model.fit(
            data,
            labels,
            epochs=epochs,
            validation_split=0.2
        )

    # Utility for running experiments.
    def eval(self, data, labels):
        _, accuracy = self.model.evaluate(data, labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    def predict(self, data):
        return self.model.predict(data[0])

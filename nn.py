from tensorflow import keras
from matplotlib import pyplot
from datetime import datetime
import tensorboard


print(tensorboard.__version__)


# Testing the neural network model. Work in progress. To be moved.

class JaiNN:
    def __init__(self, utils):
        self.utils = utils
        self.gru_model = utils.get_gru_model()

    def run_experiment(self, data, labels, epochs):
        history = self.train(data[0], labels[0], epochs)
        self.eval(data[1], labels[1])

        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Cross Entropy')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()

    def train(self, train_data, train_labels, epochs):
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = "sav/" + date_str + "/cp"

        checkpoint = keras.callbacks.ModelCheckpoint(
            save_dir,
            save_weights_only=True,
            save_best_only=True,
            verbose=1
        )

        # Define the Keras TensorBoard callback.
        log_dir = "logs/fit/" + date_str + "/"
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        history = self.gru_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            validation_split=0.2,
            epochs=epochs,
            callbacks=[checkpoint, tensorboard_callback],
        )
        self.gru_model.load_weights(save_dir)
        return history

    # Utility for running experiments.
    def eval(self, data, labels):
        _, accuracy = self.gru_model.evaluate([data[0], data[1]], labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

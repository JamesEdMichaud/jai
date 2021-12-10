from tensorflow import keras
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import tensorflow_docs.plots
import tensorflow_docs as tfdocs

class JaiSVM(keras.regularizers.Regularizer):
    def __init__(self, utils):
        self.utils = utils
        self.model = utils.get_svm_model()

    # Complete this method if implementing own gaussian_kernel
    # Possibly find a way to use 'c' and 'sigma'
    def __call__(self, x):
        print("x in regularizer: {}".format(x))

    def parameter_tuning_curve(self, data, labels):
        train_data, test_data = data
        train_labels, test_labels = labels
        tf.random.set_seed(588)
        train_data, train_labels = self.utils.shuffle_data(train_data, train_labels)
        history = {}
        test_history = {}
        x = []
        for i in range(0, 300, 7):
            l2reg = i * 0.00002
            x.append(l2reg)
            print(f"Training with l2 regularization: {l2reg}")
            tf.random.set_seed(638)
            model = self.utils.get_svm_model(l2reg=l2reg)
            history[str(i)] = model.fit(
                train_data[0],
                to_categorical(train_labels),
                epochs=self.utils.epochs,
                validation_split=0.2,
                verbose=0
            )
            test_history[str(i)] = model.evaluate(
                test_data[0],
                to_categorical(test_labels)
            )
        test_acc = []
        test_losses = []
        val_losses = []
        losses = []
        for m, hist in history.items():
            losses.append(np.mean(hist.history['loss'][-10:]))
            val_losses.append(np.mean(hist.history['val_loss'][-10:]))
        for m, hist in test_history.items():
            test_losses.append(hist[0])
            test_acc.append(hist[1])
            m = len(train_labels)
        plotter = tfdocs.plots.HistoryPlotter(metric='categorical_crossentropy', smoothing_std=10)

        pyplot.title(f"Param Tuning Curve (samples: {round(m*0.8)}/{round(m*0.2)}/{len(test_labels)} train/val/test)")
        pyplot.xlabel('l2 regularization (lambda)')
        pyplot.ylabel('loss (Cross Entropy)')
        pyplot.plot(x, losses, label='train')
        pyplot.plot(x, val_losses, label='val')
        pyplot.plot(x, test_losses, label='test')
        pyplot.legend()
        pyplot.show()

    def learning_curve(self, data, labels):
        train_data, test_data = data
        train_labels, test_labels = labels
        tf.random.set_seed(588)
        train_data, train_labels = self.utils.shuffle_data(train_data, train_labels)
        history = {}
        test_history = {}
        x = []
        for i in range(3, len(train_labels)):
            print(f"Training with {i} examples")
            x.append(i)
            tf.random.set_seed(638)
            model = self.utils.get_svm_model()
            history[str(i)] = model.fit(
                train_data[0][:i, :],
                to_categorical(train_labels[:i, :]),
                epochs=self.utils.epochs,
                validation_split=0.2,
                verbose=0
            )
            test_history[str(i)] = model.evaluate(
                test_data[0],
                to_categorical(test_labels)
            )
        test_acc = []
        test_losses = []
        val_losses = []
        losses = []
        for m, hist in history.items():
            losses.append(np.mean(hist.history['loss'][-10:]))
            val_losses.append(np.mean(hist.history['val_loss'][-10:]))
        for m, hist in test_history.items():
            test_losses.append(hist[0])
            test_acc.append(hist[1])
        pyplot.title("Learning curve. (10/18/72)% test/val/train split")
        pyplot.xlabel('m')
        pyplot.ylabel('loss (Cross Entropy)')
        pyplot.plot(x, losses, label='train')
        pyplot.plot(x, val_losses, label='val')
        pyplot.plot(x, test_losses, label='test')
        pyplot.legend()
        pyplot.show()

    def loss_over_epochs(self, data, labels, epochs):
        train_data, test_data = data
        train_labels, test_labels = labels
        classes = len(self.utils.get_vocabulary())
        history = self.model.fit(
            train_data[0],
            to_categorical(train_labels, classes),
            epochs=epochs,
            validation_split=0.2
        )
        loss, accuracy = self.model.evaluate(
            test_data[0],
            to_categorical(test_labels, classes)
        )
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        m = len(train_data[0])
        pyplot.title(f"Loss (samples: {round(m*0.8)}/{round(m*0.2)}/{len(test_labels)} train/val/test)")
        pyplot.xlabel('Epoch')
        pyplot.ylabel('loss (Cross Entropy)')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()

    def predict(self, data):
        d = np.array(data[0])[None, ...]
        return self.model.predict_on_batch(d)

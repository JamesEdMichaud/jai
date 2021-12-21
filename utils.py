import os
import tensorflow as tf
import numpy as np
import cv2
import h5py
import imageio
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shutil
import time


def crop_center_square(frame):
    """
    Crops the given frame into a square. The smallest dimension (x,y) is used
    as the length of the square.

    This method was taken from this tutorial:
    https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

    :param frame: the frame to be cropped
    :return: the cropped frame
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]


def to_gif(frames, filename):
    """
    Saves the given frames to a gif file by the given filename.

    The .gif extension is added if filename does not end with it

    :param (np.ndarray) frames: the frames to be saved s a gif
    :param (str) filename: the name of the gif file to be saved
    """
    filename = filename if filename.endswith(".gif") else filename + ".gif"
    imageio.mimsave(filename, frames.astype(np.uint8), fps=10)


def build_data_index(root_dir):
    """
    Scans through the given root directory for training examples, compiling
    their paths and labels into an array.

    Directory structure must be in this form:
    root_dir
      label1
        example1_for_label1
        example2_for_label1
      label2
        example1_for_label2
        ...

    :param (str) root_dir: the directory to search through
    :return: an array of path/label pairings
    """
    data = []
    for label in sorted(os.listdir(root_dir)):
        if label.startswith("."):
            continue
        curr_example = os.path.join(root_dir, label)
        for vid in sorted(os.listdir(curr_example)):
            if vid.startswith("."):
                continue
            capture_path = os.path.join(curr_example, vid)
            data.append([capture_path, label])
    data = np.array(data)
    np.random.shuffle(data)
    return data


def get_split_indices(m, split, seed):
    """
    Generates two sets of indices whose total length equals m, with split*m
    indices in one set and (1-split)*m indices in the other.
    :param m: the number of indices to split
    :param split: the split percentage
    :param seed: random seed to be used
    :return: two arrays of indices
    """
    m1 = m // (1 / split)
    m2 = m // (1 / (1 - split))
    m1 += 1 if m - m1 - m2 > 0 else 0
    indices = np.arange(m + 1)
    indices = tf.random.shuffle(indices, seed=seed)
    return indices[:m1], indices[m1:]


def shuffle_and_split(data, labels, split, seed):
    """
    Splits the given data and labels using random indices.

    :param data: the data to be split
    :param labels: the labels to be split
    :param split: the split ratio
    :param seed: the random seed to be used when generating indices
    :return: the split data and labels
    """
    ids1, ids2 = get_split_indices(len(labels), split, seed)
    data1, labels1 = np.take(data, ids1), np.take(labels, ids1)
    data2, labels2 = np.take(data, ids2), np.take(labels, ids2)
    return data1, labels1, data2, labels2


def load_video(filename):
    """
    Loads a video using the given filename string. Uses cv2.VideoCapture to
    load frames.

    This method was adapted from a similar method in:
    https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

    :param (str) filename: the name of the video file to be loaded
    :return: a numpy array of frames that are the loaded video
    """
    frames = []
    cap = cv2.VideoCapture(filename)
    try:
        while True:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            is_frame, frame = cap.read()
            if is_frame:
                frames.append(frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                cv2.waitKey(100)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
    finally:
        cap.release()
    return np.array(frames)


def to_grayscale(frames):
    """
    Converts the given array of frames to grayscale.

    :param frames: the frames to be converted
    :return: the converted frames
    """
    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])


class JaiUtils:
    """
    A class used to assist in building and running machine learning models.
    """
    def __init__(self, vid_path, img_size, max_seq_len, train_split, val_split, learning_rate,
                 epochs, l2_reg, rand_seed, weights_seed, training_data_updated,
                 batch_size):
        """
        Initialize a new JaiUtils object with the given arguments.
        :param vid_path: the path where training data is located
        :param img_size: the image resolution to be used in this session
        :param max_seq_len: the maximum length of a video clip
        :param train_split: the percent of examples that go to the training set
        :param val_split: the percent of examples from the training set used for validation
        :param learning_rate: the default learning rate for machine learning models
        :param epochs: the default number of epochs to train for
        :param l2_reg: the default l2 regularization factor to use
        :param rand_seed: the random number seed to be used for shuffling and random ops
        :param weights_seed: the random number seed to be used for generating weights
        :param training_data_updated: boolean flag to indicate that there is new training data
        :param batch_size: the default batch size to use
        """
        self.vid_path = vid_path
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.train_split = train_split
        self.val_split = val_split
        self.learn_rate = learning_rate
        self.epochs = epochs
        self.l2reg = l2_reg
        self.rand_seed = rand_seed
        self.weights_seed = weights_seed
        self.training_data_updated = training_data_updated
        self.batch_size = batch_size
        self.learning_curve_x_label = "number of training examples"
        self.date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_index = build_data_index(self.vid_path)
        self.label_processor = self.init_label_processor()
        self.feature_extractor = None
        self.num_features = None

    def init_label_processor(self):
        """
        Initializes and returns a StringLookup layer to be used for label processing
        :return: the StringLookup layer
        """
        lp = tf.keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.data_index[:, 1])
        )
        print("Classes (labels): {}".format(lp.get_vocabulary()))
        return lp

    def get_label_set(self):
        """
        Gets the set of labels (as strings) present in the training set
        :return: the set of labels present in the training set
        """
        return self.label_processor.get_vocabulary()

    def get_label_count(self):
        """
        Gets the number of unique labels (classes) in the training set
        :return: the number of unique labels (classes) in the training set
        """
        return len(self.label_processor.get_vocabulary())

    def get_data(self):
        """
        Loads saved data if available, or preprocesses raw video data otherwise.
        :return: the data set to be used for this session
        """
        filename = "data_file.hdf5"
        try:
            if self.training_data_updated:
                raise IOError("Updating data records")
            else:
                print("Attempting to load saved video data")
                db = h5py.File("tmp/"+filename, 'r')
                data, lbls, mapping = db['data'][...], db['lbls'][...], db['mapping'][...]
                db.close()
                print("Using saved video data that has been processed")
        except IOError:
            if os.path.exists(os.path.join("tmp/", filename)):
                fr = os.path.join("tmp/", filename)
                to = "sav/" + self.date_str + "_"+filename
                print(f"Backing up: {filename} ==> {to}")
                shutil.move(fr, to)
            print("Processing all videos for network")
            data, lbls, mapping = self.preprocess_videos()
            print(f"Saving processed videos to tmp/{filename}")
            db = h5py.File("tmp/"+filename, "w")
            db.create_dataset(name="data", data=data, chunks=True, dtype=np.uint8)
            db.create_dataset(name="lbls", data=lbls, chunks=True, dtype=np.uint8)
            db.create_dataset(name="mapping", data=mapping, chunks=True, dtype=np.uint8)
            db.close()

        example_count = np.max(mapping)
        train_idx, test_idx = get_split_indices(example_count, self.train_split, self.rand_seed)

        trn_idx = np.squeeze(np.nonzero(np.in1d(mapping, train_idx)))
        tst_idx = np.squeeze(np.nonzero(np.in1d(mapping, test_idx)))
        trn_data, trn_lbls = np.take(data, trn_idx, axis=0), np.take(lbls, trn_idx, axis=0)
        tst_data, tst_lbls = np.take(data, tst_idx, axis=0), np.take(lbls, tst_idx, axis=0)
        print(f"Examples: {example_count+1} => {len(train_idx)}/{len(test_idx)} train/test")
        print(f"Examples including video spread: {len(trn_lbls)}/{len(tst_lbls)}")

        all_data = {
            'train_data': trn_data,
            'train_labels': trn_lbls,
            'test_data': tst_data,
            'test_labels': tst_lbls
        }
        return all_data

    def crop_and_resize_frames(self, frames):
        """
        Crops and resized the given frames. The central square is cropped first,
        then the resulting image is resized to the default image size.
        :param frames: the frames to be cropped and resized
        :return: the cropped and resized frames
        """
        new_frames = []
        for frame in frames:
            if frame.shape[0:1] != self.img_size:
                new_frame = crop_center_square(frame)
                new_frame = cv2.resize(new_frame, self.img_size)
            else:
                new_frame = frame
            new_frames.append(new_frame)
        return np.array(new_frames)

    def spread_video(self, frames, step=2):
        """
        Slices the given frames into smaller clips. The length of each clip is
        specified by the util param 'max_seq_len'. The step, or stride, indicates
        how many frames are skipped before creating another slice.
        :param frames: the frames to be sliced
        :param step: the number of frames to skip before creating another slice
        :return: an array of sliced clips, each of length max_seq_len
        """
        window_start = range(0, len(frames) - self.max_seq_len + 1, step)
        window_end = range(self.max_seq_len, len(frames) + 1, step)
        return np.array([frames[i:j] for i, j in zip(window_start, window_end)])

    def load_videos(self, video_paths):
        """
        Loads, crops, and resizes all video clips found using the given list of paths
        :param video_paths: the list of paths pointing to videos
        :return: the list of videos in the form of pixel values
        """
        return [self.crop_and_resize_frames(load_video(path)) for path in video_paths]

    def spread_videos(self, vids, lbls):
        """
        Spreads all videos using the spread_video() method. The labels are also spread
        to keep the videos and labels correctly paired.

        This method also creates a mapping from the original video index to a set of indices,
        which allows a set of spread clips to be added to be obtained using the original
        video's index.
        :param vids: the videos to be spread
        :param lbls: the accompanying labels
        :return: the spread videos, labels, and a mapping from the original video to the spread
        """
        spread_vids, spread_lbls, mapping = [], [], []
        count = 0
        for idx, (vid, lbl) in enumerate(zip(vids, lbls)):
            v = [vid] if len(vid) == self.max_seq_len else self.spread_video(vid, step=5)
            l = [lbl] if len(vid) == self.max_seq_len else np.full(len(v), lbl)
            i = [idx] if len(vid) == self.max_seq_len else np.full(len(v), idx)
            count += 1 if len(vid) == self.max_seq_len else len(v)
            spread_vids.extend(v)
            spread_lbls.extend(l)
            mapping.extend(i)
        return np.array(spread_vids), np.array(spread_lbls), np.array(mapping)

    def preprocess_videos(self):
        """
        Loads, crops, resizes, and spreads all videos found in the data index
        :return: the video clips, labels, and a mapping from the original videos their spread
        """
        print("Loading data from disk")
        videos = self.load_videos(self.data_index[:, 0])
        labels = self.label_processor(self.data_index[:, 1][..., None]).numpy()
        videos = [to_grayscale(vid) for vid in videos]
        videos, labels, mapping = self.spread_videos(videos, labels)
        return videos, labels, mapping

    def shuffle_data(self, data, labels, i=0):
        """
        Shuffles the given data and labels using the stored random seed.
        Optional param i offsets the seed for different randomness.
        :param data: the data to be shuffled
        :param labels: the labels to be shuffled
        :param i: the optional seed offset to use
        :return: shuffled data and labels
        """
        indices = np.arange(len(labels))
        indices = tf.random.shuffle(indices, seed=self.rand_seed+i)
        return data.take(indices, axis=0), labels.take(indices, axis=0)

    def get_gru_model(self, l2reg=None, learn_rate=None):
        """
        WORK IN PROGRESS
        Returns the neural network model to be used in this session.
        WORK IN PROGRESS
        :param l2reg: the l2 regularization factor to use
        :param learn_rate: the learning rate to use
        :return: a compiled neural network model
        """
        l2reg = self.l2reg if l2reg is None else l2reg
        learn_rate = self.learn_rate if learn_rate is None else learn_rate
        optimizer = tf.optimizers.SGD(learn_rate)
        # optimizer = tf.keras.optimizers.Adam(learn_rate)
        input_layer = tf.keras.Input(self.get_input_shape(), name="input")
        scale = tf.keras.layers.Rescaling(1./255.)(input_layer)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='valid',
            strides=2,
            name="conv1",
            ))(scale)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            activation='relu',
            padding='valid',
            strides=3,
            name="conv2",
            ))(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
        #     filters=8,
        #     kernel_size=(3, 3),
        #     activation='relu',
        #     padding='valid',
        #     strides=2,
        #     name="conv3",
        #     ))(x)
        x = tf.keras.layers.Reshape(
            (self.max_seq_len, x.shape[2] * x.shape[3] * x.shape[4]),
            name="reshape"
            )(x)
        x = tf.keras.layers.GRU(
            units=16,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="GRU1"
            )(x)
        x = tf.keras.layers.GRU(
            units=8,
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="GRU2"
            )(x)
        x = tf.keras.layers.Dropout(rate=0.4, name="dropout")(x)
        x = tf.keras.layers.Dense(
            units=8,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="Dense1"
            )(x)
        output = tf.keras.layers.Dense(
            units=len(self.get_label_set()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="output"
            )(x)

        rnn_model = tf.keras.Model(input_layer, output)

        rnn_model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.CategoricalCrossentropy(),
                'accuracy',
            ]
        )
        return rnn_model

    def get_logistic_reg_model(self, l2reg=None, learn_rate=None):
        """
        Builds a logistic regression model with a categorical crossentropy cost function
        :param l2reg: optional l2 regularization factor
        :param learn_rate: optional learning rate
        :return: a compiled logistic regression model
        """
        return self.get_basic_model(l2reg, learn_rate, 'categorical_crossentropy')

    def get_svm_model(self, l2reg=None, learn_rate=None):
        """
        Builds a support vector machine model with a categorical hinge cost function.
        :param l2reg: optional l2 regularization factor
        :param learn_rate: optional learning rate
        :return: a compiled logistic regression model
        """
        return self.get_basic_model(l2reg, learn_rate, "categorical_hinge")

    def get_basic_model(self, l2reg, learn_rate, loss_metric):
        """
        A basic single layer model, used as the basis for logistic regression
        and support vector machine models.
        :param l2reg: the l2 regularization factor to use for this model. None means use default
        :param learn_rate: The learning rate to use for this model. None means use default
        :param loss_metric: The cost function to use for this model. Required.
        :return: a compiled model of the given specifications
        """
        optimizer = tf.optimizers.SGD(self.learn_rate if learn_rate is None else learn_rate)
        metrics = [
            'accuracy',
        ]
        if loss_metric == "categorical_hinge":
            metrics.insert(0, tf.keras.metrics.CategoricalHinge())
        elif loss_metric == "categorical_crossentropy":
            metrics.insert(0, tf.keras.metrics.CategoricalCrossentropy())

        input_layer = tf.keras.Input(self.get_input_shape())
        scale = tf.keras.layers.Rescaling(1./255.)(input_layer)
        output = tf.keras.layers.Dense(
            units=self.get_label_count(),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2reg if l2reg is None else l2reg)
        )(tf.keras.layers.Flatten()(scale))

        model = tf.keras.Model(input_layer, output)
        model.compile(optimizer=optimizer, metrics=metrics, loss=loss_metric)
        return model

    def get_input_shape(self):
        """
        A central method to obtain the input shape for any model.
        Shape is: (max_seq_len, image_size0, image_size1, 1)
        :return: the input shape of a model
        """
        return self.max_seq_len, self.img_size[0], self.img_size[1], 1

    def get_split_string(self, data):
        """
        Helper function that returns a string indicating the train/val/test split of the training data
        :param data: the data
        :return: the split string
        """
        m = len(data['train_labels'])
        m_test = len(data['test_labels'])
        return f"{round(m * self.train_split)}/{round(m * (1 - self.train_split))}/{m_test}"

    def learning_rate_tuning_curve(self, data, get_model, metric, epochs, param_range, param_factor,
                                   l2reg=None, lr=None, **kwargs):
        """
        Method to call to generate a learning rate tuning curve using the given parameters.

        :param data: the data to use for training
        :param get_model: the method to call to obtain a machine learning model
        :param metric: the unregularized metric to track
        :param epochs: the number of epochs to train for
        :param param_range: the range of the metric being tested
        :param param_factor: the multiplier to use when iterating over the range
        :param l2reg: the l2 regularization factor to apply to the cost
        :param lr: the learning rate to use (ignored)
        :param kwargs: extra args that are ignored
        :return: the best val_loss model found during training
        """
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Param Tuning Curve (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="learning rate",
            param_range=param_range,
            param_factor=param_factor,
            epochs=epochs,
            l2reg=0.0 if l2reg is None else l2reg,
        )

    def l2_tuning_curve(self, data, get_model, metric, epochs, param_range, param_factor,
                        l2reg=None, lr=None, **kwargs):
        """
        Method to call to generate a parameter tuning curve on lambda using the given parameters.

        :param data: the data to use for training
        :param get_model: the method to call to obtain a machine learning model
        :param metric: the unregularized metric to track
        :param epochs: the number of epochs to train for
        :param param_range: the range of the metric being tested
        :param param_factor: the multiplier to use when iterating over the range
        :param l2reg: the l2 regularization factor (ignored)
        :param lr: the learning rate to use
        :param kwargs: extra args that are ignored
        :return: the best val_loss model found during training
        """
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Param Tuning Curve (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="l2 regularization (lambda)",
            param_range=param_range,
            learn_rate=self.learn_rate if lr is None else lr,
            param_factor=param_factor,
            epochs=epochs
        )

    def learning_curve(self, data, get_model, metric, param_range,
                       l2reg=None, lr=None, **kwargs):
        """
        The method to call to generate a learning curve with the given parameters
        :param data: the data to use for training
        :param get_model: the method to call to obtain a machine learning model
        :param metric: the unregularized metric to track
        :param epochs: the number of epochs to train for
        :param param_range: the range of the metric being tested
        :param l2reg: the l2 regularization factor to apply to the cost
        :param lr: the learning rate to use
        :param kwargs: extra args that are ignored
        :return:
        """
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Learning curve, test examples: {len(data['test_labels'])}",
            plot_x_label=self.learning_curve_x_label,
            param_range=param_range,
            learn_rate=self.learn_rate if lr is None else lr,
            l2reg=self.l2reg if l2reg is None else l2reg,
            param_factor=1,
        )

    def loss_over_epochs(self, data, get_model, metric, epochs, l2reg=None, lr=None, **kwargs):
        """
        The method to call to generate a loss-over-epochs plot.

        :param data: the data to use for training
        :param get_model: the method to call to obtain a machine learning model
        :param metric: the unregularized metric to track
        :param epochs: the number of epochs to train for
        :param l2reg: the l2 regularization factor to apply to the cost
        :param lr: the learning rate to use
        :param kwargs: extra args that are ignored
        :return:
        """
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Loss (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="epoch",
            learn_rate=self.learn_rate if lr is None else lr,
            l2reg=self.l2reg if l2reg is None else l2reg,
            param_range=[1],
            param_factor=1,
            epochs=epochs,
            single_run=True,
        )

    def train_and_plot_curve(self, data, get_model, metric, title, plot_x_label, param_range,
                             param_factor, learn_rate=None, l2reg=None, epochs=None, single_run=False):
        """
        Trains a new model and plots the results.
        :param data: the data to use for training
        :param get_model: the method to call to obtain a machine learning model
        :param metric: the unregularized metric to track
        :param title: the title to use on the resulting plot
        :param plot_x_label: the plot's x label
        :param param_range: the param range to use for tuning
        :param param_factor: the param multiplied to use on the range
        :param learn_rate: the learning rate to use while training
        :param l2reg: the regularization factor to use on the cost
        :param epochs: the number of epochs to train for
        :param single_run: is this a single run? (false means tuning curve)
        :return: the best val_loss model found during training
        """
        making_learning_curve = plot_x_label == self.learning_curve_x_label
        making_l2_curve = l2reg is None
        making_learn_rate_curve = learn_rate is None

        epochs = self.epochs if epochs is None else epochs

        train_data, train_labels, tst_data, tst_labels = data.values()
        train_data, train_labels = self.shuffle_data(train_data, train_labels)
        tst_data, tst_labels = self.shuffle_data(tst_data, tst_labels)

        history, test_history = {}, {}
        x, x_test = [], []

        best_val_model, best_metric = None, None
        min_val_loss, min_test_loss = 1e10, 1e10

        for i in range(*param_range):
            x_val = i * param_factor
            learn_rate = x_val if making_learn_rate_curve else learn_rate
            l2reg = x_val if making_l2_curve else l2reg
            x.append(x_val)
            x_test.append(x_val)

            trn_data = train_data[:i] if making_learning_curve else train_data
            trn_lbls = train_labels[:i] if making_learning_curve else train_labels

            tf.random.set_seed(self.weights_seed)
            model = get_model(learn_rate=learn_rate, l2reg=l2reg)

            print(f"Training over {plot_x_label} {x_val} - Learning rate: {learn_rate}, l2reg: {l2reg}")
            # start_time = time.time()
            history[str(i)] = model.fit(
                trn_data,
                to_categorical(trn_lbls, self.get_label_count()),
                epochs=epochs,
                validation_split=0.2,
                verbose=1 if single_run else 0,
                batch_size=self.batch_size
            )
            # end_time = time.time()
            # print(f"Model fit in {end_time-start_time} seconds")
            test_history[str(i)] = model.evaluate(
                tst_data,
                to_categorical(tst_labels, self.get_label_count())
            )
            if np.mean(history[str(i)].history['val_loss'][-10:]) < min_val_loss:
                best_val_model = model
                best_metric = x_val
        if single_run:
            self.plot_run(history['0'], test_history['0'], metric, title, plot_x_label)
        else:
            self.plot_results(history, test_history, metric, title, x, x_test, plot_x_label)
        print(f"Best metric from this run: {best_metric}")
        return best_val_model

    def plot_run(self, history, test_history, metric, title, x_label):
        """
        Organizes history data for a single training run, then passes it to the plotting method

        :param history: the training history
        :param test_history: the test history
        :param metric: the metric being tracked
        :param title: the title of the plot
        :param x_label: the x label of the plot
        :return: Nothing
        """
        args = {
            'title': title,
            'x_label': x_label,
            'x': np.arange(0, len(history.epoch)),
            'x_test': len(history.epoch),
            'test_mark': '*',
            'metric': metric,
            'trn_losses': history.history['loss'],
            'val_losses': history.history['val_loss'],
            'trn_metrics': history.history[metric],
            'val_metrics': history.history['val_'+metric],
            'trn_acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
            'test_losses': test_history[0],
            'test_metrics': test_history[1],
            'test_acc': test_history[2],
        }
        self.plot_data(**args)

    def plot_results(self, history, test_history, metric, title, x, x_test, x_label):
        """
        Organized history data from multiple trained models and plots the results
        :param history: the training history
        :param test_history: the test history
        :param metric: the metric being tracked
        :param title: the title of the plot
        :param x: the x values being plotted
        :param x_test: the x value(s) for the test data being plotted
        :param x_label: the x label for the plot
        :return: Nothing
        """
        args = {
            'title': title,
            'x': x,
            'x_test': x_test,
            'x_label': x_label,
            'metric': metric,
            'test_mark': ':',
            'trn_losses': [], 'val_losses': [], 'test_losses': [],
            'trn_metrics': [], 'val_metrics': [], 'test_metrics': [],
            'trn_acc': [], 'val_acc': [], 'test_acc': [],
        }
        for run in history.values():
            args['trn_metrics'].append(np.mean(run.history[metric][-10:]))
            args['val_metrics'].append(np.mean(run.history['val_'+metric][-10:]))
            args['trn_losses'].append(np.mean(run.history['loss'][-10:]))
            args['val_losses'].append(np.mean(run.history['val_loss'][-10:]))
            args['trn_acc'].append(np.mean(run.history['accuracy']))
            args['val_acc'].append(np.mean(run.history['val_accuracy']))
        for run in test_history.values():
            args['test_losses'].append(run[0])
            args['test_metrics'].append(run[1])
            args['test_acc'].append(run[2])
        self.plot_data(**args)

    def plot_data(self, title, x, x_label, x_test, metric, test_mark,
                  trn_metrics, val_metrics, test_metrics,
                  trn_losses, val_losses, test_losses,
                  trn_acc, test_acc, val_acc):
        """
        Takes the organized data from 'plot_results()' or 'plot_run()' and creates
        the plots
        :param title: the title of the plot
        :param x: the x values being plotted
        :param x_label: the label for the x axis
        :param x_test: the x value(s) being plotted for test data
        :param metric: the metric being tracked (usually cost)
        :param test_mark: the plot mark to use for test data
        :param trn_metrics: list of training metrics
        :param val_metrics: list of validation metrics
        :param test_metrics: list of test metrics
        :param trn_losses: list of training losses
        :param val_losses: list of validation losses
        :param test_losses: list of test losses
        :param trn_acc: list of training accuracies
        :param test_acc: list of test accuracies
        :param val_acc: list of validation accuracies
        :return: Nothing
        """
        fig, ax1 = pyplot.subplots()
        pyplot.title(title)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('cost ('+metric+')')
        # if making_l2_curve:
        ax1.plot(x, trn_metrics, 'r-', label='train_unreg')
        ax1.plot(x, val_metrics, 'r--', label='val_unreg')
        ax1.plot(x_test, test_metrics, 'r'+test_mark, label='test_unreg')

        ax1.plot(x, trn_losses, 'g-', label='train')
        ax1.plot(x, val_losses, 'g--', label='val')
        ax1.plot(x_test, test_losses, 'g'+test_mark, label='test_loss')

        ax2 = ax1.twinx()

        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(x_test, test_acc, 'b'+test_mark, label='test_accuracy')
        ax2.plot(x, val_acc, 'b--', label='val_accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        # fig.tight_layout()
        fig.legend(loc='center right', bbox_to_anchor=(0,0,0.9, 1))
        pyplot.show()

    def plot_roc_auc(self, labels, predictions):
        """
        Calculates and plots the ROC/AUC curve using the given labels and predictions

        :param labels: the y_true values
        :param predictions: the y_pred values
        :return: Nothing
        """
        fig, ax = pyplot.subplots()  # 1, 1, figsize=(12, 8))

        label_binarizer = LabelBinarizer()
        label_binarizer.fit(labels)
        b_labels = label_binarizer.transform(labels)
        b_predictions = label_binarizer.transform(predictions)
        for idx, label_str in enumerate(self.get_label_set()):
            false_pos_rate, true_pos_rate, threshold = roc_curve(b_labels[:, idx], b_predictions[:, idx])
            legend_label = "{} (AUC: {:1.3f})".format(label_str, auc(false_pos_rate, true_pos_rate))
            ax.plot(false_pos_rate, true_pos_rate, label=legend_label)
        pyplot.title(f"ROC AUC score: {roc_auc_score(b_labels, b_predictions, average='macro')}")
        ax.legend()
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        pyplot.show()

    def error_analysis(self, data, labels, predictions):
        """
        Saves gifs of incorrectly classified examples
        :param data: the training data (used to generate gifs)
        :param labels: the y_true values
        :param predictions: the y_pred values
        :return: the list of incorrectly classified clips
        """
        incorrect = np.where(predictions != labels)
        incorrect_vids = np.take(data, incorrect, axis=0).squeeze()

        for idx, clip in enumerate(incorrect_vids):
            to_gif(clip, f"errors/incorrect{idx}.gif")
        return incorrect_vids

from tensorflow_docs.vis import embed
import os
import tensorflow as tf
import numpy as np
import cv2
import imageio
import pathlib
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical


def crop_center_square(frame):
    # The following method was taken from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]


def to_gif(images):
    # This utility is for visualization.
    # Referenced from:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file(pathlib.Path("animation.gif"))


def build_data_index(root_dir):
    data = []
    for label in sorted(os.listdir(root_dir)):
        if label.startswith("."):
            continue
        curr_example = os.path.join(root_dir, label)
        for vid in sorted(os.listdir(curr_example)):
            if vid.startswith("."):
                continue
            capPath = os.path.join(curr_example, vid)
            data.append([capPath, label])
    data = np.array(data)
    np.random.shuffle(data)
    return data


class JaiUtils:
    def __init__(self, vid_path, img_size, max_seq_len,
                 train_split, learning_rate, epochs, l2_reg, l1_reg, c, sigma):
        self.vid_path = vid_path
        self.img_size = img_size
        self.frame_count = max_seq_len
        self.train_split = train_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.c = c
        self.sigma = sigma
        self.date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_index = build_data_index(self.vid_path)
        self.label_processor = self.init_label_processor()
        self.feature_extractor = None
        self.num_features = None

    def init_label_processor(self):
        lp = tf.keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.data_index[:, 1])
        )
        print("Vocab (labels): {}".format(lp.get_vocabulary()))
        self.build_feature_extractor()
        return lp

    def build_feature_extractor(self):
        input_shape = (self.img_size[0], self.img_size[1], 3)
        inputs = tf.keras.Input(input_shape)
        preprocessed = tf.keras.applications.inception_v3.preprocess_input(inputs)
        local_feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape,
        )
        outputs = local_feature_extractor(preprocessed)
        self.feature_extractor = tf.keras.Model(inputs, outputs, name="feature_extractor")
        self.num_features = self.feature_extractor.output.shape[1]

    def get_vocabulary(self):
        return self.label_processor.get_vocabulary()

    def load_or_process_video_data(self):
        try:
            print("Attempting to load saved video data")
            sd = np.load("tmp/prepared_videos.npz", allow_pickle=True)
            loaded_shape = sd['df'].shape
            ######### # (  Number of videos  ,   frames/video  ,  features/frame)
            dir_shape = (len(self.data_index), self.frame_count, loaded_shape[2])
            if dir_shape != loaded_shape:
                print(f"Saved data {loaded_shape} does not match available data {dir_shape}")
                raise IOError
            print("Using saved video data that has been processed")
            self.num_features = loaded_shape[2]
            data, labels = (sd['df'], sd['dm']), sd['dl']
        except IOError:
            print("Processing all videos for network")
            data, labels = self.prepare_all_videos()
            np.savez("tmp/prepared_videos.npz",
                     df=data[0],
                     dm=data[1],
                     dl=labels,
                     )
        m = labels.shape[0]
        if m < 10:
            raise RuntimeError("Get more training examples")
        train_m = m // (1 / self.train_split)
        test_m = m // (1 / (1 - self.train_split))
        train_m = int(train_m + 1 if (m - train_m - test_m) > 0 else 0)
        print("trainm: {}".format(train_m))
        indices = np.arange(m)
        indices = tf.random.shuffle(indices)
        tri, tsi = indices[:train_m], indices[train_m:]
        print(f"tri: {tri}\ntsi: {tsi}")
        trd, trl = (data[0][tri, :], data[1][tri, :]), labels[tri, :]
        tsd, tsl = (data[0][tsi, :], data[1][tsi, :]), labels[tsi, :]
        return trd, trl, tsd, tsl

    def shuffle_data(self, data, labels):
        indices = np.arange(len(labels))
        indices = tf.random.shuffle(indices)
        return (data[0][indices, :], data[1][indices, :]), labels[indices, :]

    # The following method was adapted from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    def load_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            is_frame, frame = cap.read()
            if is_frame:
                frames.append(frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                cv2.waitKey(500)
            if (cv2.waitKey(10) & 0xFF) == ord('q'):
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        return np.array(frames)

    def crop_and_resize_frames(self, frames):
        new_frames = []
        for frame in frames:
            if len(new_frames) == self.frame_count:
                break
            if frame.shape[0:1] != self.img_size:
                new_frame = crop_center_square(frame)
                new_frame = cv2.resize(new_frame, self.img_size)
            else:
                new_frame = frame
            new_frames.append(new_frame)
        return np.array(new_frames)


    def prepare_single_video(self, frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, self.frame_count,), dtype="bool")
        frame_features = np.zeros(shape=(1, self.frame_count, self.num_features), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(self.frame_count, video_length)
            for j in range(length):
                frame_features[i, j, :] = self.feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask

    def prepare_all_videos(self):
        self.init_label_processor()
        num_samples = len(self.data_index)
        video_paths = self.data_index[:, 0]
        labels = self.data_index[:, 1]
        labels = self.label_processor(labels[..., None])
        labels = labels.numpy()

        # `frame_masks` and `frame_features` are what we will feed to our sequence model.
        # `frame_masks` will contain a bunch of booleans denoting if a timestep is
        # masked with padding or not.
        frame_masks = np.zeros(shape=(num_samples, self.frame_count), dtype="bool")
        frame_features = np.zeros(
            shape=(num_samples, self.frame_count, self.num_features),
            dtype="float32"
        )

        # For each video.
        for idx, path in enumerate(video_paths):
            # Gather all its frames and add a batch dimension.
            frames = self.crop_and_resize_frames(self.load_video(path))
            temp_frame_features, temp_frame_mask = self.prepare_single_video(frames)
            frame_features[idx, ] = temp_frame_features.squeeze()
            frame_masks[idx, ] = temp_frame_mask.squeeze()

        return (frame_features, frame_masks), labels

    # Utility for our gru model.
    def get_gru_model(self, l2reg=None, learn_rate=None):
        l2reg = self.l2_reg if l2reg is None else l2reg
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        # optimizer = tf.optimizers.SGD(learn_rate)
        optimizer = tf.keras.optimizers.Adam(learn_rate)
        frame_features_input = tf.keras.Input((self.frame_count, self.num_features))
        mask_input = tf.keras.Input((self.frame_count,), dtype="bool")

        x = tf.keras.layers.GRU(
            units=16,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
            )(frame_features_input, mask=mask_input)
        x = tf.keras.layers.GRU(
            units=8,
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
            )(x)
        x = tf.keras.layers.Dropout(rate=0.4)(x)
        x = tf.keras.layers.Dense(
            units=8,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
            )(x)
        output = tf.keras.layers.Dense(
            units=len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
            )(x)

        rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

        rnn_model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.CategoricalCrossentropy(),
                'accuracy']
        )
        return rnn_model

    def get_logistic_reg_model(self, l2reg=None, learn_rate=None):
        l2reg = self.l2_reg if l2reg is None else l2reg
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        optimizer = tf.optimizers.SGD(learn_rate)
        frame_features_input = tf.keras.Input((self.frame_count, self.num_features))

        x = tf.keras.layers.Flatten()(frame_features_input)
        output = tf.keras.layers.Dense(
            units=len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg))(x)

        lr_model = tf.keras.Model(frame_features_input, output)
        lr_model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.CategoricalCrossentropy(),
                'accuracy']
        )
        return lr_model

    def get_svm_model(self, l2reg=None, learn_rate=None):
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        l2reg = self.l2_reg if l2reg is None else l2reg
        frame_features_input = tf.keras.Input((self.frame_count, self.num_features))
        optimizer = tf.optimizers.SGD(learn_rate)

        x = tf.keras.layers.Flatten()(frame_features_input)
        output = tf.keras.layers.Dense(
            units=len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
        )(x)

        svm_model = tf.keras.Model(frame_features_input, output)
        svm_model.compile(
            optimizer=optimizer,
            loss='categorical_hinge',
            metrics=[
                tf.keras.metrics.CategoricalHinge(name='categorical_hinge'),
                'accuracy']
        )
        return svm_model

    def get_motion_model(self):
        features_input = tf.keras.Input(self.img_size + (3,))
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=self.img_size+(3,),
            strides=2,
        )(features_input)
        x = tf.keras.layers.Dense(
            units=100,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(
            units=2,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)

        motion_model = tf.keras.Model(features_input, output)
        optimizer = tf.optimizers.SGD(self.learning_rate)
        motion_model.compile(
            optimizer=optimizer,
            loss='squared_hinge',
            metrics=['accuracy']
        )
        return motion_model

    def prediction(self, model, features, mask, label):
        class_vocab = self.label_processor.get_vocabulary()
        print(f"Test video label: {class_vocab[label[0]]}")

        probabilities = model.predict((features, mask))[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        # to_gif(features[:self.max_seq_len])
        return features

    def learning_rate_tuning_curve(self, data, get_model, metric, param_range, param_factor, is_gru=False):
        m = len(data[1])
        m_test = len(data[3])
        title = f"Param Tuning Curve (samples: {round(m*0.8)}/{round(m*0.2)}/{m_test} train/val/test)"
        return self.train_and_plot_curve(
            data=(data[0], data[2]),
            labels=(data[1], data[3]),
            get_model=get_model,
            metric=metric,
            title=title,
            plot_x_label="learning rate",
            param_range=param_range,
            l2reg=0,
            param_factor=param_factor,
            is_gru=is_gru
        )

    def l2_tuning_curve(self, data, get_model, metric, param_range, param_factor, is_gru=False, **kwargs):
        m = len(data[1])
        m_test = len(data[3])
        title = f"Param Tuning Curve (samples: {round(m*0.8)}/{round(m*0.2)}/{m_test} train/val/test)"
        return self.train_and_plot_curve(
            data=(data[0], data[2]),
            labels=(data[1], data[3]),
            get_model=get_model,
            metric=metric,
            title=title,
            plot_x_label="l2 regularization (lambda)",
            param_range=param_range,
            learn_rate=self.learning_rate,
            param_factor=param_factor,
            is_gru=is_gru
        )

    def learning_curve(self, data, get_model, metric, param_range, is_gru=False, **kwargs):
        title = "Learning curve. (10/18/72)% test/val/train split"
        return self.train_and_plot_curve(
            data=(data[0], data[2]),
            labels=(data[1], data[3]),
            get_model=get_model,
            metric=metric,
            title=title,
            plot_x_label="number of training examples",
            param_range=param_range,
            learn_rate=self.learning_rate,
            l2reg=self.l2_reg,
            param_factor=1,
            is_gru=is_gru
        )

    def loss_over_epochs(self, data, get_model, metric, epochs, is_gru=False, **kwargs):
        param_range = [2]
        param_factor = 1
        m = len(data[1])
        m_test = len(data[3])
        title = f"Loss (samples: {round(m*0.8)}/{round(m*0.2)}/{m_test} train/val/test)"
        return self.train_and_plot_curve(
            data=(data[0], data[2]),
            labels=(data[1], data[3]),
            get_model=get_model,
            metric=metric,
            title=title,
            plot_x_label="epoch",
            learn_rate=self.learning_rate,
            l2reg=self.l2_reg,
            param_range=param_range,
            param_factor=param_factor,
            epochs=epochs,
            single_run=True,
            is_gru=is_gru
        )

    def train_and_plot_curve(self, data, labels, get_model, metric, title, plot_x_label, param_range,
                             param_factor, is_gru, learn_rate=None, l2reg=None, epochs=None, single_run=False):
        epochs = self.epochs if epochs is None else epochs
        train_data, test_data = data
        train_labels, test_labels = labels
        tf.random.set_seed(588)
        train_data, train_labels = self.shuffle_data(train_data, train_labels)
        history = {}
        test_history = {}
        x = []
        x_test = []
        test_acc = []
        test_metrics = []
        test_losses = []
        val_metrics = []
        val_losses = []
        metrics = []
        losses = []
        best_val_model, best_test_model = None, None
        min_val_loss = 1e10
        min_test_loss = 1e10
        for i in range(*param_range):
            x_val = i*param_factor
            if learn_rate is None and l2reg is None:
                tr_data = train_data[0][:i, :]
                tr_data = [tr_data, train_data[1][:i, :]] if is_gru else tr_data
                tr_labels = train_labels[:i, :]
            else:
                tr_data = train_data[0]
                tr_data = [tr_data, train_data[1]] if is_gru else tr_data
                tr_labels = train_labels
            tst_data = test_data[0]
            tst_data = [tst_data, test_data[1]] if is_gru else tst_data
            learn_rate = x_val if learn_rate is None else learn_rate
            l2reg = x_val if l2reg is None else l2reg
            x.append(x_val)
            x_test.append(x_val)
            print(f"Training with {plot_x_label}: {x_val}")
            tf.random.set_seed(638)
            model = get_model(learn_rate=learn_rate, l2reg=l2reg)
            history[str(i)] = model.fit(
                tr_data,
                to_categorical(tr_labels),
                epochs=epochs,
                validation_split=0.2,
                verbose=1 if single_run else 0
            )
            test_history[str(i)] = model.evaluate(
                tst_data,
                to_categorical(test_labels)
            )
            if single_run:
                x = np.arange(0, epochs)
                x_test = epochs

                metrics = history[str(i)].history[metric]
                val_metrics = history[str(i)].history['val_'+metric]
                test_metrics = test_history[str(i)][1]

                losses = history[str(i)].history['loss']
                val_losses = history[str(i)].history['val_loss']
                test_losses = test_history[str(i)][0]

                test_acc = test_history[str(i)][2]

                best_val_model = model
                # pyplot.yscale('log')
                break

            metrics.append(np.mean(history[str(i)].history[metric][-10:]))
            val_metrics.append(np.mean(history[str(i)].history['val_'+metric][-10:]))
            test_metrics.append(test_history[str(i)][1])

            losses.append(np.mean(history[str(i)].history['loss'][-10:]))
            val_losses.append(np.mean(history[str(i)].history['val_loss'][-10:]))
            test_losses.append(test_history[str(i)][0])

            test_acc.append(test_history[str(i)][2])
            if test_losses[-1] < min_test_loss:
                best_test_model = model
            if val_losses[-1] < min_val_loss:
                best_val_model = model

        pyplot.title(title)
        pyplot.xlabel(plot_x_label)
        pyplot.ylabel('cost ('+metric+')')
        pyplot.plot(x, losses, 'g-', label='train_reg')
        pyplot.plot(x, val_losses, 'g--', label='val_reg')
        pyplot.plot(x_test, test_losses, 'bD', label='test_reg')
        pyplot.plot(x, metrics, 'r-', label='train_unreg')
        pyplot.plot(x, val_metrics, 'r--', label='val_unreg')
        pyplot.plot(x_test, test_metrics, 'c*', label='test_unreg')
        pyplot.legend()
        pyplot.show()
        return best_val_model #, best_test_model

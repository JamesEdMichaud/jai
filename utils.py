from tensorflow_docs.vis import embed
import os
import tensorflow as tf
import numpy as np
import cv2
import imageio
import pathlib
from datetime import datetime


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
    for run_number in sorted(os.listdir(root_dir)):
        if run_number.startswith("."):
            continue
        run = os.path.join(root_dir, run_number)
        for cap in sorted(os.listdir(run)):
            if cap.startswith("."):
                continue
            capPath = os.path.join(run, cap)
            data.append([capPath, cap.split("_")[0]])
    data = np.array(data)
    np.random.shuffle(data)
    return data


class JaiUtils:
    def __init__(self, vid_path, img_size, max_seq_len, train_split, learning_rate, l2_reg, l1_reg, c, sigma):
        self.vid_path = vid_path
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.train_split = train_split
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.c = c
        self.sigma = sigma
        self.date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_index = build_data_index(self.vid_path)
        self.label_processor = None
        self.feature_extractor = None
        self.num_features = None

    def init_feature_extractor(self):
        self.label_processor = tf.keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.data_index[:, 1])
        )
        print("Vocab (labels): {}".format(self.get_vocabulary()))

        self.feature_extractor = self.build_feature_extractor()
        self.num_features = self.feature_extractor.output.shape[1]

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
        return tf.keras.Model(inputs, outputs, name="feature_extractor")

    def get_vocabulary(self):
        return self.label_processor.get_vocabulary()

    def load_or_process_video_data(self):
        try:
            print("Attempting to load saved video data")
            sd = np.load("tmp/prepared_videos.npz", allow_pickle=True)
            loaded_shape = sd['df'].shape
            real_shape = (len(self.data_index), self.max_seq_len, self.num_features)
            if real_shape != loaded_shape:
                print("Saved data {} not same shape as available data {}".format(loaded_shape, real_shape))
                raise IOError
            print("Using saved video data that has been processed")
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
        indices = np.random.permutation(m)
        tri, tsi = indices[:train_m], indices[train_m:]
        trd, trl = (data[0][tri, :], data[1][tri, :]), labels[tri, :]
        tsd, tsl = (data[0][tsi, :], data[1][tsi, :]), labels[tsi, :]
        return trd, trl, tsd, tsl

    # The following method was adapted from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    def load_video(self, path):
        frames = []
        for file in sorted(os.listdir(path)):
            file = os.fsdecode(file)
            if file.startswith('.'):
                continue
            frame = cv2.imread(os.path.join(path, file))
            # frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
            # frame = crop_center_square(frame)
            # if frame.shape[0:1] != self.img_size:
            #     frame = cv2.resize(frame, self.img_size)
            frames.append(frame)
        return np.array(frames)

    def crop_and_resize_frames(self, frames):
        new_frames = []
        for frame in frames:
            if len(new_frames) == self.max_seq_len:
                break
            if frame.shape[0:1] != self.img_size:
                new_frame = crop_center_square(frame)
                new_frame = cv2.resize(new_frame, self.img_size)
            else:
                new_frame = frame
            new_frames.append(new_frame)
        return new_frames


    def prepare_single_video(self, frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, self.max_seq_len,), dtype="bool")
        frame_features = np.zeros(shape=(1, self.max_seq_len, self.num_features), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(self.max_seq_len, video_length)
            for j in range(length):
                frame_features[i, j, :] = self.feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask

    def prepare_all_videos(self):
        num_samples = len(self.data_index)
        video_paths = self.data_index[:, 0]
        labels = self.data_index[:, 1]
        labels = self.label_processor(labels[..., None])
        labels = labels.numpy()

        # `frame_masks` and `frame_features` are what we will feed to our sequence model.
        # `frame_masks` will contain a bunch of booleans denoting if a timestep is
        # masked with padding or not.
        frame_masks = np.zeros(shape=(num_samples, self.max_seq_len), dtype="bool")
        frame_features = np.zeros(
            shape=(num_samples, self.max_seq_len, self.num_features),
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
    def get_gru_model(self):
        frame_features_input = tf.keras.Input((self.max_seq_len, self.num_features))
        mask_input = tf.keras.Input((self.max_seq_len,), dtype="bool")

        x = tf.keras.layers.GRU(16, return_sequences=True)(
            frame_features_input, mask=mask_input
        )
        x = tf.keras.layers.GRU(8)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        output = tf.keras.layers.Dense(
            len(self.get_vocabulary()),
            activation="softmax")(x)

        rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

        rnn_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return rnn_model

    def get_logistic_reg_model(self):
        frame_features_input = tf.keras.Input((self.max_seq_len, self.num_features))
        mask_input = tf.keras.Input((self.max_seq_len,), dtype="bool")

        x = tf.keras.layers.Flatten()(frame_features_input)
        output = tf.keras.layers.Dense(
            len(self.get_vocabulary()),
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(x)

        lr_model = tf.keras.Model(frame_features_input, output)
        optimizer = tf.optimizers.SGD(self.learning_rate)
        lr_model.compile(
            optimizer=optimizer, loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return lr_model

    def get_svm_model(self, gaussian_kernel):
        frame_features_input = tf.keras.Input((self.max_seq_len, self.num_features))

        x = tf.keras.layers.Flatten()(frame_features_input)
        output = tf.keras.layers.Dense(
            len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(x)

        svm_model = tf.keras.Model(frame_features_input, output)
        optimizer = tf.optimizers.SGD(self.learning_rate)
        svm_model.compile(
            optimizer=optimizer, loss='categorical_hinge',
            metrics=['accuracy']
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

    def prediction(self, model, path):
        print(f"Test video path: {path}")
        class_vocab = self.label_processor.get_vocabulary()

        frames = self.crop_and_resize_frames(self.load_video(path))
        frame_features, frame_mask = self.prepare_single_video(frames)
        probabilities = model.predict([frame_features, frame_mask])[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        to_gif(frames[:self.max_seq_len])
        return frames

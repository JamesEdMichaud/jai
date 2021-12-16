import os
import tensorflow as tf
import numpy as np
import cv2
import h5py
import imageio
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
import vidaug.augmentors as va
import shutil


def crop_center_square(frame):
    # The following method was taken from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]


def to_gif(frames, name):
    # This utility is for visualization.
    # Referenced from:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    name = name if name.endswith(".gif") else name+".gif"
    converted_images = frames.astype(np.uint8)
    imageio.mimsave(name, converted_images, fps=10)


def build_data_index(root_dir):
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


def shuffle_and_split(data, labels, train_split):
    indices = np.arange(len(labels))
    indices = tf.random.shuffle(indices)
    data = data.take(indices, axis=0)
    labels = labels.take(indices, axis=0)
    tri, tsi = get_split_indices(labels.shape[0], train_split)
    trd, trl = np.take(data, tri), np.take(labels, tri)
    tsd, tsl = np.take(data, tsi), np.take(labels, tsi)
    return trd, trl, tsd, tsl


def get_split_indices(m, train_split):
    train_m = m // (1 / train_split)
    test_m = m // (1 / (1 - train_split))
    train_m = int(train_m + 1 if (m - train_m - test_m) > 0 else 0)
    indices = np.arange(m)
    indices = tf.random.shuffle(indices)
    return indices[:train_m], indices[train_m:]


# The following method was adapted from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path):
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
        if (cv2.waitKey(2) & 0xFF) == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
    return np.array(frames)


class JaiUtils:
    def __init__(self, vid_path, img_size, max_seq_len, train_split, val_split, learning_rate,
                 epochs, l2_reg, l1_reg, c, sigma, rand_seed, weights_seed, training_data_updated,
                 batch_size, using_feature_extractor, using_augmentation):
        self.vid_path = vid_path
        self.img_size = img_size
        self.frame_count = max_seq_len
        self.train_split = train_split
        self.val_split = val_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.c = c
        self.sigma = sigma
        self.rand_seed = rand_seed
        self.weights_seed = weights_seed
        self.training_data_updated = training_data_updated
        self.batch_size = batch_size
        self.using_feature_extractor = using_feature_extractor
        self.using_augmentation = using_augmentation
        self.date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_index = build_data_index(self.vid_path)
        self.label_processor = self.init_label_processor()
        self.feature_extractor = None
        self.num_features = None
        self.augmentation_ops = self.build_aug_list()

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

    def get_vocab_length(self):
        return len(self.label_processor.get_vocabulary())

    def load_or_preprocess_video_data(self):
        ipfn = f"inception_preprocessed_data_file.hdf5"
        filename = "data_file.hdf5" if not self.using_feature_extractor else ipfn
        try:
            if self.training_data_updated:
                raise IOError("Let's update our hdf5 records (see except)")
            else:
                print("Attempting to load saved video data")
                if not self.using_feature_extractor:
                    f = h5py.File("tmp/"+filename, 'r')
                    trd, trl = f['trd'][...], f['trl'][...]
                    tsd, tsl = f['tsd'][...], f['tsl'][...]
                else:
                    f = h5py.File("tmp/"+filename, 'r')
                    train_features = f["train/features"][...]
                    train_masks = f["train/masks"][...]
                    train_labels = f["train/labels"][...]
                    test_features = f["test/features"][...]
                    test_masks = f["test/masks"][...]
                    test_labels = f["test/labels"][...]
                    self.num_features = train_features.shape[2]
                    trd, trl = (train_features, train_masks), train_labels
                    tsd, tsl = (test_features, test_masks), test_labels
                f.close()
                print("Using saved video data that has been processed")
        except IOError:
            if os.path.exists(os.path.join("tmp/", filename)):
                fr = os.path.join("tmp/", filename)
                to = "sav/" + self.date_str + "_"+filename
                print(f"Backing up: {filename} ==> {to}")
                shutil.move(fr, to)
            print("Processing all videos for network")
            if not self.using_feature_extractor:
                db = h5py.File("tmp/"+filename, "w")
                trd, trl, tsd, tsl = self.prep_vids()
                db.create_dataset(name="trd", data=trd, chunks=True, dtype=np.uint8)
                db.create_dataset(name="trl", data=trl, chunks=True, dtype=np.uint8)
                db.create_dataset(name="tsd", data=tsd, chunks=True, dtype=np.uint8)
                db.create_dataset(name="tsl", data=tsl, chunks=True, dtype=np.uint8)
            else:
                db = h5py.File("tmp/"+filename, "w")
                tf.random.set_seed(self.rand_seed)
                trd, trl, tsd, tsl = self.extract_features_from_videos()
                db.create_dataset(chunks=True, dtype=np.float32,
                                  name="train/features", data=trd[0])
                db.create_dataset(chunks=True, dtype=np.bool,
                                  name="train/masks", data=trd[1])
                db.create_dataset(chunks=True, dtype=np.bool,
                                  name="train/labels", data=trl)
                db.create_dataset(chunks=True, dtype=np.float32,
                                  name="test/features", data=tsd[0])
                db.create_dataset(chunks=True, dtype=np.bool,
                                  name="test/masks", data=tsd[1])
                db.create_dataset(chunks=True, dtype=np.bool,
                                  name="train/labels", data=tsl)
            db.close()
        return trd, trl, tsd, tsl

    def crop_and_resize_frames(self, frames):
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
        window_start = range(0, len(frames)-self.frame_count+1, step)
        window_end = range(self.frame_count, len(frames)+1, step)
        return [frames[i:j] for i, j in zip(window_start, window_end)]

    def build_aug_list(self):
        return [
            # va.HorizontalFlip(),

            # va.RandomRotate(degrees=13),
            # va.Superpixel(p_replace=[1, 1, 0, 1, 0, 0, 1, 0], n_segments=400, interpolation="nearest"),
            # va.TemporalElasticTransformation(),
            # va.ElasticTransformation(alpha=80, sigma=17, order=3, cval=0, mode="nearest"),
            # va.Add(-30),

            # va.RandomShear(x=0.15, y=0.1),
            # va.Superpixel(p_replace=[1, 0, 1, 0, 0, 1, 0, 1], n_segments=300, interpolation="nearest"),

            # va.Pepper(20),
            # va.Add(30),
            va.OneOf([va.RandomShear(x=0.3, y=0), va.TemporalElasticTransformation()]),
            va.OneOf([va.Superpixel(p_replace=[0, 1, 0, 1, 1, 0, 1, 0], n_segments=400, interpolation="nearest"),
                      va.TemporalElasticTransformation(),
                      va.ElasticTransformation(alpha=5, sigma=1, order=3, cval=0, mode="nearest")]),

            # va.Superpixel(p_replace=[0, 1, 0, 1, 1, 0, 1, 0], n_segments=400, interpolation="nearest"),
            # va.TemporalElasticTransformation(),
            # va.ElasticTransformation(alpha=40, sigma=9, order=3, cval=0, mode="nearest"),
            # va.Salt(20),
            # va.Multiply(1.75),

            # va.RandomShear(x=0, y=0.2),
            # va.Superpixel(p_replace=[1, 1, 0, 0, 1, 1, 0, 0], n_segments=100, interpolation="bilinear"),
            # va.RandomTranslate(x=30, y=20),
            # va.InvertColor(),
            # [va.ElasticTransformation(alpha=5, sigma=1, order=3, cval=0, mode="nearest"),
            #  va.InvertColor()]
        ]

    def augment_video(self, frames):
        seqs = [va.Sequential(op if isinstance(op, list) else [op]) for op in self.augmentation_ops]
        return np.array([seq(frames) for seq in seqs])

    def feature_extract_video(self, frames):
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames)
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

    def load_videos(self, video_paths, video_labels):
        videos, labels = [], []
        for idx, (path, lbl) in enumerate(zip(video_paths, video_labels)):
            videos.extend([self.crop_and_resize_frames(load_video(path))])
            labels.extend([lbl])
        return videos, labels

    def augment_videos(self, videos, labels):
        vids, lbls = [], []
        for idx, (vid, lbl) in enumerate(zip(videos, labels)):
            vids.extend([self.to_grayscale(vid)])
            lbls.extend([lbl])
            if idx % 10 == 0:
                print(f"On video {idx}")
            augmented = self.augment_video(vid)
            augmented = np.array([self.to_grayscale(augd) for augd in augmented])
            vids.extend(augmented)
            lbls.extend(np.full(len(augmented), lbl))
        return vids, lbls

    def to_grayscale(self, frames):
        return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    def spread_videos(self, vids, lbls):
        spread_vids, spread_lbls = [], []
        for vid, lbl in zip(vids, lbls):
            if len(vid) <= self.frame_count:
                spread_vids.extend([vid])
                spread_lbls.extend([lbl])
            else:
                spread_frames = self.spread_video(vid, step=5)
                spread_vids.extend(spread_frames)
                spread_lbls.extend(np.full(len(spread_frames), lbl))
        return np.array(spread_vids), np.array(spread_lbls)

    def prep_vids(self):
        video_paths = self.data_index[:, 0]
        labels = self.data_index[:, 1]
        labels = self.label_processor(labels[..., None])
        labels = labels.numpy()
        tri, tsi = get_split_indices(labels.shape[0], self.train_split)
        train_paths, train_labels = np.take(video_paths, tri), np.take(labels, tri)
        test_paths, test_labels = np.take(video_paths, tsi), np.take(labels, tsi)

        vids, lbls = self.load_videos(train_paths, train_labels)

        if self.using_augmentation:
            print(f"Augmenting training data. Initial examples count: {len(vids)}")
            vids, lbls = self.augment_videos(vids, lbls)
            print(f"Augmenting complete. New count: {len(vids)}")
        else:
            vids = np.array([self.to_grayscale(vid) for vid in vids])

        print(f"Spreading training data")
        train_vids, train_lbls = self.spread_videos(vids, lbls)
        print(f"New training data shape: {train_vids.shape}")

        vids, lbls = self.load_videos(test_paths, test_labels)
        vids = np.array([self.to_grayscale(vid) for vid in vids])

        print(f"Spreading test data. Initial examples count: {len(vids)}")
        test_vids, test_lbls = self.spread_videos(vids, lbls)
        print(f"New test data shape: {test_vids.shape}")

        return train_vids, train_lbls, test_vids, test_lbls

    def shuffle_data(self, data, labels, i=0):
        indices = np.arange(len(labels))
        indices = tf.random.shuffle(indices, seed=self.rand_seed+i)
        if self.using_feature_extractor:
            feats = data[0].take(indices, axis=0)
            masks = data[1].take(indices, axis=0)
            dta = (feats, masks)
        else:
            dta = data.take(indices, axis=0)
        labls = labels.take(indices, axis=0)
        return dta, labls

    def extract_features_from_videos(self):
        train_vids, train_lbls, test_vids, test_lbls = self.prep_vids()

        train_frame_masks = np.zeros(
            shape=(train_vids.shape[0], self.frame_count),
            dtype="bool")
        train_frame_features = np.zeros(
            shape=(train_vids.shape[0], self.frame_count, self.num_features),
            dtype="float32")
        test_frame_masks = np.zeros(
            shape=(test_vids.shape[0], self.frame_count),
            dtype="bool")
        test_frame_features = np.zeros(
            shape=(test_vids.shape[0], self.frame_count, self.num_features),
            dtype="float32")

        print(f"Extracting training features. Original shape: {train_vids.shape}")
        for idx, vid in enumerate(train_vids):
            temp_frame_features, temp_frame_mask = self.feature_extract_video(vid.copy())
            train_frame_features[idx, ] = temp_frame_features.squeeze()
            train_frame_masks[idx, ] = temp_frame_mask.squeeze()
        print(f"New training data shape: {train_frame_features.shape}")

        print(f"Extracting test features. Original shape: {test_vids.shape}")
        for idx, vid in enumerate(test_vids):
            temp_frame_features, temp_frame_mask = self.feature_extract_video(vid)
            test_frame_features[idx, ] = temp_frame_features.squeeze()
            test_frame_masks[idx, ] = temp_frame_mask.squeeze()
        print(f"New training data shape: {test_frame_features.shape}")
        trd = (train_frame_features, train_frame_masks)
        tsd = (test_frame_features, test_frame_masks)
        return trd, train_lbls, tsd, test_lbls

    def get_gru_model(self, l2reg=None, learn_rate=None):
        l2reg = self.l2_reg if l2reg is None else l2reg
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        optimizer = tf.optimizers.SGD(learn_rate)
        # optimizer = tf.keras.optimizers.Adam(learn_rate)
        if self.using_feature_extractor:
            frame_features_input = tf.keras.Input((self.frame_count, self.num_features))
            mask_input = tf.keras.Input((self.frame_count,), dtype="bool")
            inpt = {'inputs': frame_features_input, 'mask': mask_input}
            start = [frame_features_input, mask_input]
        else:
            shape = (self.frame_count, self.img_size[0], self.img_size[1], 1)
            input_layer = tf.keras.Input(shape, name="input")
            conv1 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(7, 7),
                activation='relu',
                padding='valid',
                strides=4,
                name="conv1"
            )
            conv2 = tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(5, 5),
                activation='relu',
                padding='valid',
                strides=3,
                name="conv2"
            )
            conv3 = tf.keras.layers.Conv2D(
                filters=4,
                kernel_size=(3, 3),
                activation='relu',
                padding='valid',
                strides=2,
                name="conv3"
            )
            time_conv1 = tf.keras.layers.TimeDistributed(conv1)(input_layer)
            time_conv2 = tf.keras.layers.TimeDistributed(conv2)(time_conv1)
            time_conv2 = tf.keras.layers.TimeDistributed(conv3)(time_conv2)
            size = time_conv2.shape[2]*time_conv2.shape[3]*time_conv2.shape[4]
            flat = tf.keras.layers.Reshape(
                (self.frame_count, size),
                name="reshape"
            )(time_conv2)

            inpt = {'inputs': flat}
            start = input_layer

        x = tf.keras.layers.GRU(
            units=16,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="GRU1"
            )(**inpt)
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
            units=len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            name="output"
            )(x)

        rnn_model = tf.keras.Model(start, output)

        rnn_model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=[
                'accuracy',
                tf.keras.metrics.CategoricalCrossentropy(),
            ]
        )
        return rnn_model

    def get_logistic_reg_model(self, l2reg=None, learn_rate=None):
        return self.get_basic_model(l2reg, learn_rate, 'categorical_crossentropy')

    def get_svm_model(self, l2reg=None, learn_rate=None):
        return self.get_basic_model(l2reg, learn_rate, "categorical_hinge")

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

    def get_basic_model(self, l2reg, learn_rate, loss_metric):
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        l2reg = self.l2_reg if l2reg is None else l2reg

        inpt = tf.keras.Input(self.get_input_shape())
        x = tf.keras.layers.Flatten()(inpt)
        output = tf.keras.layers.Dense(
            units=len(self.get_vocabulary()),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)
        )(x)

        model = tf.keras.Model(inpt, output)

        metrics = ['accuracy']
        if loss_metric == "categorical_hinge":
            metrics.append(tf.keras.metrics.CategoricalHinge(name='loss_unreg'))
        elif loss_metric == "categorical_crossentropy":
            metrics.append(tf.keras.metrics.CategoricalCrossentropy(name="loss_unreg"))

        optimizer = tf.optimizers.SGD(learn_rate)
        model.compile(optimizer=optimizer, metrics=metrics, loss=loss_metric)

        return model

    def get_input_shape(self):
        if self.using_feature_extractor:
            return (self.frame_count, self.num_features)
        else:
            return (self.frame_count, self.img_size[0], self.img_size[1], 1)

    def prediction(self, model, data, label):
        class_vocab = self.label_processor.get_vocabulary()
        print(f"Test video label: {class_vocab[label]}")

        probabilities = model.predict(data)[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        # to_gif(features, "testing")

    def learning_rate_tuning_curve(self, data, get_model, metric, epochs, param_range, 
                                   param_factor, is_gru=False, **kwargs):
        m = len(data[1])
        m_test = len(data[3])
        split = f"{round(m*self.train_split)}/{round(m*(1-self.train_split))}/{m_test}"
        title = f"Param Tuning Curve (samples: {split} train/val/test)"
        return self.train_and_plot_curve(
            get_model=get_model, metric=metric, title=title, epochs=epochs, is_gru=is_gru,
            param_range=param_range, param_factor=param_factor,
            data=(data[0], data[2]), labels=(data[1], data[3]),
            plot_x_label="learning rate", l2reg=0,
        )

    def l2_tuning_curve(self, data, get_model, metric, param_range, param_factor, epochs,
                        is_gru=False, **kwargs):
        m = len(data[1])
        m_test = len(data[3])
        split = f"{round(m*self.train_split)}/{round(m*(1-self.train_split))}/{m_test}"
        title = f"Param Tuning Curve (samples: {split} train/val/test)"
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
            is_gru=is_gru,
            epochs=epochs
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
        m = len(data[1])
        m_test = len(data[3])
        split = f"{round(m*self.train_split)}/{round(m*(1-self.train_split))}/{m_test}"
        title = f"Loss (samples: {split} train/val/test)"
        return self.train_and_plot_curve(
            data=(data[0], data[2]),
            labels=(data[1], data[3]),
            get_model=get_model,
            metric=metric,
            title=title,
            plot_x_label="epoch",
            learn_rate=self.learning_rate,
            l2reg=self.l2_reg,
            param_range=[2],
            param_factor=1,
            epochs=epochs,
            single_run=True,
            is_gru=is_gru
        )

    def train_and_plot_curve(self, data, labels, get_model, metric, title, plot_x_label, param_range,
                             param_factor, is_gru, learn_rate=None, l2reg=None, epochs=None, single_run=False):
        making_learning_curve = plot_x_label == "number of training examples"
        making_l2_curve = l2reg is None
        making_learn_rate_curve = learn_rate is None

        epochs = self.epochs if epochs is None else epochs

        history, test_history = {}, {}

        x, x_test = [], []
        test_acc, val_losses, test_losses, losses = [], [], [], []
        val_acc, test_acc = [], []
        val_metrics, test_metrics, metrics = [], [], []

        best_val_model, best_test_model = None, None
        min_val_loss, min_test_loss = 1e10, 1e10
        for i in range(*param_range):
            train_data, train_labels, test_data, test_labels = self.pick_data_format(
                data, labels, i, making_learning_curve, is_gru)

            x_val = i * param_factor
            learn_rate = x_val if making_learn_rate_curve else learn_rate
            l2reg = x_val if making_l2_curve else l2reg

            x.append(x_val)
            x_test.append(x_val)
            print(f"Training with {plot_x_label}: {x_val}. Learning rate: {learn_rate}, l2reg: {l2reg}")
            # learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=learn_rate*10,
            #     decay_steps=epochs//0.8,
            #     decay_rate=0.4,
            # )
            stop_early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                min_delta=0.0001,
                verbose=1
            )
            tf.random.set_seed(self.weights_seed)
            model = get_model(learn_rate=learn_rate, l2reg=l2reg)
            history[str(i)] = model.fit(
                train_data,
                to_categorical(train_labels, self.get_vocab_length()),
                epochs=epochs,
                validation_split=0.2,
                verbose=1 if single_run else 0,
                callbacks=[stop_early],
                batch_size=self.batch_size
            )
            test_history[str(i)] = model.evaluate(
                test_data,
                to_categorical(test_labels, self.get_vocab_length())
            )

            # TODO: Move plotting features into separate function
            if single_run:
                x = np.arange(0, epochs)
                x_test = epochs

                if making_l2_curve:
                    metrics = history[str(i)].history[metric]
                    val_metrics = history[str(i)].history['val_'+metric]
                    test_metrics = test_history[str(i)][1]

                losses = history[str(i)].history['loss']
                val_losses = history[str(i)].history['val_loss']
                test_losses = test_history[str(i)][0]
                val_acc = history[str(i)].history['val_accuracy']
                test_acc = test_history[str(i)][1]

                best_val_model = model
                # pyplot.yscale('log')
                break

            if making_l2_curve:
                metrics.append(np.mean(history[str(i)].history[metric][-10:]))
                val_metrics.append(np.mean(history[str(i)].history['val_'+metric][-10:]))
                test_metrics.append(test_history[str(i)][1])

            losses.append(np.mean(history[str(i)].history['loss'][-10:]))
            val_losses.append(np.mean(history[str(i)].history['val_loss'][-10:]))
            test_losses.append(test_history[str(i)][0])

            val_acc.append(np.mean(history[str(i)].history['val_accuracy']))
            test_acc.append(test_history[str(i)][1])

            if test_losses[-1] < min_test_loss:
                best_test_model = model
            if val_losses[-1] < min_val_loss:
                best_val_model = model

        if single_run:
            x = np.arange(0, len(losses))
            x_test = len(losses)
        fig, ax1 = pyplot.subplots()
        pyplot.title(title)
        ax1.set_xlabel(plot_x_label)
        ax1.set_ylabel('cost ('+metric+')')
        if making_l2_curve:
            mark = 'rd' if single_run else 'r:'
            ax1.plot(x, metrics, 'r-', label='train_unreg')
            ax1.plot(x, val_metrics, 'r--', label='val_unreg')
            ax1.plot(x_test, test_metrics, mark, label='test_unreg')

        mark = 'g*' if single_run else 'g:'
        ax1.plot(x, losses, 'g-', label='train')
        ax1.plot(x, val_losses, 'g--', label='val')
        ax1.plot(x_test, test_losses, mark, label='test')

        ax2 = ax1.twinx()

        mark = 'b*' if single_run else 'b:'
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(x_test, test_acc, mark, label='test_accuracy')
        ax2.plot(x, val_acc, 'b--', label='val_accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        fig.tight_layout()
        fig.legend()
        pyplot.show()
        return best_val_model #, best_test_model

    def pick_data_format(self, data, labels, i, making_learning_curve, is_gru):
        train_data, test_data = data
        train_labels, test_labels = labels
        train_data, train_labels = self.shuffle_data(train_data, train_labels, i)

        using_fe = self.using_feature_extractor
        using_mask = is_gru and using_fe
        if making_learning_curve:
            tr_data = train_data[0][:i] if using_fe else train_data[:i]
            tr_data = [tr_data, train_data[1][:i]] if using_mask else tr_data
            tr_labels = train_labels[:i]
        else:
            tr_data = train_data[0] if using_fe else train_data
            tr_data = [tr_data, train_data[1]] if using_mask else tr_data
            tr_labels = train_labels
        tst_data = test_data[0] if using_fe else test_data
        tst_data = [tst_data, test_data[1]] if using_mask else tst_data
        return tr_data, tr_labels, tst_data, test_labels

    # def plot_data(self, history, test_history):

import os
import tensorflow as tf
import numpy as np
import cv2
import h5py
import imageio
import pathlib
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
import vidaug.augmentors as va

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


class JaiUtils:
    def __init__(self, vid_path, img_size, max_seq_len, train_split, learning_rate,
                 epochs, l2_reg, l1_reg, c, sigma, seed, training_data_updated):
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
        self.seed = seed
        self.training_data_updated = training_data_updated
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

    def load_or_process_video_data(self):
        try:
            if self.training_data_updated:
                raise IOError
            else:
                print("Attempting to load saved video data")
                f = h5py.File("tmp/data_file.hdf5", 'r')
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
            print("Processing all videos for network")
            trd, trl, tsd, tsl = self.prepare_all_videos()
            np.savez("tmp/prepared_test_videos.npz",
                     df=tsd[0],
                     dm=tsd[1],
                     dl=tsl,
                     )
            np.savez("tmp/prepared_train_videos.npz",
                     df=trd[0],
                     dm=trd[1],
                     dl=trl,
                     )

        return trd, trl, tsd, tsl

    def shuffle_data(self, data, labels):
        indices = np.arange(len(labels))
        indices = tf.random.shuffle(indices)
        feats = data[0].take(indices, axis=0)
        masks = data[1].take(indices, axis=0)
        labls = labels.take(indices, axis=0)
        return (feats, masks), labls

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
            # if len(new_frames) == self.frame_count:
            #     break
            if frame.shape[0:1] != self.img_size:
                new_frame = crop_center_square(frame)
                new_frame = cv2.resize(new_frame, self.img_size)
            else:
                new_frame = frame
            new_frames.append(new_frame)
        return np.array(new_frames)

    def spread_video(self, frames, step=1):
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

    def prepare_single_video(self, frames):
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

    def entry_collage(self, frames):
        return [frames[:]]

    def prepare_all_videos(self):
        # TODO: Refactor this
        self.build_feature_extractor()
        video_paths = self.data_index[:, 0]
        labels = self.data_index[:, 1]
        labels = self.label_processor(labels[..., None])
        labels = labels.numpy()

        m = labels.shape[0]
        # if m < 10:
        #     raise RuntimeError("Get more training examples")
        train_m = m // (1 / self.train_split)
        test_m = m // (1 / (1 - self.train_split))
        train_m = int(train_m + 1 if (m - train_m - test_m) > 0 else 0)
        indices = np.arange(m)
        indices = tf.random.shuffle(indices)
        tri, tsi = indices[:train_m], indices[train_m:]

        train_paths, train_labels = np.take(video_paths, tri), np.take(labels, tri)
        test_paths, test_labels = np.take(video_paths, tsi), np.take(labels, tsi)

        tmp_train_vids, tmp_train_lbls = [], []
        train_vids, train_lbls = [], []
        test_vids, test_lbls = [], []

        # end_size = self.augmentation_ops
        print(f"Augmenting data. This should result in TBD")
        for idx, (path, lbl) in enumerate(zip(train_paths, train_labels)):
            if idx % 10 == 0:
                print(f"Starting videos {idx}")
            frames = self.crop_and_resize_frames(self.load_video(path))
            tmp_train_vids.extend([frames])
            tmp_train_lbls.extend([lbl])
            augmented = self.augment_video(frames)
            tmp_train_vids.extend(augmented)
            tmp_train_lbls.extend(np.full(len(augmented), lbl))

        print(f"Spreading training data. This should result in TBD")
        for vid, lbl in zip(tmp_train_vids, tmp_train_lbls):
            if len(vid) <= self.frame_count:
                train_vids.extend([vid])
                train_lbls.extend([lbl])
            else:
                spread_vids = self.spread_video(vid, step=10)
                train_vids.extend(spread_vids)
                train_lbls.extend(np.full(len(spread_vids), lbl))

        training_data = np.array(train_vids)
        training_lbls = np.array(train_lbls)

        filename = f"data_file.hdf5"
        db = h5py.File("tmp/"+filename, "w")

        trds = db.create_dataset(
            name="train/data",
            shape=training_data.shape,
            dtype=np.uint8,
            chunks=True,
            data=training_data)
        trdl = db.create_dataset(
            name="train/labels",
            shape=training_lbls.shape,
            dtype=np.uint8,
            chunks=True,
            data=training_lbls)
        db.flush()
        # os.rename("tmp/"+filename, "sav/"+filename+"_AfterTrain")

        print(f"Spreading test data. This should result in TBD")

        for idx, (path, lbl) in enumerate(zip(test_paths, test_labels)):
            frames = self.crop_and_resize_frames(self.load_video(path))
            if len(frames) <= self.frame_count:
                test_vids.extend([frames])
                test_lbls.extend([lbl])
            else:
                spread_vids = self.spread_video(frames, 10)
                test_vids.extend(spread_vids)
                test_lbls.extend(np.full(len(spread_vids), lbl))

        test_data = np.array(test_vids)
        test_labels = np.array(test_lbls)

        tstd = db.create_dataset(
            name="test/data",
            shape=test_data.shape,
            dtype=np.uint8,
            chunks=True,
            data=test_data)

        tstl = db.create_dataset(
            name="test/labels",
            shape=test_labels.shape,
            dtype=np.uint8,
            chunks=True,
            data=test_labels)
        db.flush()
        # os.rename("tmp/"+filename, "sav/"+filename+"_AfterTest")

        train_frame_masks = np.zeros(
            shape=(training_data.shape[0], self.frame_count),
            dtype="bool")
        train_frame_features = np.zeros(
            shape=(training_data.shape[0], self.frame_count, self.num_features),
            dtype="float32")
        test_frame_masks = np.zeros(
            shape=(test_data.shape[0], self.frame_count),
            dtype="bool")
        test_frame_features = np.zeros(
            shape=(test_data.shape[0], self.frame_count, self.num_features),
            dtype="float32")

        print(f"Extracting training features This should result in TBD")
        # For each train video.
        for idx, vid in enumerate(train_vids):
            temp_frame_features, temp_frame_mask = self.prepare_single_video(vid.copy())
            train_frame_features[idx, ] = temp_frame_features.squeeze()
            train_frame_masks[idx, ] = temp_frame_mask.squeeze()

        preprocessed_training_features = db.create_dataset(
            name="train/features",
            chunks=True,
            dtype=np.float32,
            data=train_frame_features)
        preprocessed_training_masks = db.create_dataset(
            name="train/masks",
            chunks=True,
            dtype=np.bool,
            data=train_frame_masks)
        db.flush()

        print(f"Extracting test features This should result in TBD")
        # For each test video.
        for idx, vid in enumerate(test_vids):
            temp_frame_features, temp_frame_mask = self.prepare_single_video(vid)
            test_frame_features[idx, ] = temp_frame_features.squeeze()
            test_frame_masks[idx, ] = temp_frame_mask.squeeze()

        preprocessed_test_features = db.create_dataset(
            name="test/features",
            chunks=True,
            dtype=np.float32,
            data=test_frame_features)
        preprocessed_test_masks = db.create_dataset(
            name="test/masks",
            chunks=True,
            dtype=np.bool,
            data=test_frame_masks)
        db.flush()
        db.close()
        return (train_frame_features, train_frame_masks), labels, (test_frame_features, test_frame_masks), test_labels

    def get_gru_model(self, l2reg=None, learn_rate=None):
        l2reg = self.l2_reg if l2reg is None else l2reg
        learn_rate = self.learning_rate if learn_rate is None else learn_rate
        optimizer = tf.optimizers.SGD(learn_rate)
        # optimizer = tf.keras.optimizers.Adam(learn_rate)
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
        print(f"Test video label: {class_vocab[label]}")

        probabilities = model.predict((features, mask))[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        to_gif(features, "testing")
        return features

    def learning_rate_tuning_curve(self, data, get_model, metric, param_range, param_factor, is_gru=False, **kwargs):
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
        making_learning_curve = plot_x_label == "number of training examples"
        making_l2_curve = l2reg is None
        making_learn_rate_curve = learn_rate is None
        epochs = self.epochs if epochs is None else epochs

        train_data, test_data = data
        train_labels, test_labels = labels
        train_data, train_labels = self.shuffle_data(train_data, train_labels)
        history, test_history = {}, {}
        x, x_test = [], []
        test_acc, val_losses, test_losses, losses = [], [], [], []
        val_metrics, test_metrics, metrics = [], [], []
        best_val_model, best_test_model = None, None
        min_val_loss = 1e10
        min_test_loss = 1e10
        for i in range(*param_range):
            x_val = i*param_factor
            if making_learning_curve:
                tr_data = train_data[0][:i]
                tr_data = [tr_data, train_data[1][:i]] if is_gru else tr_data
                tr_labels = train_labels[:i]
            else:
                tr_data = train_data[0]
                tr_data = [tr_data, train_data[1]] if is_gru else tr_data
                tr_labels = train_labels
            tst_data = test_data[0]
            tst_data = [tst_data, test_data[1]] if is_gru else tst_data
            learn_rate = x_val if making_learn_rate_curve else learn_rate
            l2reg = x_val if making_l2_curve else l2reg
            x.append(x_val)
            x_test.append(x_val)
            print(f"Training with {plot_x_label}: {x_val}. Learning rate: {learn_rate}, l2reg: {l2reg}")
            tf.random.set_seed(self.seed)
            model = get_model(learn_rate=learn_rate, l2reg=l2reg)
            stop_early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=0.001,
                verbose=1
            )
            history[str(i)] = model.fit(
                tr_data,
                to_categorical(tr_labels),
                epochs=epochs,
                validation_split=0.2,
                verbose=1 if single_run else 0,
                callbacks=[stop_early]
            )
            test_history[str(i)] = model.evaluate(
                tst_data,
                to_categorical(test_labels)
            )
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

                test_acc = test_history[str(i)][2]

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

            test_acc.append(test_history[str(i)][2])
            if test_losses[-1] < min_test_loss:
                best_test_model = model
            if val_losses[-1] < min_val_loss:
                best_val_model = model

        if single_run:
            x = np.arange(0, len(losses))
            x_test = len(losses)
        pyplot.title(title)
        pyplot.xlabel(plot_x_label)
        pyplot.ylabel('cost ('+metric+')')
        pyplot.plot(x, losses, 'g--', label='train_reg')
        pyplot.plot(x, val_losses, 'g:', label='val_reg')
        pyplot.plot(x_test, test_losses, 'bD', label='test_reg')
        if making_l2_curve:
            pyplot.plot(x, metrics, 'r--', label='train_unreg')
            pyplot.plot(x, val_metrics, 'r:', label='val_unreg')
            pyplot.plot(x_test, test_metrics, 'c*', label='test_unreg')
        pyplot.legend()
        pyplot.show()
        return best_val_model #, best_test_model

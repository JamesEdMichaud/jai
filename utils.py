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
import time
from sklearn.metrics import roc_curve


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


def shuffle_and_split(data, labels, train_split, seed):
    indices = np.arange(len(labels))
    indices = tf.random.shuffle(indices)
    data = data.take(indices, axis=0)
    labels = labels.take(indices, axis=0)
    tri, tsi = get_split_indices(labels.shape[0], train_split, seed)
    trd, trl = np.take(data, tri), np.take(labels, tri)
    tsd, tsl = np.take(data, tsi), np.take(labels, tsi)
    return trd, trl, tsd, tsl


def get_split_indices(m, train_split, seed):
    train_m = m // (1 / train_split)
    test_m = m // (1 / (1 - train_split))
    train_m = int(train_m + (1 if (m - train_m - test_m) > 0 else 0))
    indices = np.arange(m+1)
    indices = tf.random.shuffle(indices, seed=seed)
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
            cv2.waitKey(100)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
    return np.array(frames)


def build_aug_list():
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


def to_grayscale(frames):
    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])


class JaiUtils:
    def __init__(self, vid_path, img_size, max_seq_len, train_split, val_split, learning_rate,
                 epochs, l2_reg, l1_reg, c, sigma, rand_seed, weights_seed, training_data_updated,
                 batch_size, using_augmentation):
        self.vid_path = vid_path
        self.img_size = img_size
        self.frame_count = max_seq_len
        self.train_split = train_split
        self.val_split = val_split
        self.learn_rate = learning_rate
        self.epochs = epochs
        self.l2reg = l2_reg
        self.l1_reg = l1_reg
        self.c = c
        self.sigma = sigma
        self.rand_seed = rand_seed
        self.weights_seed = weights_seed
        self.training_data_updated = training_data_updated
        self.batch_size = batch_size
        self.using_augmentation = using_augmentation
        self.learning_curve_x_label = "number of training examples"
        self.date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_index = build_data_index(self.vid_path)
        self.label_processor = self.init_label_processor()
        self.feature_extractor = None
        self.num_features = None
        self.augmentation_ops = build_aug_list()

    def init_label_processor(self):
        lp = tf.keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(self.data_index[:, 1])
        )
        print("Classes (labels): {}".format(lp.get_vocabulary()))
        return lp

    def get_label_set(self):
        return self.label_processor.get_vocabulary()

    def get_label_count(self):
        return len(self.label_processor.get_vocabulary())

    def get_data(self):
        filename = "data_file.hdf5"
        try:
            if self.training_data_updated:
                raise IOError("Updating data records")
            else:
                print("Attempting to load saved video data")
                db = h5py.File("tmp/"+filename, 'r')
                data, lbls, mapping = db['data'][...], db['lbls'][...], db['mapping'][...]
                aug_data, aug_lbls, aug_map = db['aug_data'][...], db['aug_lbls'][...], db['aug_map'][...]
                db.close()
                print("Using saved video data that has been processed")
        except IOError:
            if os.path.exists(os.path.join("tmp/", filename)):
                fr = os.path.join("tmp/", filename)
                to = "sav/" + self.date_str + "_"+filename
                print(f"Backing up: {filename} ==> {to}")
                shutil.move(fr, to)
            print("Processing all videos for network")
            data, lbls, aug_data, aug_lbls, mapping, aug_map = self.preprocess_videos()
            # data, lbls, mapping = self.spread_videos(data, lbls)
            # aug_data, aug_lbls, aug_map = self.spread_videos(aug_data, aug_lbls)
            print(f"Saving processed videos to tmp/{filename}")
            db = h5py.File("tmp/"+filename, "w")
            db.create_dataset(name="data", data=data, chunks=True, dtype=np.uint8)
            db.create_dataset(name="lbls", data=lbls, chunks=True, dtype=np.uint8)
            db.create_dataset(name="mapping", data=mapping, chunks=True, dtype=np.uint8)
            db.create_dataset(name="aug_map", data=aug_map, chunks=True, dtype=np.uint8)
            db.create_dataset(name="aug_data", data=aug_data, chunks=True, dtype=np.uint8)
            db.create_dataset(name="aug_lbls", data=aug_lbls, chunks=True, dtype=np.uint8)
            db.close()

        example_count = np.max(mapping)
        train_idx, test_idx = get_split_indices(example_count, self.train_split, self.rand_seed)

        trn_idx = np.squeeze(np.nonzero(np.in1d(mapping, train_idx)))
        tst_idx = np.squeeze(np.nonzero(np.in1d(mapping, test_idx)))
        trn_data, trn_lbls = np.take(data, trn_idx, axis=0), np.take(lbls, trn_idx, axis=0)
        tst_data, tst_lbls = np.take(data, tst_idx, axis=0), np.take(lbls, tst_idx, axis=0)
        print(f"Examples: {example_count+1} => {len(train_idx)}/{len(test_idx)} train/test")
        print(f"Examples including video spread: {len(trn_lbls)}/{len(tst_lbls)}")

        if self.using_augmentation:
            aug_idx = np.squeeze(np.nonzero(np.in1d(aug_map, train_idx)))
            aug_data, aug_lbls = np.take(aug_data, aug_idx, axis=0), np.take(aug_lbls, aug_idx, axis=0)
            trn_data, trn_lbls = np.concatenate((trn_data, aug_data)), np.concatenate((trn_lbls, aug_lbls))
            print(f"Examples including augmented data: {len(trn_lbls)}/{len(tst_lbls)}")

        all_data = {
            'train_data': trn_data,
            'train_labels': trn_lbls,
            'test_data': tst_data,
            'test_labels': tst_lbls
        }
        return all_data

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
        return np.array([frames[i:j] for i, j in zip(window_start, window_end)])

    def augment_video(self, frames):
        seqs = [va.Sequential(op if isinstance(op, list) else [op]) for op in self.augmentation_ops]
        return np.array([seq(frames) for seq in seqs])

    def load_videos(self, video_paths):
        return [self.crop_and_resize_frames(load_video(path)) for path in video_paths]

    def augment_videos(self, videos, labels):
        aug_vids, aug_lbls = [], []
        count = 0
        for idx, (vid, lbl) in enumerate(zip(videos, labels)):
            if idx % 10 == 0:
                print(f"On video {idx}")
            augmented = [to_grayscale(aug) for aug in self.augment_video(vid)]
            aug_vids.append(augmented)
            aug_lbls.append(np.full(len(augmented), lbl))
            count += len(augmented)
        return aug_vids, aug_lbls, count

    def spread_videos(self, vids, lbls, stacked=False):
        spread_vids, spread_lbls, mapping = [], [], []
        count = 0
        for idx, (vid, lbl) in enumerate(zip(vids, lbls)):
            if stacked:
                for aug, albl in zip(vid, lbl):
                    v = [aug] if len(aug) == self.frame_count else self.spread_video(aug, step=5)
                    l = [albl] if len(aug) == self.frame_count else np.full(len(v), albl)
                    i = [idx] if len(aug) == self.frame_count else np.full(len(v), idx)
                    spread_vids.extend(v)
                    spread_lbls.extend(l)
                    mapping.extend(i)
            else:
                v = [vid] if len(vid) == self.frame_count else self.spread_video(vid, step=5)
                l = [lbl] if len(vid) == self.frame_count else np.full(len(v), lbl)
                i = [idx] if len(vid) == self.frame_count else np.full(len(v), idx)
                count += 1 if len(vid) == self.frame_count else len(v)
                spread_vids.extend(v)
                spread_lbls.extend(l)
                mapping.extend(i)
        return np.array(spread_vids), np.array(spread_lbls), np.array(mapping)

    def preprocess_videos(self):
        print("Loading data from disk")
        videos = self.load_videos(self.data_index[:, 0])
        labels = self.label_processor(self.data_index[:, 1][..., None]).numpy()
        count = len(labels)
        print(f"Augmenting data. Initial count: {count}")
        aug_vids, aug_lbls, aug_count = self.augment_videos(videos, labels)
        count += aug_count
        print(f"Augmenting complete. New count: {count}")
        videos = [to_grayscale(vid) for vid in videos]
        videos, labels, mapping = self.spread_videos(videos, labels)
        aug_vids, aug_lbls, aug_map = self.spread_videos(aug_vids, aug_lbls, stacked=True)

        return videos, labels, aug_vids, aug_lbls, mapping, aug_map

    def shuffle_data(self, data, labels, i=0):
        indices = np.arange(len(labels))
        indices = tf.random.shuffle(indices, seed=self.rand_seed+i)
        return data.take(indices, axis=0), labels.take(indices, axis=0)

    def get_gru_model(self, l2reg=None, learn_rate=None):
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
            (self.frame_count, x.shape[2] * x.shape[3] * x.shape[4]),
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
                tf.keras.metrics.AUC(name="auc"),
            ]
        )
        return rnn_model

    def get_logistic_reg_model(self, l2reg=None, learn_rate=None):
        return self.get_basic_model(l2reg, learn_rate, 'categorical_crossentropy')

    def get_svm_model(self, l2reg=None, learn_rate=None):
        return self.get_basic_model(l2reg, learn_rate, "categorical_hinge")

    def get_basic_model(self, l2reg, learn_rate, loss_metric):
        optimizer = tf.optimizers.SGD(self.learn_rate if learn_rate is None else learn_rate)
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name="auc"),
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
        return self.frame_count, self.img_size[0], self.img_size[1], 1

    def prediction(self, model, data, label):
        class_vocab = self.get_label_set()
        print(f"Test video label: {class_vocab[label]}")

        probabilities = model.predict(data)[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        # to_gif(features, "testing")

    def get_split_string(self, data):
        m = len(data['train_labels'])
        m_test = len(data['test_labels'])
        return f"{round(m * self.train_split)}/{round(m * (1 - self.train_split))}/{m_test}"

    def learning_rate_tuning_curve(self, data, get_model, metric, epochs, param_range, param_factor, **kwargs):
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Param Tuning Curve (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="learning rate",
            param_range=param_range,
            param_factor=param_factor,
            epochs=epochs,
            l2reg=self.l2reg
        )

    def l2_tuning_curve(self, data, get_model, metric, param_range, param_factor, epochs, **kwargs):
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Param Tuning Curve (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="l2 regularization (lambda)",
            param_range=param_range,
            learn_rate=self.learn_rate,
            param_factor=param_factor,
            epochs=epochs
        )

    def learning_curve(self, data, get_model, metric, param_range, **kwargs):
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Learning curve, test examples: {len(data['test_labels'])}",
            plot_x_label=self.learning_curve_x_label,
            param_range=param_range,
            learn_rate=self.learn_rate,
            l2reg=self.l2reg,
            param_factor=1,
        )

    def loss_over_epochs(self, data, get_model, metric, epochs, **kwargs):
        return self.train_and_plot_curve(
            data=data,
            get_model=get_model,
            metric=metric,
            title=f"Loss (samples: {self.get_split_string(data)} train/val/test)",
            plot_x_label="epoch",
            learn_rate=self.learn_rate,
            l2reg=self.l2reg,
            param_range=[1],
            param_factor=1,
            epochs=epochs,
            single_run=True,
        )

    def train_and_plot_curve(self, data, get_model, metric, title, plot_x_label, param_range,
                             param_factor, learn_rate=None, l2reg=None, epochs=None, single_run=False):
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

            # learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=learn_rate*4,
            #     decay_steps=epochs*3,
            #     decay_rate=0.9,
            # )
            stop_early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                min_delta=0.0001,
                verbose=1
            )
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
                # callbacks=[stop_early],
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
            # train_predictions_baseline = best_val_model.predict(trn_data, batch_size=self.batch_size)
            # test_predictions_baseline = best_val_model.predict(tst_data, batch_size=self.batch_size)
            # self.plot_roc("train baseline", trn_lbls, train_predictions_baseline)
            # self.plot_roc("test baseline", tst_labels, test_predictions_baseline)

        if single_run:
            self.plot_run(history['0'], test_history['0'], metric, title, plot_x_label)
        else:
            self.plot_results(history, test_history, metric, title, x, x_test, plot_x_label)
        print(f"Best metric from this run: {best_metric}")
        return best_val_model

    def plot_run(self, history, test_history, metric, title, x_label):
        args = {
            'title': title,
            'x_label': x_label,
            'metric': metric,
            'trn_losses': history.history['loss'],
            'val_losses': history.history['val_loss'],
            'trn_metrics': history.history[metric],
            'val_metrics': history.history['val_'+metric],
            'val_acc': history.history['val_accuracy'],
            'test_losses': test_history[0],
            'test_metrics': test_history[1],
            'test_acc': test_history[2],
            'auc': test_history[3],
            'x': np.arange(0, len(history.epoch)),
            'x_test': len(history.epoch),
            'test_mark': '*'
        }
        self.plot_data(**args)

    def plot_results(self, history, test_history, metric, title, x, x_test, x_label):
        args = {
            'title': title,
            'x': x,
            'x_test': x_test,
            'x_label': x_label,
            'metric': metric,
            'trn_losses': [],
            'val_losses': [],
            'trn_metrics': [],
            'val_metrics': [],
            'val_acc': [],
            'test_losses': [],
            'test_acc': [],
            'test_metrics': [],
            # 'auc': [],
            'test_mark': ':'
        }
        for run in history.values():
            args['trn_metrics'].append(np.mean(run.history[metric][-10:]))
            args['val_metrics'].append(np.mean(run.history['val_'+metric][-10:]))
            args['trn_losses'].append(np.mean(run.history['loss'][-10:]))
            args['val_losses'].append(np.mean(run.history['val_loss'][-10:]))
            args['val_acc'].append(np.mean(run.history['val_accuracy']))
            args['auc'].append(np.mean(run.history['auc']))
        for run in test_history.values():
            args['test_losses'].append(run[0])
            args['test_metrics'].append(run[1])
            args['test_acc'].append(run[2])
            args['auc'].append(run[3])
        self.plot_data(**args)

    def plot_data(self, title, x, x_label, metric, trn_metrics, val_metrics, test_mark,
                  x_test, test_metrics, trn_losses, val_losses, test_losses, test_acc, val_acc):
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
        fig.tight_layout()
        fig.legend()
        pyplot.show()

    def plot_roc(self, name, labels, predictions):
        fp, tp, _ = roc_curve(labels, predictions)
        pyplot.plot(100 * fp, 100 * tp, label=name, linewidth=2)
        pyplot.xlabel('False positives [%]')
        pyplot.ylabel('True positives [%]')
        pyplot.xlim([-0.5, 20])
        pyplot.ylim([80, 100.5])
        pyplot.grid(True)
        ax = pyplot.gca()
        ax.set_aspect('equal')

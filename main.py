import numpy as np
import tensorflow as tf
# Disable GPU on Apple architecture. It's slower sometimes.
# tf.config.set_visible_devices([], 'GPU')
from utils import JaiUtils
from models import JaiNN, JaiLR, JaiSVM
from cam_viewer import JaiCam
from cam_model import JaiCam2
import cv2

# print("TensorFlow version: ", tf.__version__)

# run_type = "Live"
# run_type = "NeuralNet"
# run_type = "Test"
run_type = "LogReg"
# run_type = "SVM"
# run_type = "Motion"

random_seed = 3232
weights_random_seed = 438

utils = JaiUtils(
    vid_path="training_data",
    img_size=(256, 256),
    max_seq_len=40,
    train_split=0.9,
    val_split=0.2,
    l1_reg=0.0,
    c=1,
    sigma=0.1,
    rand_seed=random_seed,
    weights_seed=weights_random_seed,
    epochs=300,
    # Found using parameter tuning curve:
    # Logistic Regression: ~0.0025
    # SVM: ???
    # Neural Network: 0.01 -- ??
    learning_rate=0.1,
    # Found using parameter tuning curve:
    # Logistic Regression: ~0.0025
    # SVM: ???
    # Neural Network: ???
    l2_reg=0.01,
    training_data_updated=True,
    batch_size=64,
    # using_feature_extractor=True,
    using_feature_extractor=False,
    using_augmentation=True,
)

if run_type.casefold() in ["neuralnet", "logreg", "svm"]:
    data = utils.load_or_preprocess_video_data()
    if utils.using_feature_extractor:
        print(f"Frame features in train set: {data[0][0].shape}")
        print(f"Frame masks in train set: {data[0][1].shape}")
    else:
        print(f"Training data shape: {data[0].shape}")
        print(f"Test data shape: {data[2].shape}")

    if run_type.casefold() == "neuralnet":
        model = JaiNN(utils)
    elif run_type.casefold() == 'logreg':
        model = JaiLR(utils)
    elif run_type.casefold() == 'svm':
        model = JaiSVM(utils)
    else:
        model = None
        raise IOError("invalid model type")

    args = {
        'data': data,
        'method_to_call': utils.loss_over_epochs,
        # 'method_to_call': utils.learning_rate_tuning_curve,
        # 'method_to_call': utils.l2_tuning_curve,
        # 'method_to_call': utils.learning_curve,
        'epochs': 1000,
        'param_range': [3, 900, 30],
        'param_factor': 0.0001
    }
    tf.random.set_seed(random_seed)
    model.prepare_and_run(**args)

    test_data, test_labels = data[2], data[3]
    tf.random.set_seed(random_seed)
    rand = tf.random.shuffle(np.arange(len(test_labels)))[0]
    if utils.using_feature_extractor:
        pred_input = (test_data[0][rand], test_data[1][rand])
    else:
        pred_input = test_data[rand]
    utils.prediction(model, pred_input, test_labels[rand])

else:

    if run_type.casefold() == "live":
        model = JaiCam2(utils, is_interactive=True)
        model.start_video_feed()
        # model.start_video_feed("testin.avi")
        while model.cam_is_open():
            model.next_frame()
            if (cv2.waitKey(100 if model.frame_counter > 70 else 1) & 0xFF) == ord('q'):
                model.end_video_feed()
        print("Finished with strm run")

    elif run_type.casefold() == "motion":
        print("'Motion' run type not ready. Choose another run type")
        # This one's very unfinished. Don't use

        # model = JaiCam(utils, is_interactive=True)
        # model.train_over_all_data()
        # model.start_video_feed()
        # while model.cam_is_open():
        #     model.iterate()

        # model.run_experiment((train_data, test_data), (train_labels, test_labels), EPOCHS)
        # utils.prediction(model.gru_model, np.random.choice(utils.data_index[:, 0]))
    elif run_type.casefold() == 'test':
        all_vids = utils.load_or_preprocess_video_data()
        print(len(all_vids))

        # frames = utils.crop_and_resize_frames(utils.load_video("testin.avi"))
        # frames = utils.load_video("testin.avi")
        # augmented = utils.augment_video(frames)
        # cols = 6
        # rows = augmented.shape[0] // cols + 1
        # frame_count = augmented.shape[1]
        # height = augmented.shape[2]*rows
        # width = augmented.shape[3]*cols
        # channels = augmented.shape[4]
        #
        # shape = (frame_count, height, width, channels)
        #
        # collage_frames = np.zeros(shape=shape, dtype=np.int8)
        # for i in range(augmented.shape[1]):        # for each frame
        #     for j in range(augmented.shape[0]):     # for each augment
        #
        #         collage_frames[i][j//cols ,]
        #         cv2.imshow(f"aug {j}", augmented[j][i])
        #         hspace = 8
        #         vspace = 23
        #         cv2.moveWindow(f"aug {j}",
        #                        j % row_size * (utils.img_size[0]+hspace),
        #                        j // row_size * (utils.img_size[1]+vspace))
        #     if (cv2.waitKey(100) & 0xFF) == ord('q'):
        #         break
        #
        # zero_to_49 = np.arange(50)
        # vids = utils.spread_video(zero_to_49)
        # for vid in vids:
        #     print(vid)
    # TODO: Define networks to handle the following:
    #       background vs event (binary) - work in progress
    #       entry vs. non-entry (binary)
    #       Define a network that detects person entering (if entry)
    #       Define a network that detects event type (if non-entry) e.g. package delivery

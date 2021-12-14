import numpy as np
import tensorflow as tf
# Disable GPU on Apple architecture. It's slower.
tf.config.set_visible_devices([], 'GPU')
from utils import JaiUtils
from models import JaiNN, JaiLR, JaiSVM
from cam_viewer import JaiCam
from cam_model import JaiCam2
import cv2

# print("TensorFlow version: ", tf.__version__)

user      = "jai"
pswd      = "oPC9Hxt3DZsMXp8bmap"
ip        = "192.168.1.201"
port      = "554"
frmt      = "h264Preview_01_"
mainOrSub = "sub"
streamAddress = "rtsp://"+user+":"+pswd+"@"+ip+":"+port+"//"+frmt+mainOrSub
print("Stream address: {}".format(streamAddress))

# run_type = "Live"
# run_type = "NeuralNet"
# run_type = "Test"
run_type = "LogReg"
# run_type = "SVM"
# run_type = "Motion"

random_seed = 555
weights_random_seed = 638

utils = JaiUtils(
    vid_path="training_data",
    img_size=(256, 256),
    max_seq_len=40,
    train_split=0.9,
    learning_rate=0.001,
    epochs=300,
    l2_reg=0.001,
    l1_reg=0.0,
    c=1,
    sigma=0.1,
    seed=weights_random_seed,
    training_data_updated=False
)

if run_type.casefold() == 'test':
    all_vids = utils.load_or_process_video_data()
    print(all_vids)

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

elif run_type.casefold() == "live":
    model = JaiCam2(utils, is_interactive=True)
    model.start_video_feed(streamAddress)
    # model.start_video_feed("testin.avi")
    while model.cam_is_open():
        model.next_frame()
        if (cv2.waitKey(100 if model.frame_counter > 70 else 1) & 0xFF) == ord('q'):
            model.end_video_feed()
    print("Finished with strm run")

else:
    if run_type.casefold() != "cam":
        tf.random.set_seed(random_seed)
        data = utils.load_or_process_video_data()
        test_data, test_labels = data[2], data[3]

        print(f"Frame features in train set: {data[0][0].shape}")
        print(f"Frame masks in train set: {data[0][1].shape}")

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
            # 'method_to_call': utils.loss_over_epochs,
            'method_to_call': utils.learning_rate_tuning_curve,
            # 'method_to_call': utils.l2_tuning_curve,
            # 'method_to_call': utils.learning_curve,
            'epochs': 300,
            'param_range': [0, 100, 2],
            'param_factor': 0.00001
        }
        tf.random.set_seed(random_seed)
        model.prepare_and_run(**args)

        tf.random.set_seed(random_seed)
        rand = tf.random.shuffle(np.arange(len(test_labels)))[0]

        utils.prediction(model, test_data[0][rand], test_data[1][rand], test_labels[rand])
    elif run_type.casefold() == "motion":
        print("'Motion' run type not ready. Choose another run type")
        # This one's very unfinished. Don't use

        model = JaiCam(utils, is_interactive=True)
        # model.train_over_all_data()
        # model.start_video_feed()
        # while model.cam_is_open():
        #     model.iterate()

        # model.run_experiment((train_data, test_data), (train_labels, test_labels), EPOCHS)
        # utils.prediction(model.gru_model, np.random.choice(utils.data_index[:, 0]))

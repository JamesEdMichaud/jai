import numpy as np
import tensorflow as tf
from utils import JaiUtils
from nn import JaiNN
from lr import JaiLR
from svm import JaiSVM
from cam_viewer import JaiCam
from models import JaiCam2
import cv2

print("TensorFlow version: ", tf.__version__)

user      = "jai"
pswd      = "oPC9Hxt3DZsMXp8bmap"
ip        = "192.168.1.201"
port      = "554"
frmt      = "h264Preview_01_"
mainOrSub = "sub"
streamAddress = "rtsp://"+user+":"+pswd+"@"+ip+":"+port+"//"+frmt+mainOrSub
print("Stream address: {}".format(streamAddress))

run_type = "strm"
utils = JaiUtils(
    vid_path="frameCaptures",
    img_size=(256, 256),
    max_seq_len=40,
    train_split=0.9,
    learning_rate=0.01,
    l2_reg=0.1,
    l1_reg=0.0,
    c=1,
    sigma=0.1
)

# TODO: Define networks to handle the following:
# TODO: background vs event (binary) - work in progress
# TODO: entry vs. non-entry (binary)
# TODO: Define a network that detects person entering
# TODO: Define a network that detects

if run_type == "strm":
    model = JaiCam2(utils, is_interactive=True)
    # model.start_video_feed(streamAddress)
    model.start_video_feed("testin.avi")
    while model.cam_is_open():
        model.next_frame()
        if (cv2.waitKey(150 if model.frame_counter > 70 else 1) & 0xFF) == ord('q'):
            model.end_video_feed()
    print("Finished with strm run")

else:
    if run_type != "cam":
        data = utils.load_or_process_video_data()
        train_data, train_labels, test_data, test_labels = data

        print(f"Frame features in train set: {train_data[0].shape}")
        print(f"Frame masks in train set: {train_data[1].shape}")

        if run_type == "nn":
            model = JaiNN(utils)
            model.run_experiment(
                data=(train_data, test_data),
                labels=(train_labels, test_labels),
                epochs=300
            )
            utils.prediction(model.gru_model, np.random.choice(utils.data_index[:, 0]))
        elif run_type == 'lr':
            model = JaiLR(utils)

            model.run_experiment(
                data=(train_data, test_data),
                labels=(train_labels, test_labels),
                epochs=300//10
            )
            # model = model.train(train_data, train_labels, EPOCHS)
            utils.prediction(model, np.random.choice(utils.data_index[:, 0]))
        elif run_type == 'svm':
            model = JaiSVM(utils)
            model.run_experiment(
                data=(train_data, test_data),
                labels=(train_labels, test_labels),
                epochs=300
            )
            # model = model.train(train_data, train_labels, EPOCHS)
            utils.prediction(model, np.random.choice(utils.data_index[:, 0]))
    elif run_type == "cam":
        model = JaiCam(utils, is_interactive=True)
        model.train_over_all_data()
        # model.start_video_feed()
        # while model.cam_is_open():
        #     model.iterate()

        # model.run_experiment((train_data, test_data), (train_labels, test_labels), EPOCHS)
        # utils.prediction(model.gru_model, np.random.choice(utils.data_index[:, 0]))

import os
from collections import deque
from pathlib import Path
from datetime import datetime
import cv2
import imutils
from utils import crop_center_square
from utils import build_data_index
import numpy as np



# outfile = "testout" + date_str + ".avi"
framerate = 7

# maxFrameHistory = 30
# minArea = 1500
# isFirstIter = True
#
# capturingFrames = False
#
# vcap = None
# backSub = None
#
#
# capCounter = 0
# bgCounter = 0
# postEventFrameCounter = 0
# motionEventFrameCounter = 0
# inMotionEvent = False
#
# motionEventFrames = []
# frameHistory = deque(maxlen=maxFrameHistory)


class JaiCam:
    def __init__(self, utils, is_interactive=False):
        self.utils = utils
        self.is_interactive = is_interactive
        self.frameCounter = 0
        self.frame_capture_counter = 0
        self.min_loss = 1000000
        self.using_video_feed = True
        self.current_frame = None
        self.next_frame = None
        self.vcap = None
        self.writer = None

        # self.backSub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=24, detectShadows=False)
        # self.backSub = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=1000.0, detectShadows=False)

        print("Getting the motion model")
        self.model = utils.get_motion_model()
        self.key_pressed = False

    def train_over_all_data(self):
        self.using_video_feed = False
        video_paths = self.utils.data_index[:, 0]
        video_paths = np.append(video_paths, build_data_index("frameCapturesV2")[:, 0])

        frames = self.utils.load_video(video_paths[0])
        for idx, path in enumerate(video_paths):
            if idx == 0:
                continue
            curr = self.utils.load_video(path)
            frames = np.append(frames, curr, axis=0)
        np.random.shuffle(frames)
        for idy, frame in enumerate(frames):
            self.next_frame = frame
            self.iterate()

    def get_next_frame(self):
        if self.using_video_feed:
            return self.vcap.read()
        else:
            return True, self.next_frame

    def start_video_feed(self):
        print("Instantiating VideoCapture Object")
        self.vcap = cv2.VideoCapture("testin.avi")
        # self.vcap = cv2.VideoCapture(streamAddress)

        # print("Instantiating VideoWriter Object")
        # self.writer = cv2.VideoWriter(
        #     outfile, cv2.VideoWriter_fourcc(*"MJPG"),
        #     framerate, self.utils.img_size
        # )
        if self.is_interactive:
            print("Starting video feed")
            cv2.namedWindow("main")
            # cv2.namedWindow("gy")
            # cv2.namedWindow("mask1")
            # cv2.namedWindow("mask2")
            # cv2.namedWindow("masked")
            # cv2.namedWindow("event")
            cv2.moveWindow("main", 0, 0)
            # res = self.utils.img_size
            # cv2.moveWindow("gy", 0, res[1] + 45)
            # cv2.moveWindow("mask1", res[0], 0)
            # cv2.moveWindow("mask2", res[0], res[1] + 45)
            # cv2.moveWindow("masked", res[0] * 2, 0)
            # cv2.moveWindow("event", res[0] * 2, res[1] + 45)

    def cam_is_open(self):
        return self.vcap.isOpened()

    def key_pressed(self):
        return self.key_pressed

    def test(self, y_true):
        return self.model.test_on_batch(self.current_frame[None, ...], y_true)

    def train(self, y_true):
        return self.model.train_on_batch(self.current_frame[None, ...], y_true)

    def predict(self):
        return self.model.predict_on_batch(self.current_frame[None, ...])

    def iterate(self):
        # print("Starting iterate method")
        text = "No Motion"
        hasFrame, frameOrig = self.get_next_frame()
        if hasFrame:
            res = self.utils.img_size
            # writer.write(frame)
            self.frameCounter += 1
            if frameOrig.shape != res:
                frame = crop_center_square(frameOrig)
                frame = cv2.resize(frame, res) if frame.shape != res else frame
            else:
                frame = frameOrig
            self.current_frame = np.array(frame)

            y = self.predict()[0]
            y_text = "No Motion" if y[0] > y[1] else "Motion"
            display_frame = frame.copy()
            cv2.putText(display_frame, "Status: {}".format(y_text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('main', display_frame        )
            print("Prediction: {}. Is there motion in this frame? y/n".format(y_text))
            key_pressed = cv2.waitKey(100000) & 0xFF
            if key_pressed == ord('y'):
                y_true = [0, 1]
            elif key_pressed == ord('n'):
                y_true = [1, 0]
            else:
                print("Incorrect key. Assumed No motion")
                y_true = [1, 0]
            run_dir = "./motion_training/"+self.utils.date_str
            save_dir = run_dir+"/"+str(np.argmax(y_true))
            os.makedirs(save_dir, exist_ok=True)
            save_string = save_dir+"/frame"+str(self.frame_capture_counter)+".png"
            cv2.imwrite(save_string, frameOrig)
            self.frame_capture_counter += 1
            loss, acc = self.test(np.array([y_true]))
            if loss > 1:
                print("Loss: {}, accuracy: {}. Training on this example".format(loss, acc))
                loss, acc = self.train(np.array([y_true]))
            if loss < self.min_loss:
                self.min_loss = loss
                print("Best loss so far. Saving model")
                self.model.save(run_dir+"/motion_model")
            # y = self.model.predict_on_batch(self.current_frame[None, ...])
            # return y
    #         gy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Needed to check - is BGR
    #         gy = cv2.GaussianBlur(gy, (21, 21), 0)
    #         cv2.imshow('gy', gy)
    #
    #         fgMask = self.backSub.apply(gy)
    #         cv2.imshow('mask1', fgMask)
    #         fgMask = cv2.dilate(fgMask, None, iterations=2)
    #         cv2.imshow('mask2', fgMask)
    #
    #         # Contours will be used to generate bounding box
    #         contours = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         contours = imutils.grab_contours(contours)
    #         for cont in contours:
    #             if cv2.contourArea(cont) > minArea:
    #                 (x, y, w, h) = cv2.boundingRect(cont)
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #                 text = "Motion"
    #         cv2.putText(frame, "Status: {}".format(text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #         cv2.putText(frame, "Frame: {}".format(frameCounter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
    #                     2)
    #
    #         fore = cv2.bitwise_and(frame, frame, mask=fgMask)
    #         cv2.drawContours(fore, contours, -1, (255, 0, 0), 2)
    #         cv2.imshow('main', frame)
    #         cv2.imshow('masked', fore)
    #         if isFirstIter:
    #             # print(frame.shape, type(frame), frame.ndim, frame.size, len(frame))
    #             # cv2.imwrite("tempFrame"+runNameString+".png", frame)
    #             isFirstIter = False
    #
    #         # Decision tree that handles the frame buffer and saving motion events.
    #         if inMotionEvent:
    #             cv2.imshow('event', frameOrig)
    #             motionEventFrameCounter += 1
    #             if text == "Motion":
    #                 motionEventFrames.append(frameOrig)
    #                 # TODO: Maybe classify here? (if per frame)
    #             else:  # No Motion
    #                 if postEventFrameCounter < maxFrameHistory / 2:
    #                     # Keep collecting frames until max is reached
    #                     motionEventFrames.append(frameOrig)
    #                     postEventFrameCounter += 1
    #                 else:  # Max reached. Save motion event and reset counters
    #                     # TODO: Or classify here? (if per video clip)
    #                     pth = "./frameCaptures/" + date_str + "/" + str(capCounter)
    #                     print("Saving motion event frames to: {}".format(pth))
    #                     Path(pth).mkdir(parents=True, exist_ok=True)
    #                     for i, motionFrame in enumerate(motionEventFrames, start=1):
    #                         cv2.imwrite(pth + "/" + str(i) + ".png", motionFrame)
    #                     capCounter += 1
    #                     inMotionEvent = False
    #                     motionEventFrameCounter = 0
    #                     postEventFrameCounter = 0
    #                     motionEventFrames.clear()
    #         else:  # Not in motion event
    #             if text == "Motion":
    #                 motionEventFrameCounter += 1
    #                 if motionEventFrameCounter > 5:
    #                     inMotionEvent = True
    #                     motionEventFrames = list(frameHistory)[-3:]
    #                     motionEventFrames.append(frameOrig)
    #             else:  # No motion
    #                 motionEventFrameCounter = 0
    #                 postEventFrameCounter = 0
    #
    #         frameHistory.append(frameOrig)
    #         # if capturingFrames and frameCounter > 4:
    #         #     if text == "Motion":
    #         #         capCounter += 1
    #         #         cv2.imwrite("./frameCaptures/motion_" + runNameString + "_" + str(capCounter) + ".png", frameOrig)
    #         #         print("Saving unlabeled training image")
    #         #     elif bgCounter < capCounter and text == "No Motion" and frameCounter % 7 == 0:
    #         #         bgCounter += 1
    #         #         cv2.imwrite("./frameCaptures/noMotion_" + runNameString + "_" + str(bgCounter) + ".png", frameOrig)
    #         if (cv2.waitKey(150 if frameCounter > 70 else 1) & 0xFF) == ord('q'):
    #             break
    #     else:
    #         print('Cannot read video file/stream')
    #         break
    #
    # print("Releasing video file/stream")
    # vcap.release()
    # print("Destroying all cv2 windows")
    # cv2.destroyAllWindows()








 #    while vcap.isOpened():
 #        text = "No Motion"
 #        hasFrame, frameOrig = self.vcap.read()
 #        if hasFrame:
 #            # writer.write(frame)
 #            self.frameCounter += 1
 #
 #            frame = crop_center_square(frameOrig)
 #            frame = cv2.resize(frame, res) if frame.shape != res else frame
 #
 #            gy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Needed to check - is BGR
 #            gy = cv2.GaussianBlur(gy, (21, 21), 0)
 #            cv2.imshow('gy', gy)
 #
 #            fgMask = self.backSub.apply(gy)
 #            cv2.imshow('mask1', fgMask)
 #            fgMask = cv2.dilate(fgMask, None, iterations=2)
 #            cv2.imshow('mask2', fgMask)
 #
 #            # Contours will be used to generate bounding box
 #            contours = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 #            contours = imutils.grab_contours(contours)
 #            for cont in contours:
 #                if cv2.contourArea(cont) > minArea:
 #                    (x, y, w, h) = cv2.boundingRect(cont)
 #                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 #                    text = "Motion"
 #            cv2.putText(frame, "Status: {}".format(text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 #            cv2.putText(frame, "Frame: {}".format(frameCounter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 #
 #            fore = cv2.bitwise_and(frame, frame, mask=fgMask)
 #            cv2.drawContours(fore, contours, -1, (255, 0, 0), 2)
 #            cv2.imshow('main', frame)
 #            cv2.imshow('masked', fore)
 #            if isFirstIter:
 #                # print(frame.shape, type(frame), frame.ndim, frame.size, len(frame))
 #                # cv2.imwrite("tempFrame"+runNameString+".png", frame)
 #                isFirstIter = False
 #
 #            # Decision tree that handles the frame buffer and saving motion events.
 #            if inMotionEvent:
 #                cv2.imshow('event', frameOrig)
 #                motionEventFrameCounter += 1
 #                if text == "Motion":
 #                    motionEventFrames.append(frameOrig)
 #                    # TODO: Maybe classify here? (if per frame)
 #                else:  # No Motion
 #                    if postEventFrameCounter < maxFrameHistory/2:
 #                        # Keep collecting frames until max is reached
 #                        motionEventFrames.append(frameOrig)
 #                        postEventFrameCounter += 1
 #                    else:  # Max reached. Save motion event and reset counters
 #                        # TODO: Or classify here? (if per video clip)
 #                        pth = "./frameCaptures/" + date_str + "/" + str(capCounter)
 #                        print("Saving motion event frames to: {}".format(pth))
 #                        Path(pth).mkdir(parents=True, exist_ok=True)
 #                        for i, motionFrame in enumerate(motionEventFrames, start=1):
 #                            cv2.imwrite(pth+"/"+str(i)+".png", motionFrame)
 #                        capCounter += 1
 #                        inMotionEvent = False
 #                        motionEventFrameCounter = 0
 #                        postEventFrameCounter = 0
 #                        motionEventFrames.clear()
 #            else:  # Not in motion event
 #                if text == "Motion":
 #                    motionEventFrameCounter += 1
 #                    if motionEventFrameCounter > 5:
 #                        inMotionEvent = True
 #                        motionEventFrames = list(frameHistory)[-3:]
 #                        motionEventFrames.append(frameOrig)
 #                else:  # No motion
 #                    motionEventFrameCounter = 0
 #                    postEventFrameCounter = 0
 #
 #            frameHistory.append(frameOrig)
 #            # if capturingFrames and frameCounter > 4:
 #            #     if text == "Motion":
 #            #         capCounter += 1
 #            #         cv2.imwrite("./frameCaptures/motion_" + runNameString + "_" + str(capCounter) + ".png", frameOrig)
 #            #         print("Saving unlabeled training image")
 #            #     elif bgCounter < capCounter and text == "No Motion" and frameCounter % 7 == 0:
 #            #         bgCounter += 1
 #            #         cv2.imwrite("./frameCaptures/noMotion_" + runNameString + "_" + str(bgCounter) + ".png", frameOrig)
 #            if (cv2.waitKey(150 if frameCounter > 70 else 1) & 0xFF) == ord('q'):
 #                break
 #        else:
 #            print('Cannot read video file/stream')
 #            break
 #
 #    print("Releasing video file/stream")
 #    vcap.release()
 #    print("Destroying all cv2 windows")
 #    cv2.destroyAllWindows()


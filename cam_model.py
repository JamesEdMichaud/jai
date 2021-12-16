import cv2
import numpy as np
import imutils
from pathlib import Path
from collections import deque
from utils import crop_center_square

user      = "jai"
pswd      = "oPC9Hxt3DZsMXp8bmap"
ip        = "192.168.1.201"
port      = "554"
frmt      = "h264Preview_01_"
mainOrSub = "sub"
streamAddress = "rtsp://"+user+":"+pswd+"@"+ip+":"+port+"//"+frmt+mainOrSub

class JaiCam2:
    def __init__(self, utils, is_interactive=False):
        print("Stream address: {}".format(streamAddress))
        self.utils = utils
        self.is_interactive = is_interactive
        self.min_area = 2000
        self.in_motion_event = False
        self.motion_event_frame_counter = 0
        self.post_event_frame_counter = 0
        self.cap_counter = 0
        self.error_count = 0
        self.frame_counter = 0
        self.max_frame_history = 40
        self.framerate = 7
        self.motion_event_frames = []
        self.frame_history = deque(maxlen=self.max_frame_history)
        self.background_frames = deque(maxlen=self.max_frame_history)
        self.vcap = None
        self.writer = None
        self.using_optical_flow = False
        self.prev = None
        self.prev_is_initialized = False
        self.hsv = None
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=24, detectShadows=False)
        # self.backSub = cv2.createBackgroundSubtractorGMG()
        # self.backSub = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=1000.0, detectShadows=False)

    def cam_is_open(self):
        return self.vcap.isOpened()

    def start_video_feed(self, video):
        print("Instantiating VideoCapture Object")
        self.vcap = cv2.VideoCapture(video)

        if self.is_interactive:
            print("Starting video feed")
            cv2.namedWindow("main")
            cv2.namedWindow("background_average")
            cv2.namedWindow("mask")
            cv2.namedWindow("frame_delta")
            cv2.namedWindow("masked")
            cv2.namedWindow("event")
            cv2.moveWindow("main", 0, 0)
            res = self.utils.img_size
            cv2.moveWindow("background_average", 0, res[1] + 45)
            cv2.moveWindow("mask", res[0], 0)
            cv2.moveWindow("frame_delta", res[0], res[1] + 45)
            cv2.moveWindow("masked", res[0] * 2, 0)
            cv2.moveWindow("event", res[0] * 2, res[1] + 45)
            if self.using_optical_flow:
                cv2.namedWindow("flow")
                cv2.moveWindow("flow", 0, (res[1] + 45)*2)

    def update_optical_flow(self):
        frame = self.utils.current_frame
        if self.prev_is_initialized:
            nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.prev, next=nxt, flow=None,
                pyr_scale=0.5,
                levels=7,
                winsize=15,
                iterations=9,
                poly_n=5,
                poly_sigma=1.1,
                flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.hsv[..., 0] = ang * 180 / np.pi / 2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('flow', bgr)
            self.prev = nxt
        else:
            self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            self.prev_is_initialized = True

    def preprocess_frame(self, original_frame):
        res = self.utils.img_size
        if original_frame.shape != res:
            frame = crop_center_square(original_frame)
            frame = np.array(cv2.resize(frame, res) if frame.shape != res else frame)
        else:
            frame = np.array(original_frame)
        return frame, frame.copy()

    def get_foreground_mask(self):
        # Needed to check - captured frames are BGR
        gy = cv2.cvtColor(self.utils.current_frame, cv2.COLOR_BGR2GRAY)
        gy = cv2.GaussianBlur(gy, (21, 21), 0)
        if not self.in_motion_event:
            self.background_frames.append(gy)
            avg = np.mean(list(self.background_frames), axis=0).astype(np.uint8)
        else:
            frms = list(self.background_frames)
            frms.append(gy)
            avg = np.mean(frms, axis=0).astype(np.uint8)
        cv2.imshow('background_average', avg)
        frame_delta = cv2.absdiff(avg, gy)
        thresh = cv2.threshold(frame_delta, 60, 255, cv2.THRESH_BINARY)[1]

        # fg_mask = self.backSub.apply(gy)
        fg_mask = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow('mask', fg_mask)
        cv2.imshow('frame_delta', frame_delta)
        return fg_mask

    def next_frame(self):
        text = "No Motion"
        hasFrame, frameOrig = self.vcap.read()
        if hasFrame:
            frame, display_frame = self.preprocess_frame(frameOrig)

            self.frame_counter += 1
            self.utils.current_frame = frame
            self.utils.current_frame_orig = frameOrig

            if self.using_optical_flow:
                self.update_optical_flow()

            fg_mask = self.get_foreground_mask()

            # Contours will be used to generate bounding box
            contours = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)
            for cont in contours:
                if cv2.contourArea(cont) > self.min_area:
                    (x, y, w, h) = cv2.boundingRect(cont)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "Motion"
            cv2.putText(display_frame, "Status: {}".format(text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(display_frame, "Frame: {}".format(frameCounter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fore = cv2.bitwise_and(frame, frame, mask=fg_mask)
            cv2.drawContours(fore, contours, -1, (255, 0, 0), 2)
            cv2.imshow('main', display_frame)
            cv2.imshow('masked', fore)
            # Decision tree that handles the frame buffer and saving motion events.
            if self.in_motion_event:
                cv2.imshow('event', frameOrig)
                self.motion_event_frame_counter += 1
                if text == "Motion":
                    self.motion_event_frames.append(frameOrig)
                    # TODO: Maybe classify here? (if per frame)
                else:  # No Motion
                    if self.post_event_frame_counter < self.max_frame_history / 2:
                        # Keep collecting frames until max is reached
                        self.motion_event_frames.append(frameOrig)
                        self.post_event_frame_counter += 1
                    else:  # Max reached. Save motion event and reset counters
                        # TODO: Or classify here? (if per video clip)
                        print("Instantiating VideoWriter Object")
                        pth = "./vidEventCaptures/" + self.utils.date_str
                        Path(pth).mkdir(parents=True, exist_ok=True)
                        full_path = pth+"/event_"+str(self.cap_counter)+".avi"
                        self.writer = cv2.VideoWriter(
                            full_path, cv2.VideoWriter_fourcc(*"X264"),
                            self.framerate, (frameOrig.shape[1], frameOrig.shape[0])
                        )
                        print("Saving motion event video to: {}".format(full_path))
                        for i, motionFrame in enumerate(self.motion_event_frames):
                            self.writer.write(motionFrame)
                            # cv2.imwrite(pth + "/" + str(i) + ".png", motionFrame)
                        self.cap_counter += 1
                        self.in_motion_event = False
                        self.motion_event_frame_counter = 0
                        self.post_event_frame_counter = 0
                        self.motion_event_frames.clear()
                        self.writer.release()
            else:  # Not in motion event
                if text == "Motion":
                    self.motion_event_frame_counter += 1
                    if self.motion_event_frame_counter > 5:
                        self.in_motion_event = True
                        self.motion_event_frames = list(self.frame_history)
                        self.motion_event_frames.append(frameOrig)
                # else:  # No motion. No else, do nothing
            self.frame_history.append(frameOrig)
            self.error_count = 0
        else:
            if self.error_count > 5:
                print("Cannot read video file/stream. Shutting it down")
                self.end_video_feed()
            else:
                self.error_count += 1
                print("Cannot read video file/stream.")

    def end_video_feed(self):
        print("Releasing video file/stream")
        self.vcap.release()
        print("Destroying all cv2 windows")
        cv2.destroyAllWindows()

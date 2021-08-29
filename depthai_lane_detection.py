import threading
import time
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import pafy
import urllib.request

from lane_detection_utils import ModelType, ModelConfig, draw_lanes

def create_pipeline(model_path):
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if use_camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(800, 288)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)
        controlIn = pipeline.createXLinkIn()
        controlIn.setStreamName('control')
        controlIn.out.link(cam.inputControl)

    # NeuralNetwork
    print("Creating Lane Detection Neural Network...")
    lane_nn = pipeline.createNeuralNetwork()
    lane_nn.setBlobPath(model_path)

    # Increase threads for detection
    lane_nn.setNumInferenceThreads(2)

    # Specify that network takes latest arriving frame in non-blocking manner
    lane_nn.input.setQueueSize(1)
    lane_nn.input.setBlocking(False)
    lane_nn_xout = pipeline.createXLinkOut()
    lane_nn_xout.setStreamName("lane_nn")
    lane_nn.out.link(lane_nn_xout.input)

    if use_camera:
        cam.preview.link(lane_nn.input)
    else:
        lane_in = pipeline.createXLinkIn()
        lane_in.setStreamName("lane_in")
        lane_in.out.link(lane_nn.input)

    print("Pipeline created.")
    return pipeline

class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not use_camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

def get_frame():
    if use_camera:
        return True, np.array(cam_out.get().getData()).reshape((3, 288, 800)).transpose(1, 2, 0).astype(np.uint8)
    else:
        return cap.read()

if __name__ == '__main__':

    use_camera = True
    videoUrl = 'https://youtu.be/2CIxM7x-Clc'
    videoPafy = pafy.new(videoUrl)
    print(videoPafy.streams)
    video_path = videoPafy.streams[-1].url

    model_path = "models/ultra_falst_lane_detection_tusimple_288x800.blob"
    model_type = ModelType.TUSIMPLE
    model_cfg = ModelConfig(model_type)

    if use_camera:
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(video_path)
        fps = FPSHandler(cap)      
            

    pipeline = create_pipeline(model_path)

    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        if use_camera:
            cam_out = device.getOutputQueue("cam_out", 1, True)
            controlQueue = device.getInputQueue('control')
        else:
            lane_in = device.getInputQueue("lane_in")

        lane_nn = device.getOutputQueue("lane_nn", 1, False)

        # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
        frame = None
        detections = []

        # Main host-side application loop
        while True:
            # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
            ret, input_img = get_frame()
            output_img = input_img.copy()
            # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # Input values should be from -1 to 1 with a size of 288 x 800 pixels
            input_img = cv2.resize(input_img, (800,288))
            cv2.imshow("input", input_img)
            
            # Scale input pixel values to -1 to 1
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]

            input_img = ((input_img.astype(np.float32)/ 255.0 - mean[::-1]) / std[::-1]).astype(np.float32)
            input_img = input_img[np.newaxis,:,:,:]      

            
            if not use_camera:
                nn_data = dai.NNData()
                nn_data.setLayer("input", input_img)
                lane_in.send(nn_data)

            in_nn = lane_nn.get()

            output = np.array(in_nn.getLayerFp16('200')).reshape((101, 56, 4))
            
            fps.next_iter()

            lane_img = draw_lanes(output_img, output, model_cfg)
            cv2.putText(lane_img, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0))

            cv2.imshow("Detected lanes", lane_img)
            cv2.imwrite("output.jpg", lane_img)

            # Press key q to stop
            if cv2.waitKey(1) == ord('q'):
                break

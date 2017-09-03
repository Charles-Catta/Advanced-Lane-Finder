import cv2
import numpy as np
import glob
from Camera import LaneCamera
from ChessBoard import ChessBoard
from LaneIsolatorGUI import LaneIsolatorGUI
from matplotlib import pyplot as plt

if __name__ == "__main__":
    lane_viewer = LaneCamera("project_video.mp4")
    lane_viewer.load_camera_calibration("camera_cal/calibration_data.npy")
    test_imgs = glob.glob("test_images/*.jpg")
    view = lane_viewer.read_img(test_imgs[0])
    view = lane_viewer.birds_eye_view(view)
    lane_detect = LaneIsolatorGUI(view)
